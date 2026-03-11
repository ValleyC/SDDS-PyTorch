"""
SDDS Chip Placement Training Script

Corrected SDDS with:
- True net-level WA-HPWL energy
- 14D node features (sizes + Laplacian PE + centrality)
- Topology-first synthetic data
- No legalization inside SDDS — downstream CP-SAT handles legality
- Per-step energy (SDDS innovation)

Usage:
    python train_sdds.py --epochs 1000 --eval_every 50
    python train_sdds.py --epochs 500 --benchmark_base path/to/iccad04
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import torch
import torch.optim as optim
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archive', 'original_sdds'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ChipSAT'))

from sdds_step_model import SDDSStepModel
from sdds_data import SDDSDataset, SDDSCircuit, load_synthetic_circuit, load_benchmark_sdds
from sdds_energy import make_sdds_energy_fn, make_batched_energy_fn
from sdds_trajectory import collect_sdds_trajectory, sample_sdds, _expand_graph_for_basis
from noise_schedule import GaussianNoiseSchedule


# ---- Inlined PPO utilities (from archive/superseded_approaches/ppo_trainer.py) ----

class MovingAverage:
    """Moving average for reward normalization."""
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.step_count = 0
        self.mean_value = 0.0
        self.std_value = 0.0

    def update_mov_averages(self, data):
        if self.alpha == -1:
            return 0.0, 1.0
        mean_data = data.mean().item()
        std_data = data.std(unbiased=False).item()
        if self.step_count == 0:
            self.mean_value = mean_data
            self.std_value = std_data
        else:
            self.mean_value = self.alpha * mean_data + (1 - self.alpha) * self.mean_value
            self.std_value = self.beta * std_data + (1 - self.beta) * self.std_value
        self.step_count += 1
        return self.mean_value, self.std_value

    def calculate_average(self, rewards, mov_average_reward, mov_std_reward):
        return (rewards - mov_average_reward) / (mov_std_reward + 1e-10)


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """Compute GAE (Generalized Advantage Estimation) - TD(lambda)."""
    n_steps = rewards.size(0)
    advantage = torch.zeros_like(rewards)
    for t in range(n_steps):
        idx = n_steps - t - 1
        next_value = values[idx + 1] if values.size(0) > idx + 1 else torch.zeros_like(values[idx])
        delta = rewards[idx] + gamma * next_value - values[idx]
        next_advantage = advantage[idx + 1] if idx < n_steps - 1 else torch.zeros_like(delta)
        advantage[idx] = delta + gamma * lam * next_advantage
    value_target = advantage + values[:n_steps]
    return value_target, advantage


def normalize_advantages(advantages, exclude_last=True):
    """Normalize advantages."""
    if exclude_last and advantages.shape[2] > 1:
        unpadded_adv = advantages[:, :, :-1]
        mean = unpadded_adv.mean()
        std = unpadded_adv.std(unbiased=False)
    else:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
    return (advantages - mean) / (std + 1e-10)


class SDDSTrainer:
    """SDDS chip placement trainer. Trains on one circuit per epoch (mixed sampling)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_message_passes: int = 5,
        n_diffusion_steps: int = 50,
        n_basis_states: int = 8,
        T_temperature: float = 0.5,
        lr: float = 1e-4,
        hpwl_gamma: float = 50.0,
        overlap_weight: float = 10.0,
        overlap_weight_final: float = 100.0,
        overlap_ramp_epochs: int = 100,
        overlap_margin: float = 0.1,
        boundary_weight: float = 5.0,
        ppo_clip: float = 0.2,
        ppo_inner_steps: int = 2,
        value_weight: float = 0.65,
        grad_clip: float = 1.0,
        TD_k: float = 3.0,
        mov_avg_alpha: float = 0.0009,
        device: str = 'cuda',
    ):
        self.n_diffusion_steps = n_diffusion_steps
        self.n_basis_states = n_basis_states
        self.T_temperature = T_temperature
        self.hpwl_gamma = hpwl_gamma
        self.overlap_weight_init = overlap_weight
        self.overlap_weight_final = overlap_weight_final
        self.overlap_ramp_epochs = overlap_ramp_epochs
        self.overlap_margin = overlap_margin
        self.overlap_weight = overlap_weight  # current (updated per epoch)
        self.boundary_weight = boundary_weight
        self.ppo_clip = ppo_clip
        self.ppo_inner_steps = ppo_inner_steps
        self.value_weight = value_weight
        self.grad_clip = grad_clip
        self.device = device

        self.model = SDDSStepModel(
            hidden_dim=hidden_dim,
            n_message_passes=n_message_passes,
            n_diffusion_steps=n_diffusion_steps,
        ).to(device)

        self.noise_schedule = GaussianNoiseSchedule(n_diffusion_steps)
        self.noise_schedule.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.moving_avg = MovingAverage(mov_avg_alpha, mov_avg_alpha)

        self.gamma = 1.0
        self.lam = np.exp(-np.log(TD_k) / n_diffusion_steps)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"SDDSStepModel: {n_params:,} params, device={device}")
        print(f"  Overlap: weight {overlap_weight}->{overlap_weight_final} over {overlap_ramp_epochs} epochs, margin={overlap_margin}")

    def update_overlap_weight(self, epoch: int):
        """Curriculum: linearly ramp overlap weight from init to final.

        epoch=1 → init weight, epoch=ramp_epochs → final weight.
        """
        if epoch >= self.overlap_ramp_epochs:
            self.overlap_weight = self.overlap_weight_final
        else:
            frac = (epoch - 1) / max(1, self.overlap_ramp_epochs - 1)
            self.overlap_weight = (
                self.overlap_weight_init + frac * (self.overlap_weight_final - self.overlap_weight_init)
            )

    def train_batch(self, circuits: List[SDDSCircuit], step: int) -> dict:
        """One PPO training step on a batch of circuits (n_graphs >= 1)."""
        self.model.train()

        n_graphs = len(circuits)

        # Build batched graph
        all_nf, all_sizes, all_ei_src, all_ei_dst, all_ea, all_ngi = [], [], [], [], [], []
        circuit_net_tensors = []
        circuit_sizes = []
        circuit_node_offsets = []
        node_offset = 0

        for g, circ in enumerate(circuits):
            V = circ.n_components
            all_nf.append(circ.node_features)
            all_sizes.append(circ.sizes)
            all_ei_src.append(circ.edge_index[0] + node_offset)
            all_ei_dst.append(circ.edge_index[1] + node_offset)
            all_ea.append(circ.edge_attr)
            all_ngi.append(torch.full((V,), g, dtype=torch.long))
            nt = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in circ.net_tensors.items()}
            circuit_net_tensors.append(nt)
            circuit_sizes.append(circ.sizes.to(self.device))
            circuit_node_offsets.append(node_offset)
            node_offset += V

        nf = torch.cat(all_nf).to(self.device)
        sizes = torch.cat(all_sizes).to(self.device)
        ei = torch.stack([torch.cat(all_ei_src), torch.cat(all_ei_dst)]).to(self.device)
        ea = torch.cat(all_ea).to(self.device)
        ngi = torch.cat(all_ngi).to(self.device)

        # Build batched energy function
        energy_fn = make_batched_energy_fn(
            circuit_sizes, circuit_net_tensors, circuit_node_offsets,
            ngi, n_graphs,
            self.hpwl_gamma, self.overlap_weight, self.boundary_weight,
            overlap_margin=self.overlap_margin,
        )

        # Collect trajectory
        with torch.no_grad():
            buffer = collect_sdds_trajectory(
                self.model, self.noise_schedule,
                ei, ea, ngi, n_graphs,
                self.n_basis_states, self.T_temperature,
                nf, energy_fn, device=self.device,
            )

        # Compute advantages
        mov_mean, mov_std = self.moving_avg.update_mov_averages(buffer.rewards)
        normed_rewards = self.moving_avg.calculate_average(buffer.rewards, mov_mean, mov_std)

        values_boot = torch.cat([
            buffer.values,
            torch.zeros(1, *buffer.values.shape[1:], device=self.device),
        ], dim=0)
        value_targets, advantages = compute_gae(normed_rewards, values_boot, self.gamma, self.lam)
        normed_adv = normalize_advantages(advantages)

        # PPO updates
        loss_info = self._ppo_update(
            buffer, normed_adv, value_targets,
            ei, ea, ngi, n_graphs, nf,
        )

        # Add trajectory stats
        loss_info['noise_reward'] = buffer.noise_rewards.sum().item()
        loss_info['entropy_reward'] = buffer.entropy_rewards.sum().item()
        loss_info['energy_reward_final'] = buffer.energy_rewards.mean().item()
        loss_info['reward_mean'] = buffer.rewards.mean().item()
        loss_info['reward_std'] = buffer.rewards.std().item()
        loss_info['final_energy'] = -buffer.energy_rewards.mean().item()
        loss_info['n_graphs'] = n_graphs
        loss_info['total_nodes'] = nf.shape[0]

        return loss_info

    def _ppo_update(
        self, buffer, advantages, value_targets,
        edge_index, edge_attr, node_graph_idx, n_graphs, node_features,
    ) -> dict:
        """Run PPO inner loop."""
        T = self.n_diffusion_steps
        n_basis = self.n_basis_states
        V = buffer.states.shape[1]
        continuous_dim = buffer.states.shape[3]

        # Expanded graph for PPO replay
        (exp_ei, exp_ea, exp_ngi, exp_nf, exp_ng) = _expand_graph_for_basis(
            edge_index, edge_attr, node_graph_idx, node_features,
            V, n_graphs, n_basis,
        )

        total_actor = 0.0
        total_critic = 0.0
        all_ratios = []
        n_updates = 0

        for _ in range(self.ppo_inner_steps):
            perm_t = torch.randperm(T, device=self.device)
            perm_b = torch.randperm(n_basis, device=self.device)

            for t_idx in range(T):
                t = perm_t[t_idx].item()

                # Flatten basis for batched forward
                X_t_flat = buffer.states[t, :, perm_b].transpose(0, 1).reshape(-1, continuous_dim)
                X_next_flat = buffer.actions[t, :, perm_b].transpose(0, 1).reshape(-1, continuous_dim)
                rand_flat = buffer.rand_node_features[t, :, perm_b].transpose(0, 1).reshape(
                    -1, buffer.rand_node_features.shape[-1]
                )

                step_idx = buffer.time_index_per_node[t, 0, 0].item()

                mean_flat, log_var_flat, vals_flat, _ = self.model(
                    X_t_flat, step_idx, exp_ei, exp_ea, exp_ngi, exp_ng, exp_nf,
                    rand_nodes=rand_flat,
                )

                new_lp = self.model.get_state_log_prob(
                    mean_flat, log_var_flat, X_next_flat, exp_ngi, exp_ng,
                )

                new_lp_g = new_lp.reshape(n_basis, n_graphs).t()
                vals_g = vals_flat.reshape(n_basis, n_graphs).t()
                old_lp = buffer.policies[t, :, perm_b]
                adv = advantages[t, :, perm_b]
                vtgt = value_targets[t, :, perm_b]

                ratios = torch.exp(new_lp_g - old_lp)
                all_ratios.append(ratios.reshape(-1).detach())

                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.functional.mse_loss(vals_g, vtgt)

                loss = (1 - self.value_weight) * actor_loss + self.value_weight * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_actor += actor_loss.item()
                total_critic += critic_loss.item()
                n_updates += 1

        all_ratios = torch.cat(all_ratios) if all_ratios else torch.tensor([1.0])
        return {
            'actor_loss': total_actor / max(n_updates, 1),
            'critic_loss': total_critic / max(n_updates, 1),
            'mean_ratio': all_ratios.mean().item(),
            'max_ratio': all_ratios.max().item(),
            'clip_frac': ((all_ratios < 1 - self.ppo_clip) | (all_ratios > 1 + self.ppo_clip)).float().mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, circuit: SDDSCircuit, n_basis: int = 8) -> dict:
        """
        Evaluate: SDDS sample -> pick best -> CP-SAT legalize -> short ALNS.
        """
        nf = circuit.node_features.to(self.device)
        sizes = circuit.sizes.to(self.device)
        ei = circuit.edge_index.to(self.device)
        ea = circuit.edge_attr.to(self.device)
        ngi = torch.zeros(circuit.n_components, dtype=torch.long, device=self.device)

        # Sample
        X_0 = sample_sdds(
            self.model, self.noise_schedule,
            ei, ea, ngi, 1, n_basis, nf,
            T_temperature=0.0, device=self.device,
        )  # (V, n_basis, 2)

        # Pick best by energy
        nt = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
              for k, v in circuit.net_tensors.items()}
        energy_fn = make_sdds_energy_fn(
            sizes, nt, ngi, 1, self.hpwl_gamma, self.overlap_weight, self.boundary_weight,
            overlap_margin=self.overlap_margin,
        )

        best_e = float('inf')
        best_pos = None
        for b in range(n_basis):
            pos_b = X_0[:, b, :]
            e = energy_fn(pos_b, ngi, 1).item()
            if e < best_e:
                best_e = e
                best_pos = pos_b.cpu().numpy()

        # Compute overlap metrics on raw continuous positions
        from sdds_energy import compute_overlap_penalty
        pos_t = torch.from_numpy(best_pos).float().to(self.device)
        overlap_area = compute_overlap_penalty(pos_t, sizes, ngi, 1, margin=0.0).item()

        # Count overlapping pairs
        half = sizes / 2.0
        x_min = pos_t[:, 0] - half[:, 0]
        y_min = pos_t[:, 1] - half[:, 1]
        x_max = pos_t[:, 0] + half[:, 0]
        y_max = pos_t[:, 1] + half[:, 1]
        ow = torch.clamp(torch.minimum(x_max.unsqueeze(1), x_max.unsqueeze(0))
                         - torch.maximum(x_min.unsqueeze(1), x_min.unsqueeze(0)), min=0)
        oh = torch.clamp(torch.minimum(y_max.unsqueeze(1), y_max.unsqueeze(0))
                         - torch.maximum(y_min.unsqueeze(1), y_min.unsqueeze(0)), min=0)
        overlap_pairs = int(((ow * oh) > 1e-8).triu(diagonal=1).sum().item())

        result = {
            'best_energy': best_e,
            'n_components': circuit.n_components,
            'circuit_name': circuit.circuit_name,
            'overlap_area': overlap_area,
            'overlap_pairs': overlap_pairs,
        }

        # Pre-legalization HPWL
        try:
            from cpsat_solver import compute_net_hpwl, legalize
            from lns_solver import LNSSolver

            sizes_np = circuit.sizes.cpu().numpy()
            pre_hpwl = compute_net_hpwl(best_pos, sizes_np, circuit.nets)
            result['pre_legal_hpwl'] = pre_hpwl

            # CP-SAT legalize
            legal_pos = None
            for window in [0.3, 0.5, 1.0]:
                legal_pos = legalize(best_pos, sizes_np, time_limit=30.0, window_fraction=window)
                if legal_pos is not None:
                    result['legal_window'] = window
                    break

            if legal_pos is not None:
                post_hpwl = compute_net_hpwl(legal_pos, sizes_np, circuit.nets)
                result['post_legal_hpwl'] = post_hpwl
                result['distortion'] = (post_hpwl - pre_hpwl) / max(1e-6, pre_hpwl)

                # Short ALNS
                edge_index_np = circuit.edge_index.cpu().numpy()
                edge_attr_np = circuit.edge_attr.cpu().numpy()
                solver = LNSSolver(
                    legal_pos, sizes_np, circuit.nets, edge_index_np,
                    congestion_weight=0.0, subset_size=min(30, circuit.n_components),
                    window_fraction=0.15, cpsat_time_limit=5.0,
                )
                lns_result = solver.solve(n_iterations=50, log_every=25)
                result['post_alns_hpwl'] = lns_result['best_hpwl']
            else:
                result['post_legal_hpwl'] = None
                result['legalization_failed'] = True

        except ImportError as e:
            result['eval_error'] = str(e)

        return result


def main():
    parser = argparse.ArgumentParser(description='SDDS Chip Placement Training')
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_message_passes', type=int, default=5)
    parser.add_argument('--n_diffusion_steps', type=int, default=50)
    parser.add_argument('--n_basis_states', type=int, default=8)
    parser.add_argument('--T_temperature', type=float, default=0.5)
    # Energy
    parser.add_argument('--hpwl_gamma', type=float, default=50.0)
    parser.add_argument('--overlap_weight', type=float, default=10.0,
                        help='Initial overlap weight (curriculum start)')
    parser.add_argument('--overlap_weight_final', type=float, default=100.0,
                        help='Final overlap weight (curriculum end)')
    parser.add_argument('--overlap_ramp_epochs', type=int, default=100,
                        help='Epochs to linearly ramp overlap weight')
    parser.add_argument('--overlap_margin', type=float, default=0.1,
                        help='Cell inflation margin (0.1 = 10%% larger during training)')
    parser.add_argument('--boundary_weight', type=float, default=5.0)
    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--ppo_inner_steps', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of circuits per PPO step (DIffUCO default: 16)')
    # Data
    parser.add_argument('--synthetic_ratio', type=float, default=1.0)
    parser.add_argument('--n_instances', type=int, default=200,
                        help='Number of training instances (pre-generated)')
    parser.add_argument('--v_min', type=int, default=30)
    parser.add_argument('--v_max', type=int, default=200)
    parser.add_argument('--benchmark_base', type=str, default=None)
    parser.add_argument('--eval_circuits', type=str, default=None,
                        help='Comma-separated circuit names for eval')
    # Output
    parser.add_argument('--save_dir', type=str, default='checkpoints_sdds')
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create trainer
    trainer = SDDSTrainer(
        hidden_dim=args.hidden_dim,
        n_message_passes=args.n_message_passes,
        n_diffusion_steps=args.n_diffusion_steps,
        n_basis_states=args.n_basis_states,
        T_temperature=args.T_temperature,
        lr=args.lr,
        hpwl_gamma=args.hpwl_gamma,
        overlap_weight=args.overlap_weight,
        overlap_weight_final=args.overlap_weight_final,
        overlap_ramp_epochs=args.overlap_ramp_epochs,
        overlap_margin=args.overlap_margin,
        boundary_weight=args.boundary_weight,
        ppo_inner_steps=args.ppo_inner_steps,
        grad_clip=args.grad_clip,
        device=device,
    )

    # Load benchmark circuits for evaluation
    eval_circuits = []
    if args.benchmark_base and args.eval_circuits:
        for name in args.eval_circuits.split(','):
            name = name.strip()
            circuit_dir = os.path.join(args.benchmark_base, name)
            if os.path.exists(circuit_dir):
                circ = load_benchmark_sdds(circuit_dir, name, macros_only=True, device='cpu')
                eval_circuits.append(circ)
                print(f"Loaded eval circuit: {name} ({circ.n_components} macros)")

    # Load benchmark circuits for training
    benchmark_circuits = []
    if args.benchmark_base and args.synthetic_ratio < 1.0:
        # Load all ibm circuits for training
        for name in ['ibm01', 'ibm02', 'ibm03', 'ibm04', 'ibm06', 'ibm07', 'ibm08', 'ibm09']:
            circuit_dir = os.path.join(args.benchmark_base, name)
            if os.path.exists(circuit_dir):
                circ = load_benchmark_sdds(circuit_dir, name, macros_only=True, device='cpu')
                benchmark_circuits.append(circ)

    # Pre-generate fixed dataset (DIffUCO paradigm)
    dataset = SDDSDataset(
        n_synthetic=args.n_instances,
        synthetic_ratio=args.synthetic_ratio,
        v_range=(args.v_min, args.v_max),
        benchmark_circuits=benchmark_circuits,
        device='cpu',
    )

    # Training loop — each epoch iterates through ALL instances (shuffled)
    os.makedirs(args.save_dir, exist_ok=True)
    history = []
    best_eval = float('inf')
    global_step = 0

    n_batches_per_epoch = max(1, len(dataset) // args.batch_size)
    print(f"\nTraining for {args.epochs} epochs x {n_batches_per_epoch} batches/epoch "
          f"(batch_size={args.batch_size}, {len(dataset)} instances)")
    print(f"= {args.epochs * n_batches_per_epoch} total PPO steps")
    print()

    for epoch in range(1, args.epochs + 1):
        t0_epoch = time.time()
        epoch_losses = []

        # Curriculum: ramp up overlap weight
        trainer.update_overlap_weight(epoch)

        # Shuffled order for this epoch, then batch contiguous items
        order = dataset.get_epoch_order(epoch)

        for batch_start in range(0, len(order), args.batch_size):
            batch_indices = order[batch_start:batch_start + args.batch_size]
            batch_circuits = [dataset[idx] for idx in batch_indices]
            loss_info = trainer.train_batch(batch_circuits, global_step)
            global_step += 1
            epoch_losses.append(loss_info)

        dt_epoch = time.time() - t0_epoch

        # Epoch summary
        avg_actor = np.mean([l['actor_loss'] for l in epoch_losses])
        avg_critic = np.mean([l['critic_loss'] for l in epoch_losses])
        avg_ratio = np.mean([l['mean_ratio'] for l in epoch_losses])
        avg_clip = np.mean([l['clip_frac'] for l in epoch_losses])
        avg_energy = np.mean([l['final_energy'] for l in epoch_losses])

        history.append({
            'epoch': epoch,
            'n_instances': len(order),
            'avg_actor_loss': avg_actor,
            'avg_critic_loss': avg_critic,
            'avg_ratio': avg_ratio,
            'avg_clip_frac': avg_clip,
            'avg_energy': avg_energy,
            'time': dt_epoch,
        })

        print(f"[Epoch {epoch:3d}] {len(order)} instances | "
              f"actor={avg_actor:.4f} critic={avg_critic:.4f} "
              f"ratio={avg_ratio:.3f} clip={avg_clip:.3f} "
              f"energy={avg_energy:.1f} ov_w={trainer.overlap_weight:.1f} | {dt_epoch:.1f}s")

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == 1:
            print(f"\n--- Evaluation (epoch {epoch}) ---")

            # Eval on a held-out synthetic circuit (not in training set)
            eval_synth = load_synthetic_circuit(seed=99999, v_range=(50, 100), device='cpu')
            res = trainer.evaluate(eval_synth, n_basis=8)
            pre = res.get('pre_legal_hpwl', '?')
            post = res.get('post_legal_hpwl', '?')
            alns = res.get('post_alns_hpwl', '?')
            ovlp = res.get('overlap_pairs', '?')
            ov_area = res.get('overlap_area', 0)
            print(f"  Synthetic ({res['circuit_name']}, V={res['n_components']}): "
                  f"pre={pre}, post={post}, alns={alns} | "
                  f"overlaps={ovlp} area={ov_area:.4f}")

            # Track best from synthetic when no benchmark eval circuits
            if not eval_circuits and res.get('post_legal_hpwl') is not None:
                if res['post_legal_hpwl'] < best_eval:
                    best_eval = res['post_legal_hpwl']
                    torch.save(trainer.model.state_dict(),
                               os.path.join(args.save_dir, 'best_model.pt'))
                    print(f"  -> New best! Saved to {args.save_dir}/best_model.pt")

            # Eval on benchmark circuits
            for circ in eval_circuits:
                res = trainer.evaluate(circ, n_basis=8)
                pre = res.get('pre_legal_hpwl', '?')
                post = res.get('post_legal_hpwl', '?')
                alns = res.get('post_alns_hpwl', '?')
                ovlp = res.get('overlap_pairs', '?')
                ov_area = res.get('overlap_area', 0)
                print(f"  {res['circuit_name']} (V={res['n_components']}): "
                      f"pre={pre}, post={post}, alns={alns} | "
                      f"overlaps={ovlp} area={ov_area:.4f}")

                if res.get('post_legal_hpwl') is not None:
                    if res['post_legal_hpwl'] < best_eval:
                        best_eval = res['post_legal_hpwl']
                        torch.save(trainer.model.state_dict(),
                                   os.path.join(args.save_dir, 'best_model.pt'))
                        print(f"  -> New best! Saved to {args.save_dir}/best_model.pt")

            print()

        # Periodic save
        if epoch % 100 == 0:
            torch.save(trainer.model.state_dict(),
                       os.path.join(args.save_dir, f'model_epoch{epoch}.pt'))

    # Final save
    torch.save(trainer.model.state_dict(),
               os.path.join(args.save_dir, 'final_model.pt'))

    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTraining complete. Best eval HPWL: {best_eval}")
    print(f"Models saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
