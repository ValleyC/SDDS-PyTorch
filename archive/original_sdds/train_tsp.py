"""
TSP Training Script - DiffUCO-style Categorical Diffusion with PPO

This script matches DiffUCO's training setup exactly:
- Temperature annealing (N_warmup + N_anneal + N_equil epochs)
- Cosine learning rate schedule
- PPO with TD(lambda), moving average reward normalization
- Same metrics: mean_energy, rel_error, best_energy, etc.
- Wandb logging support

Usage:
    # Quick test
    python train_tsp.py --n_nodes 20 --n_graphs 32 --N_anneal 500

    # Full training (matching DiffUCO defaults)
    python train_tsp.py --n_nodes 20 --n_graphs 30 --N_anneal 2000 --T_max 0.05

    # With wandb logging
    python train_tsp.py --wandb --project_name "TSP_Diffusion"

Reference: DIffUCO/train.py, DIffUCO/argparse_ray_main.py
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import os
import math

from noise_schedule import CategoricalNoiseSchedule
from step_model import DiffusionStepModel, TSPDiffusionStepModel, scatter_sum
from trajectory import collect_trajectory, sample_with_eval_step_factor
from ppo_trainer import PPOTrainer, MovingAverage
from tsp_energy import TSPEnergy, compute_tour_length_only, check_tour_validity

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def cosine_schedule(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Cosine learning rate schedule matching DiffUCO's cos_schedule.

    From utils/lr_schedule.py
    """
    if total_steps == 0:
        return max_lr
    progress = min(step / total_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def linear_annealing(epoch: int, N_warmup: int, N_anneal: int, N_equil: int,
                     T_max: float, T_target: float) -> float:
    """
    Linear temperature annealing matching DiffUCO's __linear_annealing.

    From train.py lines 545-553
    """
    total_epochs = N_warmup + N_anneal + N_equil

    if epoch < N_warmup:
        T_curr = T_max
    elif epoch >= N_warmup and epoch < total_epochs - N_equil - 1:
        T_curr = max(T_max - T_target - (T_max - T_target) * (epoch - N_warmup) / N_anneal, 0) + T_target
    else:
        T_curr = T_target

    return T_curr


def generate_tsp_instance(n_nodes: int, n_graphs: int, device: torch.device):
    """
    Generate random TSP instances.

    Returns:
        positions: Node coordinates (n_graphs * n_nodes, 2)
        edge_index: Fully connected graph edges (2, n_edges)
        edge_attr: Edge distances (n_edges, 1)
        node_graph_idx: Graph assignment per node
    """
    total_nodes = n_graphs * n_nodes

    # Random 2D positions in [0, 1]^2
    positions = torch.rand(total_nodes, 2, device=device)

    # Node-to-graph mapping
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs, device=device),
        n_nodes
    )

    # Build fully connected edges within each graph
    senders = []
    receivers = []

    for g in range(n_graphs):
        offset = g * n_nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    senders.append(offset + i)
                    receivers.append(offset + j)

    edge_index = torch.tensor([senders, receivers], dtype=torch.long, device=device)

    # Edge attributes: Euclidean distances
    pos_senders = positions[edge_index[0]]
    pos_receivers = positions[edge_index[1]]
    distances = torch.sqrt(((pos_senders - pos_receivers) ** 2).sum(dim=-1, keepdim=True) + 1e-10)
    edge_attr = distances

    return positions, edge_index, edge_attr, node_graph_idx


def evaluate_solutions(
    model: DiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_nodes: int,
    n_basis_states: int = 8,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
):
    """
    Generate solutions and evaluate tour lengths.

    Matches DiffUCO's evaluation_step metrics.

    Args:
        eval_step_factor: Factor to multiply diffusion steps during eval
                         (matches DiffUCO's eval_step_factor)

    Returns both:
    - energy: tour_length + penalties (for invalid tours)
    - tour_cost: pure tour length (always, even for invalid tours)
    """
    model.eval()
    device = positions.device

    # Use sample_with_eval_step_factor for proper DiffUCO-style evaluation
    X_final_all = sample_with_eval_step_factor(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        eval_step_factor=eval_step_factor,
        device=device,
    )

    # X_final_all shape: (n_nodes_total, n_basis_states)
    # Reshape to (n_basis_states, n_graphs, n_nodes_per_graph)
    X_final_all = X_final_all.T.reshape(n_basis_states, n_graphs, n_nodes)

    all_energies = []      # Energy with penalties
    all_tour_costs = []    # Pure tour length (TSP cost)
    all_valid = []
    all_violations = []

    tsp_energy_fn = TSPEnergy(n_nodes, penalty_coeff=1.45)

    with torch.no_grad():
        for b in range(n_basis_states):
            for g in range(n_graphs):
                pos_g = positions[g * n_nodes:(g + 1) * n_nodes]
                X_g = X_final_all[b, g]

                is_valid, info = check_tour_validity(X_g, n_nodes)
                all_valid.append(is_valid)
                all_violations.append(info['position_violations'])

                # Always compute pure tour length (TSP cost)
                tour_cost = compute_tour_length_only(pos_g, X_g)
                all_tour_costs.append(tour_cost.item())

                # Compute energy (tour + penalties)
                energy, _, _ = tsp_energy_fn.compute_energy_from_positions(
                    pos_g, X_g,
                    torch.zeros(n_nodes, dtype=torch.long, device=device),
                    n_graphs=1
                )
                all_energies.append(energy[0].item())

    model.train()

    energies = np.array(all_energies)
    tour_costs = np.array(all_tour_costs)
    energies_per_graph = energies.reshape(n_basis_states, n_graphs)
    tour_costs_per_graph = tour_costs.reshape(n_basis_states, n_graphs)

    # Metrics matching DiffUCO (using energy)
    mean_energy = np.mean(energies)
    min_energy_per_graph = np.min(energies_per_graph, axis=0)
    best_energy = np.mean(min_energy_per_graph)

    # Pure tour cost metrics (TSP objective)
    mean_tour_cost = np.mean(tour_costs)
    min_tour_cost_per_graph = np.min(tour_costs_per_graph, axis=0)
    best_tour_cost = np.mean(min_tour_cost_per_graph)

    # Best valid tour cost (only from valid solutions)
    valid_mask = np.array(all_valid).reshape(n_basis_states, n_graphs)
    if valid_mask.any():
        valid_tour_costs = tour_costs_per_graph.copy()
        valid_tour_costs[~valid_mask] = np.inf
        best_valid_tour_cost = np.mean(np.min(valid_tour_costs, axis=0))
    else:
        best_valid_tour_cost = np.nan

    valid_ratio = np.mean(all_valid)
    mean_violations = np.mean(all_violations)

    return {
        'mean_energy': mean_energy,
        'best_energy': best_energy,
        'min_energy_per_graph': min_energy_per_graph.tolist(),
        'mean_tour_cost': mean_tour_cost,
        'best_tour_cost': best_tour_cost,
        'best_valid_tour_cost': best_valid_tour_cost,
        'valid_ratio': valid_ratio,
        'mean_violations': mean_violations,
        'all_tour_lengths': tour_costs_per_graph,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Model for TSP (DiffUCO-style)')

    # Problem parameters
    parser.add_argument('--n_nodes', type=int, default=20, help='Number of nodes per TSP instance')
    parser.add_argument('--n_graphs', type=int, default=30, help='Batch size (H in DiffUCO)')

    # Model parameters (matching DiffUCO defaults)
    parser.add_argument('--n_hidden', type=int, default=120, help='Hidden dimension (DiffUCO uses 120 for TSP)')
    parser.add_argument('--n_message_passes', type=int, default=8, help='Number of GNN layers')
    parser.add_argument('--n_diffusion_steps', type=int, default=9, help='Number of diffusion steps')
    parser.add_argument('--n_random_features', type=int, default=5, help='Random node features')
    parser.add_argument('--time_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'one_hot'])
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads (TSP GAT)')

    # Temperature annealing (DiffUCO style)
    # NOTE: DiffUCO uses T_max around 1.0 for proper exploration, not 0.05!
    parser.add_argument('--T_max', type=float, default=1.0, help='Initial/max temperature (use 1.0 for exploration)')
    parser.add_argument('--T_target', type=float, default=0.01, help='Target temperature')
    parser.add_argument('--N_warmup', type=int, default=0, help='Warmup epochs at T_max')
    parser.add_argument('--N_anneal', type=int, default=2000, help='Annealing epochs')
    parser.add_argument('--N_equil', type=int, default=0, help='Equilibration epochs at T_target')

    # Training parameters (DiffUCO PPO defaults)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'constant'])
    parser.add_argument('--n_basis_states', type=int, default=10, help='Number of parallel trajectories (n_s)')
    parser.add_argument('--TD_k', type=float, default=3.0, help='TD(lambda) parameter')
    parser.add_argument('--clip_value', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_weighting', type=float, default=0.65, help='Critic loss weight (c1)')
    parser.add_argument('--inner_loop_steps', type=int, default=4, help='PPO inner loop steps')
    parser.add_argument('--mov_average', type=float, default=0.0009, help='Moving average alpha')

    # Evaluation
    parser.add_argument('--n_test_basis_states', type=int, default=8, help='Basis states for evaluation')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--stop_epochs', type=int, default=10000, help='Early stopping patience')
    parser.add_argument('--eval_step_factor', type=int, default=1,
                        help='Factor to multiply diffusion steps during eval (DiffUCO-style)')

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='TSP_Diffusion', help='Wandb project name')

    # Other
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Total epochs
    total_epochs = args.N_warmup + args.N_anneal + args.N_equil

    # n_classes = n_nodes for TSP
    n_classes = args.n_nodes

    print(f"\n{'='*70}")
    print(f"TSP Diffusion Training (DiffUCO-style)")
    print(f"{'='*70}")
    print(f"Problem:")
    print(f"  n_nodes: {args.n_nodes}")
    print(f"  n_graphs (batch H): {args.n_graphs}")
    print(f"  n_basis_states (n_s): {args.n_basis_states}")
    print(f"\nModel:")
    print(f"  hidden_dim: {args.n_hidden}")
    print(f"  n_message_passes: {args.n_message_passes}")
    print(f"  n_diffusion_steps: {args.n_diffusion_steps}")
    print(f"\nTraining:")
    print(f"  total_epochs: {total_epochs} (warmup={args.N_warmup}, anneal={args.N_anneal}, equil={args.N_equil})")
    print(f"  T_max -> T_target: {args.T_max} -> {args.T_target}")
    print(f"  lr: {args.lr} ({args.lr_schedule} schedule)")
    print(f"  TD_k: {args.TD_k}, clip: {args.clip_value}, c1: {args.value_weighting}")
    print(f"{'='*70}\n")

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb_project = f"{args.project_name}_TSP{args.n_nodes}"
        wandb_group = f"seed{args.seed}_T{args.T_max}_anneal{args.N_anneal}"
        wandb_run = f"lr{args.lr}_nh{args.n_hidden}_diff{args.n_diffusion_steps}"

        wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=wandb_run,
            config=vars(args),
        )
        use_wandb = True
    else:
        use_wandb = False

    # Create model
    # Use simpler DiffusionStepModel (same as MaxCut) - much faster than GAT
    # TSPDiffusionStepModel with GAT is too slow for practical use
    model = DiffusionStepModel(
        n_classes=n_classes,
        edge_dim=1,
        hidden_dim=args.n_hidden,
        n_diffusion_steps=args.n_diffusion_steps,
        n_message_passes=args.n_message_passes,
        n_random_features=args.n_random_features,
        time_encoding=args.time_encoding,
        embedding_dim=32,
        mean_aggr=False,
    ).to(device)

    # Create noise schedule
    noise_schedule = CategoricalNoiseSchedule(
        n_steps=args.n_diffusion_steps,
        n_classes=n_classes,
        schedule='diffuco',
    )

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        noise_schedule=noise_schedule,
        lr=args.lr,
        gamma=1.0,
        TD_k=args.TD_k,
        clip_value=args.clip_value,
        value_weighting=args.value_weighting,
        inner_loop_steps=args.inner_loop_steps,
        mov_average_alpha=args.mov_average,
    )

    # TSP energy function
    tsp_energy = TSPEnergy(n_classes, penalty_coeff=1.45)

    # Training state
    best_energy = float('inf')
    best_rel_error = float('inf')
    epochs_since_best = 0
    global_step = 0

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Training loop
    epoch_range = tqdm(range(total_epochs), desc="Training")

    for epoch in epoch_range:
        start_time = time.time()

        # Temperature annealing
        T_curr = linear_annealing(
            epoch, args.N_warmup, args.N_anneal, args.N_equil,
            args.T_max, args.T_target
        )

        # Learning rate schedule
        if args.lr_schedule == 'cosine':
            lr_curr = cosine_schedule(
                global_step, total_epochs * args.inner_loop_steps,
                args.lr, args.lr / 10
            )
            # Update optimizer lr
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr_curr
        else:
            lr_curr = args.lr

        # Generate new TSP instances each epoch
        positions, edge_index, edge_attr, node_graph_idx = generate_tsp_instance(
            args.n_nodes, args.n_graphs, device
        )

        # Create energy function for this batch
        def energy_fn(X_0, node_idx, n_g):
            energy, violations, penalty = tsp_energy.compute_energy_from_positions(
                positions, X_0, node_idx, n_g
            )
            return energy

        # Training step
        loss_dict = trainer.train_step(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_graph_idx=node_graph_idx,
            n_graphs=args.n_graphs,
            n_basis_states=args.n_basis_states,
            T_temperature=T_curr,
            energy_fn=energy_fn,
        )

        global_step += args.inner_loop_steps
        epoch_time = time.time() - start_time

        # Logging
        log_dict = {
            'train/epoch': epoch,
            'train/overall_loss': loss_dict['overall_loss'],
            'train/actor_loss': loss_dict['actor_loss'],
            'train/critic_loss': loss_dict['critic_loss'],
            'train/clip_fraction': loss_dict['clip_fraction'],
            'train/mean_ratios': loss_dict['mean_ratios'],
            'train/noise_rewards': loss_dict['noise_reward_sum'],
            'train/entropy_rewards': loss_dict['entropy_reward_sum'],
            'train/energy_rewards': loss_dict['energy_reward_mean'],
            'schedules/T': T_curr,
            'schedules/lr': lr_curr,
            'schedules/time': epoch_time,
        }

        # Periodic logging
        if epoch % 10 == 0:
            tqdm.write(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss_dict['overall_loss']:.4f} | "
                f"Actor: {loss_dict['actor_loss']:.4f} | "
                f"Critic: {loss_dict['critic_loss']:.4f} | "
                f"T: {T_curr:.4f} | "
                f"AdvStd: {loss_dict['advantage_std']:.3f} | "
                f"Energy: {loss_dict['energy_reward_mean']:.2f}"
            )

        # Evaluation
        if (epoch + 1) % args.eval_every == 0 or epoch == total_epochs - 1:
            tqdm.write(f"\n--- Evaluation at epoch {epoch + 1} ---")

            # Generate test instances
            test_positions, test_edge_index, test_edge_attr, test_node_graph_idx = generate_tsp_instance(
                args.n_nodes, args.n_graphs, device
            )

            eval_results = evaluate_solutions(
                model, noise_schedule,
                test_positions, test_edge_index, test_edge_attr, test_node_graph_idx,
                args.n_graphs, args.n_nodes,
                n_basis_states=args.n_test_basis_states,
                T_temperature=args.T_target,
                eval_step_factor=args.eval_step_factor,
            )

            # Compute relative error (comparing to best known, using mean as proxy)
            # In practice, you'd compare to ground truth from solver
            rel_error = 0.0  # Placeholder - would need ground truth

            tqdm.write(f"  Valid ratio: {eval_results['valid_ratio']:.2%}")
            tqdm.write(f"  Mean tour cost (TSP): {eval_results['mean_tour_cost']:.4f}")
            tqdm.write(f"  Best tour cost (TSP): {eval_results['best_tour_cost']:.4f}")
            if not np.isnan(eval_results['best_valid_tour_cost']):
                tqdm.write(f"  Best valid tour cost: {eval_results['best_valid_tour_cost']:.4f}")
            tqdm.write(f"  Mean energy (w/ penalty): {eval_results['mean_energy']:.4f}")
            tqdm.write(f"  Mean violations: {eval_results['mean_violations']:.2f}")

            # Update best
            if eval_results['best_energy'] < best_energy:
                best_energy = eval_results['best_energy']
                epochs_since_best = 0
                tqdm.write(f"  ** New best energy! **")

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_energy': best_energy,
                    'config': vars(args),
                }, os.path.join(args.save_dir, 'best_model.pt'))
            else:
                epochs_since_best += 1

            # Eval logging
            log_dict.update({
                'eval/mean_energy': eval_results['mean_energy'],
                'eval/best_energy': eval_results['best_energy'],
                'eval/mean_tour_cost': eval_results['mean_tour_cost'],
                'eval/best_tour_cost': eval_results['best_tour_cost'],
                'eval/best_valid_tour_cost': eval_results['best_valid_tour_cost'],
                'eval/valid_ratio': eval_results['valid_ratio'],
                'eval/mean_violations': eval_results['mean_violations'],
                'eval/epochs_since_best': epochs_since_best,
                'eval/best_energy_overall': best_energy,
            })

            tqdm.write("")

        # Wandb logging
        if use_wandb:
            wandb.log(log_dict)

        # Early stopping
        if epochs_since_best >= args.stop_epochs:
            tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {args.stop_epochs} epochs)")
            break

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best energy: {best_energy:.4f}")
    print(f"  Total epochs: {epoch + 1}")
    print(f"{'='*70}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
