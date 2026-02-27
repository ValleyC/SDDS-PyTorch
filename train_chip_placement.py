"""
Chip Placement Training Script - Continuous Gaussian Diffusion with PPO

Faithful port of DIffUCO chip placement to SDDS-PyTorch.

Energy = HPWL + overlap_weight * (overlap * HPWL) + boundary_weight * (boundary * HPWL)

Usage:
    # Quick smoke test (5 components, 10 steps, 50 epochs)
    python train_chip_placement.py --dataset Chip_5_components --n_diffusion_steps 10 --N_anneal 50

    # Medium training (20 components)
    python train_chip_placement.py --dataset Chip_20_components --n_diffusion_steps 50 --N_anneal 2000

    # With wandb logging
    python train_chip_placement.py --wandb --project_name "ChipPlacement_Diffusion"

Reference: DIffUCO/train.py, DIffUCO/argparse_ray_main.py
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
import os
import math

from noise_schedule import GaussianNoiseSchedule
from continuous_step_model import ContinuousDiffusionStepModel
from trajectory import sample_continuous_with_eval_step_factor
from ppo_trainer import ContinuousPPOTrainer, MovingAverage
from chip_placement_energy import (
    compute_chip_placement_energy,
    create_chip_placement_energy_fn,
)
from chip_placement_data import generate_chip_batch, generate_chip_instance

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def cosine_schedule(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule matching DiffUCO's cos_schedule."""
    if total_steps == 0:
        return max_lr
    progress = min(step / total_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def linear_annealing(epoch: int, N_warmup: int, N_anneal: int, N_equil: int,
                     T_max: float, T_target: float) -> float:
    """Linear temperature annealing matching DiffUCO."""
    total_epochs = N_warmup + N_anneal + N_equil

    if epoch < N_warmup:
        T_curr = T_max
    elif epoch >= N_warmup and epoch < total_epochs - N_equil - 1:
        T_curr = max(T_max - T_target - (T_max - T_target) * (epoch - N_warmup) / N_anneal, 0) + T_target
    else:
        T_curr = T_target

    return T_curr


def evaluate_chip_placement(
    model: ContinuousDiffusionStepModel,
    noise_schedule: GaussianNoiseSchedule,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    node_features: torch.Tensor,
    component_sizes: torch.Tensor,
    n_basis_states: int = 8,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
    overlap_weight: float = 10.0,
    boundary_weight: float = 10.0,
):
    """
    Generate placements and evaluate metrics.

    Returns:
        Dict with mean/best energy, HPWL, overlap, boundary metrics.
    """
    model.eval()

    # Sample positions
    X_final = sample_continuous_with_eval_step_factor(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        node_features=node_features,
        T_temperature=T_temperature,
        eval_step_factor=eval_step_factor,
    )
    # X_final: (n_nodes, n_basis_states, continuous_dim)

    all_energies = []
    all_hpwl = []
    all_overlap = []
    all_boundary = []

    with torch.no_grad():
        for b in range(n_basis_states):
            positions_b = X_final[:, b, :]  # (n_nodes, 2)

            energy, hpwl, overlap, boundary = compute_chip_placement_energy(
                positions_b, component_sizes, edge_index,
                node_graph_idx, n_graphs,
                overlap_weight=overlap_weight,
                boundary_weight=boundary_weight,
            )

            all_energies.append(energy.cpu().numpy())
            all_hpwl.append(hpwl.cpu().numpy())
            all_overlap.append(overlap.cpu().numpy())
            all_boundary.append(boundary.cpu().numpy())

    model.train()

    # (n_basis_states, n_graphs) arrays
    energies = np.array(all_energies)
    hpwls = np.array(all_hpwl)
    overlaps = np.array(all_overlap)
    boundaries = np.array(all_boundary)

    # Best per graph (minimum energy across basis states)
    best_energy_per_graph = np.min(energies, axis=0)
    best_idx_per_graph = np.argmin(energies, axis=0)

    # Corresponding metrics for best energy solutions
    best_hpwl_per_graph = np.array([
        hpwls[best_idx_per_graph[g], g] for g in range(n_graphs)
    ])
    best_overlap_per_graph = np.array([
        overlaps[best_idx_per_graph[g], g] for g in range(n_graphs)
    ])
    best_boundary_per_graph = np.array([
        boundaries[best_idx_per_graph[g], g] for g in range(n_graphs)
    ])

    return {
        'mean_energy': np.mean(energies),
        'best_energy': np.mean(best_energy_per_graph),
        'mean_hpwl': np.mean(hpwls),
        'best_hpwl': np.mean(best_hpwl_per_graph),
        'mean_overlap': np.mean(overlaps),
        'best_overlap': np.mean(best_overlap_per_graph),
        'mean_boundary': np.mean(boundaries),
        'best_boundary': np.mean(best_boundary_per_graph),
        'all_energies': energies,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Continuous Diffusion for Chip Placement')

    # Problem parameters
    parser.add_argument('--dataset', type=str, default='Chip_20_components',
                        choices=['Chip_5_components', 'Chip_10_components',
                                 'Chip_20_components', 'Chip_50_components'],
                        help='Dataset scale preset')
    parser.add_argument('--n_graphs', type=int, default=10, help='Batch size (H)')
    parser.add_argument('--overlap_weight', type=float, default=10.0, help='Overlap penalty weight')
    parser.add_argument('--boundary_weight', type=float, default=10.0, help='Boundary penalty weight')

    # Model parameters
    parser.add_argument('--n_hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_message_passes', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--n_diffusion_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--n_random_features', type=int, default=5, help='Random node features')
    parser.add_argument('--time_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'one_hot'])
    parser.add_argument('--embedding_dim', type=int, default=32, help='Time embedding dim')

    # Temperature annealing
    parser.add_argument('--T_max', type=float, default=1.0, help='Initial/max temperature')
    parser.add_argument('--T_target', type=float, default=0.01, help='Target temperature')
    parser.add_argument('--N_warmup', type=int, default=0, help='Warmup epochs at T_max')
    parser.add_argument('--N_anneal', type=int, default=2000, help='Annealing epochs')
    parser.add_argument('--N_equil', type=int, default=0, help='Equilibration epochs')

    # Training parameters (PPO)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'constant'])
    parser.add_argument('--n_basis_states', type=int, default=10, help='Parallel trajectories (n_s)')
    parser.add_argument('--TD_k', type=float, default=3.0, help='TD(lambda) parameter')
    parser.add_argument('--clip_value', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_weighting', type=float, default=0.65, help='Critic loss weight')
    parser.add_argument('--inner_loop_steps', type=int, default=2, help='PPO inner loop steps')
    parser.add_argument('--mov_average', type=float, default=0.0009, help='Moving average alpha')
    parser.add_argument('--per_step_energy', action='store_true', default=True,
                        help='Compute energy at every diffusion step (SDDS fix)')
    parser.add_argument('--no_per_step_energy', dest='per_step_energy', action='store_false')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clip norm')

    # Evaluation
    parser.add_argument('--n_test_basis_states', type=int, default=8, help='Basis states for eval')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--eval_step_factor', type=int, default=1, help='Eval step factor')

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='ChipPlacement_Diffusion')

    # Other
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--fresh_data_every', type=int, default=1,
                        help='Generate fresh chip instances every N epochs (1=every epoch)')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Total epochs
    total_epochs = args.N_warmup + args.N_anneal + args.N_equil

    print(f"\n{'='*70}")
    print(f"Chip Placement - Continuous Gaussian Diffusion with PPO")
    print(f"{'='*70}")
    print(f"Problem:")
    print(f"  dataset: {args.dataset}")
    print(f"  n_graphs (batch H): {args.n_graphs}")
    print(f"  overlap_weight: {args.overlap_weight}")
    print(f"  boundary_weight: {args.boundary_weight}")
    print(f"\nModel:")
    print(f"  hidden_dim: {args.n_hidden}")
    print(f"  n_message_passes: {args.n_message_passes}")
    print(f"  n_diffusion_steps: {args.n_diffusion_steps}")
    print(f"  per_step_energy: {args.per_step_energy}")
    print(f"\nTraining:")
    print(f"  total_epochs: {total_epochs} (warmup={args.N_warmup}, anneal={args.N_anneal}, equil={args.N_equil})")
    print(f"  T_max -> T_target: {args.T_max} -> {args.T_target}")
    print(f"  lr: {args.lr} ({args.lr_schedule} schedule)")
    print(f"  n_basis_states: {args.n_basis_states}")
    print(f"  TD_k: {args.TD_k}, clip: {args.clip_value}, c1: {args.value_weighting}")
    print(f"{'='*70}\n")

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb_project = f"{args.project_name}_{args.dataset}"
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
    model = ContinuousDiffusionStepModel(
        continuous_dim=2,
        node_feature_dim=2,  # Component (width, height)
        edge_dim=4,  # Terminal offsets (src_dx, src_dy, dst_dx, dst_dy)
        hidden_dim=args.n_hidden,
        n_diffusion_steps=args.n_diffusion_steps,
        n_message_passes=args.n_message_passes,
        n_random_features=args.n_random_features,
        time_encoding=args.time_encoding,
        embedding_dim=args.embedding_dim,
        mean_aggr=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create noise schedule
    noise_schedule = GaussianNoiseSchedule(
        n_steps=args.n_diffusion_steps,
        continuous_dim=2,
        schedule='diffuco',
    )

    # Create trainer
    trainer = ContinuousPPOTrainer(
        model=model,
        noise_schedule=noise_schedule,
        lr=args.lr,
        gamma=1.0,
        TD_k=args.TD_k,
        clip_value=args.clip_value,
        value_weighting=args.value_weighting,
        inner_loop_steps=args.inner_loop_steps,
        mov_average_alpha=args.mov_average,
        per_step_energy=args.per_step_energy,
        grad_clip=args.grad_clip,
    )

    # Training state
    best_energy = float('inf')
    epochs_since_best = 0
    global_step = 0

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Generate initial data
    data_seed = args.seed
    node_features, edge_index, edge_attr, node_graph_idx, initial_positions, legal_positions = \
        generate_chip_batch(args.n_graphs, args.dataset, seed=data_seed, device=device)

    print(f"Initial batch: {node_features.shape[0]} total nodes, "
          f"{edge_index.shape[1]} edges, "
          f"{args.n_graphs} graphs")

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
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr_curr
        else:
            lr_curr = args.lr

        # Optionally generate fresh chip data
        if args.fresh_data_every > 0 and epoch > 0 and epoch % args.fresh_data_every == 0:
            data_seed += args.n_graphs
            node_features, edge_index, edge_attr, node_graph_idx, initial_positions, legal_positions = \
                generate_chip_batch(args.n_graphs, args.dataset, seed=data_seed, device=device)

        # Create energy function for this batch
        energy_fn = create_chip_placement_energy_fn(
            component_sizes=node_features,  # node_features = (width, height) = component sizes
            edge_index=edge_index,
            overlap_weight=args.overlap_weight,
            boundary_weight=args.boundary_weight,
        )

        # Training step
        loss_dict = trainer.train_step(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_graph_idx=node_graph_idx,
            n_graphs=args.n_graphs,
            n_basis_states=args.n_basis_states,
            T_temperature=T_curr,
            node_features=node_features,
            energy_fn=energy_fn,
        )

        global_step += args.inner_loop_steps
        epoch_time = time.time() - start_time

        # Logging dict
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

        # Periodic console logging
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

            # Generate fresh test instances
            test_seed = args.seed + 10000
            test_nf, test_ei, test_ea, test_ngi, test_pos, test_legal = \
                generate_chip_batch(args.n_graphs, args.dataset, seed=test_seed, device=device)

            eval_results = evaluate_chip_placement(
                model, noise_schedule,
                test_ei, test_ea, test_ngi, args.n_graphs,
                test_nf, test_nf,  # node_features = component_sizes
                n_basis_states=args.n_test_basis_states,
                T_temperature=args.T_target,
                eval_step_factor=args.eval_step_factor,
                overlap_weight=args.overlap_weight,
                boundary_weight=args.boundary_weight,
            )

            tqdm.write(f"  Mean energy: {eval_results['mean_energy']:.4f}")
            tqdm.write(f"  Best energy: {eval_results['best_energy']:.4f}")
            tqdm.write(f"  Best HPWL:   {eval_results['best_hpwl']:.4f}")
            tqdm.write(f"  Best overlap: {eval_results['best_overlap']:.6f}")
            tqdm.write(f"  Best boundary: {eval_results['best_boundary']:.6f}")

            # Also evaluate on legal placements as reference
            legal_energy, legal_hpwl, legal_overlap, legal_boundary = compute_chip_placement_energy(
                test_legal, test_nf, test_ei, test_ngi, args.n_graphs,
                args.overlap_weight, args.boundary_weight,
            )
            tqdm.write(f"  [Reference] Legal HPWL: {legal_hpwl.mean().item():.4f}, "
                        f"overlap: {legal_overlap.mean().item():.6f}, "
                        f"boundary: {legal_boundary.mean().item():.6f}")

            # Update best
            if eval_results['best_energy'] < best_energy:
                best_energy = eval_results['best_energy']
                epochs_since_best = 0
                tqdm.write(f"  ** New best energy! **")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_energy': best_energy,
                    'config': vars(args),
                }, os.path.join(args.save_dir, 'best_chip_model.pt'))
            else:
                epochs_since_best += 1

            log_dict.update({
                'eval/mean_energy': eval_results['mean_energy'],
                'eval/best_energy': eval_results['best_energy'],
                'eval/mean_hpwl': eval_results['mean_hpwl'],
                'eval/best_hpwl': eval_results['best_hpwl'],
                'eval/mean_overlap': eval_results['mean_overlap'],
                'eval/best_overlap': eval_results['best_overlap'],
                'eval/mean_boundary': eval_results['mean_boundary'],
                'eval/best_boundary': eval_results['best_boundary'],
                'eval/legal_hpwl': legal_hpwl.mean().item(),
                'eval/epochs_since_best': epochs_since_best,
                'eval/best_energy_overall': best_energy,
            })

            tqdm.write("")

        # Wandb logging
        if use_wandb:
            wandb.log(log_dict)

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
