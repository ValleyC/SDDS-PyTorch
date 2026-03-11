"""
MaxCut Training Script - DiffUCO-style Categorical Diffusion with PPO

MaxCut is a good validation problem because:
- Binary variables (2 classes) - simpler than TSP
- No validity constraints - all solutions are valid
- Clear objective: maximize cut value

Usage:
    python train_maxcut.py --n_nodes 20 --n_graphs 32 --N_anneal 1000
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
from step_model import DiffusionStepModel
from trajectory import collect_trajectory, sample_with_eval_step_factor
from ppo_trainer import PPOTrainer
from maxcut_energy import (
    compute_maxcut_energy,
    compute_maxcut_cut_value,
    generate_batch_random_graphs,
    create_maxcut_energy_fn,
)

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def cosine_schedule(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule."""
    if total_steps == 0:
        return max_lr
    progress = min(step / total_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def linear_annealing(epoch: int, N_warmup: int, N_anneal: int, N_equil: int,
                     T_max: float, T_target: float) -> float:
    """Linear temperature annealing."""
    total_epochs = N_warmup + N_anneal + N_equil

    if epoch < N_warmup:
        T_curr = T_max
    elif epoch >= N_warmup and epoch < total_epochs - N_equil - 1:
        T_curr = max(T_max - T_target - (T_max - T_target) * (epoch - N_warmup) / N_anneal, 0) + T_target
    else:
        T_curr = T_target

    return T_curr


def evaluate_maxcut(
    model: DiffusionStepModel,
    noise_schedule: CategoricalNoiseSchedule,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    n_nodes: int,
    n_basis_states: int = 8,
    T_temperature: float = 0.0,
    eval_step_factor: int = 1,
):
    """
    Evaluate MaxCut solutions.

    Args:
        eval_step_factor: Factor to multiply diffusion steps during eval
                         (matches DiffUCO's eval_step_factor)
    """
    model.eval()
    device = edge_index.device

    # Use sample_with_eval_step_factor for proper DiffUCO-style evaluation
    X_final = sample_with_eval_step_factor(
        model=model,
        noise_schedule=noise_schedule,
        edge_index=edge_index,
        edge_attr=edge_weights.unsqueeze(-1),
        node_graph_idx=node_graph_idx,
        n_graphs=n_graphs,
        n_basis_states=n_basis_states,
        T_temperature=T_temperature,
        eval_step_factor=eval_step_factor,
        device=device,
    )

    # X_final shape: (n_nodes_total, n_basis_states)
    # Reshape to (n_basis_states, n_graphs, n_nodes_per_graph)
    X_final_reshaped = X_final.T.reshape(n_basis_states, n_graphs, n_nodes)

    all_cut_values = []

    with torch.no_grad():
        for b in range(n_basis_states):
            for g in range(n_graphs):
                # Get edges for this graph
                mask = node_graph_idx[edge_index[0]] == g
                g_edge_index = edge_index[:, mask]
                g_edge_weights = edge_weights[mask]

                # Adjust indices to be relative to this graph
                offset = g * n_nodes
                g_edge_index = g_edge_index - offset

                X_g = X_final_reshaped[b, g]
                cut_val = compute_maxcut_cut_value(X_g, g_edge_index, g_edge_weights)
                all_cut_values.append(cut_val.item())

    model.train()

    cut_values = np.array(all_cut_values).reshape(n_basis_states, n_graphs)

    mean_cut = np.mean(cut_values)
    max_cut_per_graph = np.max(cut_values, axis=0)
    best_cut = np.mean(max_cut_per_graph)

    return {
        'mean_cut': mean_cut,
        'best_cut': best_cut,
        'max_cut_per_graph': max_cut_per_graph.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Model for MaxCut')

    # Problem
    parser.add_argument('--n_nodes', type=int, default=20)
    parser.add_argument('--n_graphs', type=int, default=32)
    parser.add_argument('--edge_prob', type=float, default=0.5, help='Graph density')

    # Model
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_message_passes', type=int, default=8)
    parser.add_argument('--n_diffusion_steps', type=int, default=9)
    parser.add_argument('--n_random_features', type=int, default=5)

    # Temperature
    parser.add_argument('--T_max', type=float, default=1.0)
    parser.add_argument('--T_target', type=float, default=0.01)
    parser.add_argument('--N_warmup', type=int, default=0)
    parser.add_argument('--N_anneal', type=int, default=1000)
    parser.add_argument('--N_equil', type=int, default=0)

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_basis_states', type=int, default=10)
    parser.add_argument('--TD_k', type=float, default=3.0)
    parser.add_argument('--clip_value', type=float, default=0.2)
    parser.add_argument('--value_weighting', type=float, default=0.65)
    parser.add_argument('--inner_loop_steps', type=int, default=4)
    parser.add_argument('--mov_average', type=float, default=0.1)

    # Eval
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--n_test_basis_states', type=int, default=8)
    parser.add_argument('--eval_step_factor', type=int, default=1,
                        help='Factor to multiply diffusion steps during eval (DiffUCO-style)')

    # Other
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='MaxCut_Diffusion')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    total_epochs = args.N_warmup + args.N_anneal + args.N_equil
    n_classes = 2  # Binary for MaxCut

    print(f"\n{'='*70}")
    print(f"MaxCut Diffusion Training")
    print(f"{'='*70}")
    print(f"Problem: n_nodes={args.n_nodes}, n_graphs={args.n_graphs}, edge_prob={args.edge_prob}")
    print(f"Model: hidden={args.n_hidden}, layers={args.n_message_passes}, steps={args.n_diffusion_steps}")
    print(f"Training: epochs={total_epochs}, T={args.T_max}->{args.T_target}, lr={args.lr}")
    print(f"{'='*70}\n")

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project=args.project_name, config=vars(args))
        use_wandb = True
    else:
        use_wandb = False

    # Model
    model = DiffusionStepModel(
        n_classes=n_classes,
        edge_dim=1,
        hidden_dim=args.n_hidden,
        n_diffusion_steps=args.n_diffusion_steps,
        n_message_passes=args.n_message_passes,
        n_random_features=args.n_random_features,
        time_encoding='sinusoidal',
        embedding_dim=32,
        mean_aggr=False,
    ).to(device)

    # Noise schedule
    noise_schedule = CategoricalNoiseSchedule(
        n_steps=args.n_diffusion_steps,
        n_classes=n_classes,
        schedule='diffuco',
    )

    # Trainer
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

    # Training
    best_cut = 0.0
    epoch_range = tqdm(range(total_epochs), desc="Training")

    for epoch in epoch_range:
        # Temperature
        T_curr = linear_annealing(
            epoch, args.N_warmup, args.N_anneal, args.N_equil,
            args.T_max, args.T_target
        )

        # Learning rate
        lr_curr = cosine_schedule(epoch, total_epochs, args.lr, args.lr / 10)
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = lr_curr

        # Generate new graphs each epoch
        edge_index, edge_weights, node_graph_idx = generate_batch_random_graphs(
            args.n_nodes, args.n_graphs, args.edge_prob, device
        )

        # Energy function
        def energy_fn(X, node_idx, n_g):
            return compute_maxcut_energy(X, edge_index, edge_weights, node_idx, n_g)

        # Training step
        loss_dict = trainer.train_step(
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(-1),
            node_graph_idx=node_graph_idx,
            n_graphs=args.n_graphs,
            n_basis_states=args.n_basis_states,
            T_temperature=T_curr,
            energy_fn=energy_fn,
        )

        # Logging
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
            # Generate test graphs
            test_edge_index, test_edge_weights, test_node_graph_idx = generate_batch_random_graphs(
                args.n_nodes, args.n_graphs, args.edge_prob, device
            )

            eval_results = evaluate_maxcut(
                model, noise_schedule,
                test_edge_index, test_edge_weights, test_node_graph_idx,
                args.n_graphs, args.n_nodes,
                n_basis_states=args.n_test_basis_states,
                T_temperature=args.T_target,
                eval_step_factor=args.eval_step_factor,
            )

            tqdm.write(f"\n--- Evaluation at epoch {epoch + 1} ---")
            tqdm.write(f"  Mean cut: {eval_results['mean_cut']:.2f}")
            tqdm.write(f"  Best cut: {eval_results['best_cut']:.2f}")

            if eval_results['best_cut'] > best_cut:
                best_cut = eval_results['best_cut']
                tqdm.write(f"  ** New best! **")

            tqdm.write("")

            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': loss_dict['overall_loss'],
                    'train/actor_loss': loss_dict['actor_loss'],
                    'train/critic_loss': loss_dict['critic_loss'],
                    'train/advantage_std': loss_dict['advantage_std'],
                    'eval/mean_cut': eval_results['mean_cut'],
                    'eval/best_cut': eval_results['best_cut'],
                    'schedules/T': T_curr,
                })

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best cut value: {best_cut:.2f}")
    print(f"{'='*70}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
