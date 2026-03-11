"""
Heatmap-based Chip Placement Training Script

GLOP-inspired architecture: GNN predicts heatmap logits, sequential greedy
placer enforces hard constraints via masking, REINFORCE trains the GNN.

Usage:
    # Quick smoke test (5 components, grid 8)
    python train_chip_heatmap.py --dataset Chip_5_components --grid_size 8 --n_epochs 20

    # Synthetic training (20 components)
    python train_chip_heatmap.py --dataset Chip_20_components --grid_size 32 --n_epochs 500

    # ICCAD04 benchmark (ibm01 macros)
    python train_chip_heatmap.py --benchmark --circuit ibm01 --macros_only --grid_size 32 --n_epochs 500

    # With imitation pretraining + wandb
    python train_chip_heatmap.py --benchmark --circuit ibm01 --macros_only --pretrain_epochs 50 --wandb
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
import os
import math

from heatmap_model import ChipHeatmapModel
from greedy_placer import GreedyPlacer
from reinforce_trainer import REINFORCETrainer
from reviser import compute_cost, local_improvement
from chip_placement_energy import compute_chip_placement_energy
from chip_placement_data import generate_chip_batch
from benchmark_loader import (
    load_benchmark_batch, load_iccad04_batch, ICCAD04_CIRCUITS
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def cosine_schedule(step, total_steps, max_lr, min_lr):
    """Cosine learning rate schedule."""
    if total_steps == 0:
        return max_lr
    progress = min(step / total_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(
        description='Heatmap GNN + Greedy Placer for Chip Placement (REINFORCE)')

    # Data
    parser.add_argument('--dataset', type=str, default='Chip_20_components',
                        help='Synthetic dataset name')
    parser.add_argument('--n_graphs', type=int, default=10,
                        help='Number of graph instances per batch')
    parser.add_argument('--benchmark', action='store_true',
                        help='Use ICCAD04 benchmark instead of synthetic')
    parser.add_argument('--circuit', type=str, default='ibm01',
                        help='ICCAD04 circuit name')
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='Max nodes (BFS subsample)')
    parser.add_argument('--macros_only', action='store_true', default=True,
                        help='Only place macros (filter standard cells)')
    parser.add_argument('--benchmark_base', type=str, default='benchmarks',
                        help='Benchmark base directory')
    parser.add_argument('--fresh_data_every', type=int, default=1,
                        help='Generate fresh data every N epochs (0=never)')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='GNN hidden dimension')
    parser.add_argument('--n_message_passes', type=int, default=5,
                        help='Number of GNN message passing layers')
    parser.add_argument('--grid_size', type=int, default=32,
                        help='Placement grid resolution (G for G×G)')
    parser.add_argument('--alpha_max', type=float, default=2.0,
                        help='Max occupancy gate value')

    # Training
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'constant'], help='LR schedule')
    parser.add_argument('--lr_min_ratio', type=float, default=0.01,
                        help='Min LR as fraction of max LR')
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of RL training epochs')
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Placement samples per instance (basis states)')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Entropy bonus weight')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--congestion_weight', type=float, default=0.1,
                        help='Congestion penalty weight in cost')
    parser.add_argument('--congestion_grid', type=int, default=8,
                        help='Grid resolution for congestion estimation')
    parser.add_argument('--reviser_iters', type=int, default=20,
                        help='Local improvement iterations (0=disable)')
    parser.add_argument('--invalid_penalty', type=float, default=1000.0,
                        help='Cost penalty for invalid placements')

    # Curriculum
    parser.add_argument('--repair_until_epoch', type=int, default=50,
                        help='Use repair mode for first N epochs, then strict')
    parser.add_argument('--pretrain_epochs', type=int, default=0,
                        help='Imitation pretraining epochs (0=skip)')

    # Eval
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluation frequency')
    parser.add_argument('--n_eval_samples', type=int, default=64,
                        help='Samples for evaluation')
    parser.add_argument('--overlap_threshold', type=float, default=0.01,
                        help='Max overlap for legal placement (default: 0.01)')
    parser.add_argument('--boundary_threshold', type=float, default=0.01,
                        help='Max boundary violation for legal placement (default: 0.01)')

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Enable wandb')
    parser.add_argument('--project_name', type=str, default='ChipPlacement_Heatmap')
    parser.add_argument('--save_dir', type=str, default='checkpoints_heatmap')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    use_benchmark = args.benchmark

    # Print config
    print(f"\n{'='*70}")
    print(f"Heatmap GNN + Greedy Placer — REINFORCE Training")
    print(f"{'='*70}")
    print(f"Data: {'ICCAD04 ' + args.circuit if use_benchmark else args.dataset}")
    if use_benchmark:
        print(f"  macros_only={args.macros_only}, max_nodes={args.max_nodes}")
    print(f"  n_graphs={args.n_graphs}")
    print(f"\nModel:")
    print(f"  hidden_dim={args.hidden_dim}, n_message_passes={args.n_message_passes}")
    print(f"  grid_size={args.grid_size} ({args.grid_size}x{args.grid_size}={args.grid_size**2} cells)")
    print(f"  alpha_max={args.alpha_max}")
    print(f"\nTraining:")
    print(f"  lr={args.lr} ({args.lr_schedule}), n_epochs={args.n_epochs}")
    print(f"  n_samples={args.n_samples}, entropy_weight={args.entropy_weight}")
    print(f"  congestion_weight={args.congestion_weight}, congestion_grid={args.congestion_grid}")
    print(f"  reviser_iters={args.reviser_iters}, invalid_penalty={args.invalid_penalty}")
    print(f"\nCurriculum:")
    print(f"  pretrain_epochs={args.pretrain_epochs}")
    print(f"  repair_until_epoch={args.repair_until_epoch}")
    print(f"\nEval:")
    print(f"  overlap_threshold={args.overlap_threshold}, boundary_threshold={args.boundary_threshold}")
    print(f"  eval_every={args.eval_every}, n_eval_samples={args.n_eval_samples}")
    print(f"{'='*70}\n")

    # Wandb
    use_wandb = False
    if args.wandb and WANDB_AVAILABLE:
        dataset_tag = (f"{args.circuit}_n{args.max_nodes}" if use_benchmark
                       else args.dataset)
        wandb.init(
            project=f"{args.project_name}_{dataset_tag}",
            name=f"G{args.grid_size}_lr{args.lr}_s{args.n_samples}",
            config=vars(args),
        )
        use_wandb = True

    # ------- Data Loading -------
    data_seed = args.seed

    def load_data(seed):
        if use_benchmark:
            circuit_dirs = []
            circuit_names = []
            for g in range(args.n_graphs):
                circuit_dirs.append(
                    os.path.join(args.benchmark_base, "iccad04", "extracted",
                                 args.circuit))
                circuit_names.append(args.circuit)
            return load_benchmark_batch(
                circuit_dirs, circuit_names,
                max_nodes=args.max_nodes,
                macros_only=args.macros_only,
                seed=seed, device=device,
            )
        else:
            return generate_chip_batch(
                args.n_graphs, args.dataset, seed=seed, device=device)

    (node_features, edge_index, edge_attr, node_graph_idx,
     initial_positions, legal_positions) = load_data(data_seed)

    n_nodes = node_features.shape[0]
    n_edges = edge_index.shape[1]
    component_sizes = node_features  # (N, 2) = (width, height)
    print(f"Loaded: {n_nodes} nodes, {n_edges} edges, {args.n_graphs} graphs")

    # ------- Model -------
    model = ChipHeatmapModel(
        node_feature_dim=2,
        edge_dim=4,
        hidden_dim=args.hidden_dim,
        grid_size=args.grid_size,
        n_message_passes=args.n_message_passes,
        alpha_max=args.alpha_max,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------- Placer -------
    placer = GreedyPlacer(grid_size=args.grid_size).to(device)

    # ------- Trainer -------
    trainer = REINFORCETrainer(
        model=model,
        placer=placer,
        lr=args.lr,
        grad_clip=args.grad_clip,
        entropy_weight=args.entropy_weight,
        congestion_weight=args.congestion_weight,
        congestion_grid=args.congestion_grid,
        reviser_iters=args.reviser_iters,
        invalid_penalty=args.invalid_penalty,
    )

    # ------- Checkpointing -------
    os.makedirs(args.save_dir, exist_ok=True)
    best_hpwl = float('inf')

    # ------- Imitation Pretraining -------
    if args.pretrain_epochs > 0 and legal_positions is not None:
        print(f"\n--- Imitation Pretraining ({args.pretrain_epochs} epochs) ---")
        trainer.pretrain_imitation(
            node_features, edge_index, edge_attr,
            legal_positions, args.grid_size,
            n_epochs=args.pretrain_epochs,
            log_every=max(1, args.pretrain_epochs // 10),
        )
        print("--- Pretraining complete ---\n")

    # ------- Reference HPWL -------
    if legal_positions is not None:
        with torch.no_grad():
            _, ref_hpwl, ref_overlap, ref_boundary = compute_chip_placement_energy(
                legal_positions, component_sizes, edge_index,
                node_graph_idx, args.n_graphs,
                overlap_weight=0.0, boundary_weight=0.0,
                edge_attr=edge_attr,
            )
        print(f"Reference placement: HPWL={ref_hpwl.mean().item():.4f}, "
              f"overlap={ref_overlap.mean().item():.6f}, "
              f"boundary={ref_boundary.mean().item():.6f}")

    # ------- Training Loop -------
    print(f"\n--- RL Training ({args.n_epochs} epochs) ---")
    epoch_range = tqdm(range(args.n_epochs), desc="Training")

    for epoch in epoch_range:
        t0 = time.time()

        # Curriculum: repair mode for early epochs
        repair_mode = (epoch < args.repair_until_epoch)

        # LR schedule
        if args.lr_schedule == 'cosine':
            new_lr = cosine_schedule(
                epoch, args.n_epochs, args.lr, args.lr * args.lr_min_ratio)
            for pg in trainer.optimizer.param_groups:
                pg['lr'] = new_lr

        # Fresh data
        if (args.fresh_data_every > 0 and epoch > 0 and
                epoch % args.fresh_data_every == 0 and not use_benchmark):
            data_seed += 1
            (node_features, edge_index, edge_attr, node_graph_idx,
             initial_positions, legal_positions) = load_data(data_seed)
            component_sizes = node_features

        # Train step
        metrics = trainer.train_step(
            node_features, edge_index, edge_attr,
            node_graph_idx, args.n_graphs, component_sizes,
            n_samples=args.n_samples,
            repair_mode=repair_mode,
            use_reviser=(args.reviser_iters > 0),
        )

        dt = time.time() - t0

        # Logging
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            mode = "repair" if repair_mode else "strict"
            lr_now = trainer.optimizer.param_groups[0]['lr']
            tqdm.write(
                f"Ep {epoch:4d} [{mode}] "
                f"loss={metrics['loss']:.4f} "
                f"HPWL={metrics['mean_hpwl']:.4f} "
                f"(greedy={metrics['greedy_hpwl']:.4f}, best={metrics['best_hpwl']:.4f}) "
                f"cong={metrics['mean_congestion']:.4f} "
                f"inv={metrics['invalid_ratio']:.2f} "
                f"rep={metrics['repaired_ratio']:.2f} "
                f"ent={metrics['entropy']:.3f} "
                f"lp={metrics['mean_log_prob']:.3f} "
                f"alpha={metrics['occ_alpha']:.3f} "
                f"lr={lr_now:.2e} "
                f"({dt:.1f}s)"
            )

        # Wandb logging
        if use_wandb:
            log_dict = {f"train/{k}": v for k, v in metrics.items()}
            log_dict['train/lr'] = trainer.optimizer.param_groups[0]['lr']
            log_dict['train/epoch'] = epoch
            log_dict['train/repair_mode'] = int(repair_mode)
            wandb.log(log_dict, step=epoch)

        # Evaluation
        if (epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1:
            eval_metrics = trainer.evaluate(
                node_features, edge_index, edge_attr,
                node_graph_idx, args.n_graphs, component_sizes,
                n_eval_samples=args.n_eval_samples,
                overlap_threshold=args.overlap_threshold,
                boundary_threshold=args.boundary_threshold,
                use_reviser=(args.reviser_iters > 0),
            )

            tqdm.write(
                f"\n  EVAL @ {epoch+1}: "
                f"greedy_HPWL={eval_metrics['greedy_hpwl']:.4f} "
                f"best_sampled={eval_metrics['best_sampled_hpwl']:.4f} "
                f"best_legal={eval_metrics['best_legal_hpwl']:.4f} "
                f"feasibility={eval_metrics['strict_feasibility']:.3f} "
                f"legal={eval_metrics['legal_ratio']:.3f} "
                f"overlap={eval_metrics['greedy_overlap']:.6f} "
                f"boundary={eval_metrics['greedy_boundary']:.6f}\n"
            )

            if use_wandb:
                eval_log = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(eval_log, step=epoch)

            # Save best checkpoint
            current_hpwl = eval_metrics['greedy_hpwl']
            if current_hpwl < best_hpwl:
                best_hpwl = current_hpwl
                ckpt_path = os.path.join(args.save_dir, 'best_heatmap.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_hpwl': best_hpwl,
                    'eval_metrics': eval_metrics,
                    'config': vars(args),
                }, ckpt_path)
                tqdm.write(f"  Saved best checkpoint: HPWL={best_hpwl:.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training complete. Best greedy HPWL: {best_hpwl:.4f}")
    if legal_positions is not None:
        print(f"Reference HPWL: {ref_hpwl.mean().item():.4f}")
    print(f"{'='*70}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
