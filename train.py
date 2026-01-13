"""
Training script for SDDS-PyTorch.

Usage:
    python train.py --n_nodes 20 --n_epochs 100 --batch_size 32
"""

import argparse
import torch
import os
from tqdm import tqdm
import json
from datetime import datetime

from Networks import DiffusionModelDense
from NoiseDistributions import BernoulliNoise
from EnergyFunctions import TSPEnergyClass
from Trainers import PPOTrainer
from Data import get_tsp_dataloader


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SDDS for TSP")

    # Problem settings
    parser.add_argument("--n_nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--problem", type=str, default="TSP", help="Problem type")

    # Diffusion settings
    parser.add_argument("--n_diffusion_steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--diff_schedule", type=str, default="DiffUCO", help="Noise schedule")

    # Model settings
    parser.add_argument("--n_layers", type=int, default=4, help="EGNN layers")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")

    # Training settings
    parser.add_argument("--n_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_instances", type=int, default=10000, help="Training instances")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_basis_states", type=int, default=8, help="Samples per instance")

    # PPO settings
    parser.add_argument("--clip_value", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coef")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--n_inner_steps", type=int, default=4, help="PPO epochs")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=10, help="Eval interval")
    parser.add_argument("--eval_samples", type=int, default=16, help="Eval samples")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()


def setup_config(args):
    """Create config dictionary from args."""
    return {
        # Problem
        "n_nodes": args.n_nodes,
        "problem_name": args.problem,

        # Diffusion
        "n_diffusion_steps": args.n_diffusion_steps,
        "diff_schedule": args.diff_schedule,
        "n_bernoulli_features": 2,

        # Model
        "n_layers": args.n_layers,
        "hidden_dim": args.hidden_dim,
        "node_dim": args.hidden_dim,
        "edge_dim": args.hidden_dim,
        "time_dim": args.hidden_dim,

        # Training
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_basis_states": args.n_basis_states,

        # PPO
        "clip_value": args.clip_value,
        "value_coef": args.value_coef,
        "gae_lambda": args.gae_lambda,
        "n_inner_steps": args.n_inner_steps,

        # Energy
        "penalty_weight": 1.0,

        # Other
        "seed": args.seed,
    }


def main():
    args = get_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)

    # Create config
    config = setup_config(args)

    # Create model
    print("Creating model...")
    model = DiffusionModelDense(
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    # Create noise distribution
    print("Creating noise distribution...")
    noise_class = BernoulliNoise(config)

    # Create energy function
    print("Creating energy function...")
    energy_class = TSPEnergyClass(config)

    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(
        config=config,
        model=model,
        energy_class=energy_class,
        noise_class=noise_class,
        device=device
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader = get_tsp_dataloader(
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        n_instances=args.n_instances,
        seed=args.seed
    )

    eval_loader = get_tsp_dataloader(
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        n_instances=1000,
        seed=args.seed + 1000
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"tsp{args.n_nodes}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nStarting training for {args.n_epochs} epochs...")
    best_energy = float("inf")

    for epoch in range(args.n_epochs):
        epoch_loss = 0
        epoch_metrics = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}")

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = trainer.train_step(batch)

            epoch_loss += loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            pbar.set_postfix({"loss": loss.item(), "energy": metrics.get("mean_energy", 0)})

        # Average metrics
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Mean Energy={epoch_metrics.get('mean_energy', 0):.4f}")

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            print("Evaluating...")
            trainer.model.eval()

            eval_energies = []
            for batch in tqdm(eval_loader, desc="Eval"):
                coords = batch["coords"].to(device)
                with torch.no_grad():
                    result = trainer.sample(coords, n_samples=args.eval_samples)
                    eval_energies.append(result["best_energy"].mean().item())

            mean_eval_energy = sum(eval_energies) / len(eval_energies)
            print(f"Eval Mean Best Energy: {mean_eval_energy:.4f}")

            # Save best model
            if mean_eval_energy < best_energy:
                best_energy = mean_eval_energy
                trainer.save_checkpoint(
                    os.path.join(output_dir, "best_model.pt"),
                    epoch + 1,
                    {"best_energy": best_energy}
                )
                print(f"New best model saved! Energy: {best_energy:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt"),
                epoch + 1
            )

    # Save final model
    trainer.save_checkpoint(
        os.path.join(output_dir, "final_model.pt"),
        args.n_epochs
    )
    print(f"\nTraining complete! Best energy: {best_energy:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
