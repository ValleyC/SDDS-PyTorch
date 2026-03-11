"""
CVRP Partition Training Script

Train a categorical diffusion model for CVRP partitioning using PPO.
Following DiffUCO mechanics with CVRP-specific energy function.

Usage:
    python -m diffusion.train_cvrp --n_nodes 50 --n_epochs 100
"""

import torch
import torch.optim as optim
import argparse
import math
import time
import os
from typing import Optional, Dict
from torch_geometric.data import Batch
from tqdm import tqdm

# Local imports
try:
    from .step_model import CVRPDiffusionStepModel
    from .noise_schedule import CategoricalNoiseSchedule
    from .trajectory import collect_cvrp_trajectory, sample_cvrp_with_eval_step_factor
    from .ppo_trainer import PPOTrainer, MovingAverage, compute_gae, normalize_advantages
    from .cvrp_energy import (
        compute_K, compute_cvrp_cost, create_energy_fn,
        project_to_feasible, compute_capacity_violation
    )
except ImportError:
    from step_model import CVRPDiffusionStepModel
    from noise_schedule import CategoricalNoiseSchedule
    from trajectory import collect_cvrp_trajectory, sample_cvrp_with_eval_step_factor
    from ppo_trainer import PPOTrainer, MovingAverage, compute_gae, normalize_advantages
    from cvrp_energy import (
        compute_K, compute_cvrp_cost, create_energy_fn,
        project_to_feasible, compute_capacity_violation
    )

# CVRP instance generation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatmap.cvrp.inst import gen_inst, gen_pyg_data, CAPACITIES

# Sparse graph configuration (GLOP exact values)
# GLOP only trains on n=1000 and n=2000
# For smaller sizes, we use proportional values
K_SPARSE = {
    20: 10,
    50: 25,
    100: 50,
    200: 100,
    500: 100,
    1000: 100,  # GLOP exact
    2000: 200,  # GLOP exact
}


def generate_cvrp_batch(
    n_nodes: int,
    batch_size: int,
    k_sparse: int,
    device: torch.device
) -> Dict:
    """
    Generate a batch of CVRP instances.

    Args:
        n_nodes: Number of customer nodes (excluding depot)
        batch_size: Number of instances
        k_sparse: Sparsity parameter for graph construction
        device: Torch device

    Returns:
        Dict with keys:
            - pyg_batch: Batched PyG data
            - coors_list: List of (n+1, 2) coordinate tensors
            - demands_list: List of (n,) demand tensors (customers only)
            - capacity: Vehicle capacity
            - K_list: List of K values per instance
    """
    graphs = []
    coors_list = []
    demands_list = []
    K_list = []

    for _ in range(batch_size):
        coors, demand, capacity = gen_inst(n_nodes, device)
        pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse)

        # Store for later use
        pyg_data.coors = coors
        pyg_data.demand = demand[1:]  # Customer demands only (exclude depot)
        pyg_data.capacity = capacity

        graphs.append(pyg_data)
        coors_list.append(coors)
        demands_list.append(demand[1:])  # Exclude depot

        # Compute K for this instance
        K = compute_K(demand, capacity)
        K_list.append(K)

    pyg_batch = Batch.from_data_list(graphs)

    return {
        'pyg_batch': pyg_batch,
        'coors_list': coors_list,
        'demands_list': demands_list,
        'capacity': capacity,
        'K_list': K_list,
    }


def create_cvrp_energy_fn(
    coors: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    K: int,
    use_projection: bool = True,
    use_batched_sa: bool = True,  # Default True to match GLOP training
    penalty_lambda: float = 10.0,
    energy_scale: float = 1.0,  # Scale factor for energy reward
):
    """
    Create energy function for a single CVRP instance.

    This wraps the CVRP cost computation to match the expected
    signature: energy_fn(X, node_graph_idx, n_graphs) -> energy

    Args:
        energy_scale: Multiplier for energy to balance with entropy reward.
                      For 100 nodes, entropy ~300, cost ~30, so scale=10 balances them.
    """
    def energy_fn(X, node_graph_idx, n_graphs):
        # For single graph, compute CVRP cost
        total_cost, _, _ = compute_cvrp_cost(
            X, coors, demands, capacity, K,
            use_projection=use_projection,
            use_batched_sa=use_batched_sa,
            penalty_lambda=penalty_lambda
        )
        # Scale and return per-graph energy
        return (energy_scale * total_cost).unsqueeze(0)

    return energy_fn


class CVRPPPOTrainer:
    """
    PPO Trainer specialized for CVRP with node features.

    Extends the base PPO trainer logic to handle:
    1. CVRP node features (demand/cap, r, theta)
    2. CVRP-specific energy function
    3. Dynamic K (number of vehicles) per instance
    """

    def __init__(
        self,
        model: CVRPDiffusionStepModel,
        noise_schedule: CategoricalNoiseSchedule,
        lr: float = 3e-4,
        gamma: float = 1.0,
        TD_k: float = 3.0,
        clip_value: float = 0.2,
        value_weighting: float = 0.65,
        inner_loop_steps: int = 4,
        mov_average_alpha: float = 0.2,
        max_grad_norm: float = 1.0,  # Gradient clipping
    ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.gamma = gamma
        self.clip_value = clip_value
        self.value_weighting = value_weighting
        self.inner_loop_steps = inner_loop_steps
        self.max_grad_norm = max_grad_norm

        # Compute lambda from TD_k
        time_horizon = noise_schedule.n_steps
        self.lam = math.exp(-math.log(TD_k) / time_horizon)

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Moving average for reward normalization
        self.moving_avg = MovingAverage(mov_average_alpha, mov_average_alpha)

    def train_step(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_features: torch.Tensor,
        n_basis_states: int,
        T_temperature: float,
        energy_fn,
        K_instance: int,
        eval_step_factor: int = 1,
    ) -> Dict:
        """
        Single training step for CVRP.

        Args:
            edge_index: Graph edges (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            node_features: (n_nodes, node_feat_dim) CVRP features
            n_basis_states: Number of parallel samples
            T_temperature: Temperature parameter
            energy_fn: Energy function for CVRP cost
            K_instance: Number of vehicles for this instance (for masking)
            eval_step_factor: Diffusion step factor

        Returns:
            Dict with training metrics
        """
        device = node_features.device
        n_nodes_total = node_features.size(0)  # Including depot
        n_customers = n_nodes_total - 1
        n_graphs = 1  # Single graph training
        depot_idx = 0

        # Create node_graph_idx (all nodes belong to graph 0)
        node_graph_idx = torch.zeros(n_nodes_total, dtype=torch.long, device=device)
        customer_graph_idx = node_graph_idx[1:]  # Exclude depot

        # Customer mask
        customer_mask = torch.ones(n_nodes_total, dtype=torch.bool, device=device)
        customer_mask[depot_idx] = False

        # Collect trajectory with K_instance for masking invalid vehicle labels
        self.model.eval()
        with torch.no_grad():
            buffer = collect_cvrp_trajectory(
                model=self.model,
                noise_schedule=self.noise_schedule,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_graph_idx=node_graph_idx,
                node_features=node_features,
                n_graphs=n_graphs,
                n_basis_states=n_basis_states,
                T_temperature=T_temperature,
                energy_fn=energy_fn,
                device=device,
                eval_step_factor=eval_step_factor,
                K_valid=K_instance,
                depot_idx=depot_idx,
            )

        # Normalize rewards using moving average
        # DiffUCO: exclude padding (last element in dim 2)
        reduced_rewards = buffer.rewards[:, :, :-1] if buffer.rewards.shape[2] > 1 else buffer.rewards
        mov_mean, mov_std = self.moving_avg.update_mov_averages(reduced_rewards)
        normed_rewards = self.moving_avg.calculate_average(buffer.rewards, mov_mean, mov_std)

        # Compute GAE advantages
        # Need to append bootstrap value
        values_with_bootstrap = torch.cat([
            buffer.values,
            torch.zeros(1, n_graphs, n_basis_states, device=device)
        ], dim=0)

        value_targets, advantages = compute_gae(
            normed_rewards, values_with_bootstrap, self.gamma, self.lam
        )

        # Normalize advantages
        advantages = normalize_advantages(advantages, exclude_last=(n_basis_states > 1))

        # PPO updates
        self.model.train()
        total_loss = 0.0
        n_updates = 0

        n_classes = self.model.n_classes

        for _ in range(self.inner_loop_steps):
            # For simplicity, use all data (no minibatching for now)
            for t in range(buffer.states.shape[0]):
                for b in range(n_basis_states):
                    # Buffer contains CUSTOMER states only
                    X_customers_t = buffer.states[t, :, b]
                    X_customers_next = buffer.actions[t, :, b]
                    rand_nodes = buffer.rand_node_features[t, :, b, :]
                    old_log_prob = buffer.policies[t, :, b]
                    adv = advantages[t, :, b]
                    v_target = value_targets[t, :, b]

                    # Get time index
                    t_idx = buffer.time_index_per_node[t, 0, b].item()

                    # Build full state with depot (fixed at 0) for model forward
                    X_full_t = torch.zeros(n_nodes_total, dtype=torch.long, device=device)
                    X_full_t[depot_idx] = 0
                    X_full_t[customer_mask] = X_customers_t

                    # Forward pass with full graph
                    # Pass customer_mask so value head only aggregates customer nodes
                    spin_log_probs_full, values, _ = self.model(
                        X_full_t, t_idx, edge_index, edge_attr,
                        node_graph_idx, n_graphs, node_features,
                        rand_nodes=rand_nodes,
                        customer_mask=customer_mask
                    )

                    # Extract CUSTOMER log probs only
                    spin_log_probs = spin_log_probs_full[customer_mask]

                    # Mask invalid vehicle labels (same as trajectory collection)
                    if K_instance < n_classes:
                        spin_log_probs = spin_log_probs.clone()
                        spin_log_probs[:, K_instance:] = float('-inf')

                    # New log prob (over CUSTOMERS only)
                    new_log_prob = self.model.get_state_log_prob(
                        spin_log_probs, X_customers_next, customer_graph_idx, n_graphs
                    )

                    # PPO ratio with numerical stability
                    log_ratio = new_log_prob - old_log_prob
                    # Clamp log_ratio to prevent exp overflow
                    # exp(5) ≈ 148, exp(10) ≈ 22026 - keep ratio reasonable
                    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                    ratio = torch.exp(log_ratio)

                    # Clipped surrogate
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value) * adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    # Note: rewards are normalized via moving average before GAE,
                    # so value targets should already be in a reasonable range.
                    # No hard clipping needed (was causing issues with large energy_scale).
                    critic_loss = torch.nn.functional.mse_loss(values, v_target)

                    # Combined loss
                    loss = (1 - self.value_weighting) * actor_loss + self.value_weighting * critic_loss

                    # Skip update if loss is NaN or Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                    total_loss += loss.item()
                    n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)

        # Compute best cost from final states (customer states only)
        best_costs = []
        for b in range(n_basis_states):
            X_customers_0 = buffer.final_states[:, b]
            cost = energy_fn(X_customers_0, customer_graph_idx, n_graphs)
            best_costs.append(cost.item())

        return {
            'loss': avg_loss,
            'best_cost': min(best_costs),
            'mean_cost': sum(best_costs) / len(best_costs),
            'energy_reward': buffer.energy_rewards.mean().item(),
        }


def train_cvrp(
    n_nodes: int = 50,
    n_epochs: int = 100,
    n_instances_per_epoch: int = 10,
    n_basis_states: int = 8,
    n_diffusion_steps: int = 10,
    hidden_dim: int = 64,
    n_message_passes: int = 5,
    lr: float = 3e-4,
    T_temperature: float = 1.0,
    k_sparse: int = None,  # None = auto from K_SPARSE
    use_projection: bool = True,
    use_batched_sa: bool = True,  # Default True to match GLOP training (SA-based sub-TSP)
    eval_step_factor: int = 1,
    energy_scale: float = 10.0,  # Scale energy to balance with entropy (~n_nodes/cost)
    checkpoint_dir: str = 'checkpoints',
    device: str = 'cuda',
    seed: int = 42,
):
    """
    Main training function for CVRP partition.

    Args:
        energy_scale: Multiplier for energy reward. Default 10.0 balances energy (~30)
                      with entropy reward (~300 for 100 nodes with T=1).
        checkpoint_dir: Directory to save model checkpoints.
    """
    torch.manual_seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)

    # Auto-select k_sparse based on problem size
    if k_sparse is None:
        k_sparse = K_SPARSE.get(n_nodes, min(n_nodes // 2, 100))

    print("=" * 60)
    print("CVRP Partition Training (Categorical Diffusion + PPO)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  n_nodes: {n_nodes}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  n_instances_per_epoch: {n_instances_per_epoch}")
    print(f"  n_basis_states: {n_basis_states}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  T_temperature: {T_temperature}")
    print(f"  k_sparse: {k_sparse}")
    print(f"  use_projection: {use_projection}")
    print(f"  use_batched_sa: {use_batched_sa}")
    tsp_solver = "Batched SA" if use_batched_sa else "Greedy Nearest Neighbor"
    print(f"  TSP solver: {tsp_solver}")
    print(f"  energy_scale: {energy_scale}")
    print(f"  device: {device}")

    # Compute K_max (maximum possible vehicles for worst-case demand)
    # Demands are randint(1, 10), so max total = 9*n, worst-case K = ceil(9n/cap) + 1
    # Add buffer of 2 for safety
    capacity = CAPACITIES.get(n_nodes, 50.0)
    K_max = math.ceil(9 * n_nodes / capacity) + 2
    print(f"  capacity: {capacity}")
    print(f"  K_max (worst-case): {K_max}")

    # Create model
    model = CVRPDiffusionStepModel(
        n_classes=K_max,
        node_feat_dim=3,  # (demand/cap, r, theta)
        edge_dim=2,
        hidden_dim=hidden_dim,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=n_message_passes,
        n_random_features=5,
        time_encoding='sinusoidal',
        embedding_dim=32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Create noise schedule
    noise_schedule = CategoricalNoiseSchedule(
        n_steps=n_diffusion_steps,
        n_classes=K_max,
        schedule='diffuco'
    )

    # Create trainer
    trainer = CVRPPPOTrainer(
        model=model,
        noise_schedule=noise_schedule,
        lr=lr,
        gamma=1.0,
        TD_k=3.0,
        clip_value=0.2,
        value_weighting=0.65,
        inner_loop_steps=4,
        mov_average_alpha=0.2,
    )

    print("\n" + "=" * 60)
    print("Training started...")
    print("=" * 60)

    best_overall_cost = float('inf')

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_start = time.time()
        epoch_losses = []
        epoch_costs = []

        inst_pbar = tqdm(range(n_instances_per_epoch), desc=f"Epoch {epoch+1}",
                         leave=False, unit="inst")
        for inst_idx in inst_pbar:
            # Generate instance
            coors, demand, capacity = gen_inst(n_nodes, device)
            pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse)

            # Extract customer demands (exclude depot)
            customer_demands = demand[1:]

            # Compute per-instance K (GLOP-faithful)
            K_instance = compute_K(demand, capacity)

            # GLOP-faithful: Keep ALL nodes including depot in graph
            # Node features: (norm_demand, r, theta) for all n+1 nodes
            # Depot has demand=0, r=0, theta=0
            node_features = pyg_data.x.to(device)  # All n+1 nodes including depot

            # Use GLOP's sparse graph directly (includes depot)
            edge_index = pyg_data.edge_index.to(device)
            edge_attr = pyg_data.edge_attr.to(device)

            # Create energy function for CVRP cost using actual K_instance
            # Note: Trajectory collection now passes CUSTOMER states directly (depot excluded)
            energy_fn = create_cvrp_energy_fn(
                coors, customer_demands, capacity, K_instance,
                use_projection=use_projection,
                use_batched_sa=use_batched_sa,
                energy_scale=energy_scale,
            )

            # Train step with per-instance K for masking
            metrics = trainer.train_step(
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_features=node_features,
                n_basis_states=n_basis_states,
                T_temperature=T_temperature,
                energy_fn=energy_fn,
                K_instance=K_instance,
                eval_step_factor=eval_step_factor,
            )

            epoch_losses.append(metrics['loss'])
            # Unscale the cost for reporting (energy_fn returns scaled cost)
            actual_cost = metrics['best_cost'] / energy_scale
            epoch_costs.append(actual_cost)

            # Update instance progress bar
            inst_pbar.set_postfix(loss=f"{metrics['loss']:.3f}", cost=f"{actual_cost:.2f}")

        inst_pbar.close()
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_cost = sum(epoch_costs) / len(epoch_costs)
        best_cost = min(epoch_costs)

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            # Save checkpoint
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'cvrp_{n_nodes}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_cost': best_overall_cost,
                'config': {
                    'n_nodes': n_nodes,
                    'n_diffusion_steps': n_diffusion_steps,
                    'hidden_dim': hidden_dim,
                    'K_max': K_max,
                    'energy_scale': energy_scale,
                    'use_projection': use_projection,
                }
            }, checkpoint_path)

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            loss=f"{avg_loss:.3f}",
            avg_cost=f"{avg_cost:.1f}",
            best=f"{best_overall_cost:.1f}"
        )

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print("=" * 60)

    # Report costs
    if use_projection:
        print(f"\nBest CVRP cost (with projection): {best_overall_cost:.2f}")
    else:
        print(f"\nBest cost (TSP + penalty): {best_overall_cost:.2f}")
        print(f"  Note: This includes penalty_lambda * capacity_violation")
        print(f"  To get pure CVRP cost, run evaluation with --use_projection")

    print(f"\nCheckpoint saved to: {os.path.join(checkpoint_dir, f'cvrp_{n_nodes}_best.pt')}")
    print(f"\nTraining config:")
    print(f"  - Capacity constraint: {'Hard (projection)' if use_projection else 'Soft (penalty)'}")
    print(f"  - Sub-TSP solver: {tsp_solver}")
    print(f"  - Energy scale: {energy_scale}")

    return model, trainer


def test_train_cvrp():
    """Quick test of CVRP training."""
    print("=" * 60)
    print("Testing CVRP Training (Quick)")
    print("=" * 60)

    model, trainer = train_cvrp(
        n_nodes=20,
        n_epochs=3,
        n_instances_per_epoch=2,
        n_basis_states=4,
        n_diffusion_steps=5,
        hidden_dim=32,
        n_message_passes=3,
        lr=1e-3,
        T_temperature=1.0,
        k_sparse=10,
        use_projection=True,
        use_batched_sa=False,
        device='cpu',
    )

    print("\n[OK] CVRP training test passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVRP partition with diffusion")
    parser.add_argument('--n_nodes', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_instances_per_epoch', type=int, default=10)
    parser.add_argument('--n_basis_states', type=int, default=8)
    parser.add_argument('--n_diffusion_steps', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--T_temperature', type=float, default=1.0)
    parser.add_argument('--k_sparse', type=int, default=None,
                        help='Sparse graph k-NN (default: auto based on n_nodes)')
    parser.add_argument('--no_batched_sa', action='store_true',
                        help='Disable Batched SA (use greedy NN instead, faster but worse quality)')
    parser.add_argument('--energy_scale', type=float, default=10.0,
                        help='Scale factor for energy reward (default: 10.0 to balance with entropy)')
    parser.add_argument('--no_projection', action='store_true',
                        help='Disable hard projection to feasible solution. Use soft penalty instead. '
                             'RECOMMENDED for training (projection breaks credit assignment).')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test', action='store_true', help='Run quick test')

    args = parser.parse_args()

    if args.test:
        test_train_cvrp()
    else:
        train_cvrp(
            n_nodes=args.n_nodes,
            n_epochs=args.n_epochs,
            n_instances_per_epoch=args.n_instances_per_epoch,
            n_basis_states=args.n_basis_states,
            n_diffusion_steps=args.n_diffusion_steps,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            T_temperature=args.T_temperature,
            k_sparse=args.k_sparse,
            use_projection=not args.no_projection,
            use_batched_sa=not args.no_batched_sa,
            energy_scale=args.energy_scale,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )
