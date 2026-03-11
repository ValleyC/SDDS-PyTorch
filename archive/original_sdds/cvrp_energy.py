"""
CVRP Energy Function for Diffusion-based Partition

This module implements the energy/cost function for CVRP partitioning:
1. Capacity violation penalty (soft constraint during training)
2. Feasibility projection (hard constraint during evaluation)
3. Sub-TSP cost computation using Batched SA

Reference: diffusion_partition_plan.md Section 3 and 4
"""

import torch
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Native PyTorch scatter_sum."""
    if dim_size is None:
        dim_size = int(index.max()) + 1

    if src.dim() == 1:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index, src)
    else:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        index_expanded = index.unsqueeze(-1).expand_as(src)
        out.scatter_add_(dim, index_expanded, src)

    return out


@dataclass
class CVRPInstance:
    """CVRP instance data."""
    coors: torch.Tensor        # (n+1, 2) coordinates including depot at index 0
    demands: torch.Tensor      # (n+1,) demands, depot has demand 0
    capacity: float            # vehicle capacity
    K: int                     # number of vehicles = ceil(total_demand / capacity) + 1


def compute_K(demands: torch.Tensor, capacity: float) -> int:
    """
    Compute number of vehicles K following GLOP convention.

    K = ceil(total_demand / capacity) + 1

    Args:
        demands: (n+1,) demands including depot (depot demand = 0)
        capacity: vehicle capacity

    Returns:
        K: number of vehicles
    """
    total_demand = demands.sum().item()  # depot demand is 0
    return math.ceil(total_demand / capacity) + 1


def compute_partition_demands(
    X: torch.Tensor,
    demands: torch.Tensor,
    K: int
) -> torch.Tensor:
    """
    Compute total demand per partition.

    Args:
        X: (n,) partition assignments for customer nodes (0 to K-1)
        demands: (n,) customer demands (NOT including depot)
        K: number of partitions

    Returns:
        partition_demands: (K,) total demand per partition
    """
    return scatter_sum(demands.float(), X.long(), dim=0, dim_size=K)


def compute_capacity_violation(
    X: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    K: int
) -> torch.Tensor:
    """
    Compute capacity violation cost.

    Violation = sum over k of max(0, partition_demand[k] - capacity)

    Args:
        X: (n,) partition assignments for customer nodes
        demands: (n,) customer demands
        capacity: vehicle capacity
        K: number of partitions

    Returns:
        violation: scalar tensor - total violation cost
    """
    partition_demands = compute_partition_demands(X, demands, K)
    violations = torch.clamp(partition_demands - capacity, min=0)
    return violations.sum()


def project_to_feasible(
    X: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    K: int,
    max_iterations: int = 1000
) -> torch.Tensor:
    """
    Project partition assignment to feasible solution.

    Greedy reassignment: repeatedly move smallest-demand node from
    overloaded partition to partition with most remaining capacity.

    Args:
        X: (n,) partition assignments
        demands: (n,) customer demands
        capacity: vehicle capacity
        K: number of partitions
        max_iterations: maximum iterations to prevent infinite loops

    Returns:
        X_proj: (n,) feasible partition assignments
    """
    X_proj = X.clone()
    n = X.size(0)
    device = X.device

    # Ensure demands is float for inf comparisons
    demands_float = demands.float()

    partition_demands = compute_partition_demands(X_proj, demands_float, K)

    for iteration in range(max_iterations):
        # Find overloaded partitions
        overloaded = partition_demands > capacity

        if not overloaded.any():
            break

        # Get first overloaded partition
        k = overloaded.nonzero(as_tuple=True)[0][0].item()

        # Find nodes in partition k
        in_partition_k = (X_proj == k)
        if not in_partition_k.any():
            # No nodes to move (shouldn't happen but handle gracefully)
            partition_demands[k] = 0
            continue

        # Find node with smallest demand in partition k
        node_demands_k = torch.where(in_partition_k, demands_float, torch.full_like(demands_float, float('inf')))
        node_to_move = node_demands_k.argmin().item()
        demand_to_move = demands_float[node_to_move]

        # Find partition with most remaining capacity
        remaining_capacity = capacity - partition_demands
        remaining_capacity[k] = -float('inf')  # exclude current partition

        # Only consider partitions that can accommodate this node
        can_fit = remaining_capacity >= demand_to_move
        if not can_fit.any():
            # No partition can fit this node - choose one with most capacity anyway
            target_k = remaining_capacity.argmax().item()
        else:
            # Choose partition with most remaining capacity that can fit
            remaining_capacity_masked = torch.where(can_fit, remaining_capacity,
                                                    torch.full_like(remaining_capacity, -float('inf')))
            target_k = remaining_capacity_masked.argmax().item()

        # Move node
        X_proj[node_to_move] = target_k
        partition_demands[k] -= demand_to_move
        partition_demands[target_k] += demand_to_move

    return X_proj


def partition_to_subtours(
    X: torch.Tensor,
    coors: torch.Tensor,
    K: int
) -> List[torch.Tensor]:
    """
    Convert partition assignment to list of subtour coordinates.

    Each subtour includes depot (coors[0]) as first node.

    Args:
        X: (n,) partition assignments for customer nodes
        coors: (n+1, 2) coordinates with depot at index 0
        K: number of partitions

    Returns:
        subtours: list of K tensors, each (n_k+1, 2) where n_k is number
                  of customers in partition k. First row is always depot.
    """
    n = X.size(0)
    device = X.device
    depot = coors[0:1]  # (1, 2)

    subtours = []
    for k in range(K):
        # Find customer nodes in partition k
        in_partition = (X == k)
        customer_indices = in_partition.nonzero(as_tuple=True)[0]

        if len(customer_indices) == 0:
            # Empty partition: just depot
            subtours.append(depot.clone())
        else:
            # Customer coordinates (add 1 to index because coors includes depot)
            customer_coors = coors[customer_indices + 1]
            # Prepend depot
            subtour_coors = torch.cat([depot, customer_coors], dim=0)
            subtours.append(subtour_coors)

    return subtours


def compute_subtour_costs_simple(
    subtours: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute closed-loop tour costs using simple greedy nearest neighbor.

    This is a fallback for when batched SA is not available.
    For each subtour: cost = sum of edges in greedy tour (closed loop).

    Args:
        subtours: list of (n_k, 2) tensors

    Returns:
        costs: (K,) tensor of subtour costs
    """
    device = subtours[0].device
    costs = []

    for coors in subtours:
        n = coors.size(0)
        if n <= 1:
            costs.append(torch.tensor(0.0, device=device))
        elif n == 2:
            # Just depot and one customer: closed loop = 2 * distance
            dist = (coors[0] - coors[1]).norm()
            costs.append(2 * dist)
        else:
            # Greedy nearest neighbor from depot
            visited = torch.zeros(n, dtype=torch.bool, device=device)
            tour_cost = torch.tensor(0.0, device=device)
            current = 0  # start at depot
            visited[0] = True

            for _ in range(n - 1):
                # Find nearest unvisited
                dists = (coors[current] - coors).norm(dim=1)
                dists[visited] = float('inf')
                nearest = dists.argmin()
                tour_cost = tour_cost + dists[nearest]
                visited[nearest] = True
                current = nearest.item()

            # Return to depot
            tour_cost = tour_cost + (coors[current] - coors[0]).norm()
            costs.append(tour_cost)

    return torch.stack(costs)


def compute_cvrp_cost(
    X: torch.Tensor,
    coors: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    K: int,
    use_projection: bool = True,
    use_batched_sa: bool = False,
    sa_steps: int = 100,
    penalty_lambda: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total CVRP cost for a partition.

    Cost = sum of sub-TSP costs + (optional) capacity violation penalty

    Args:
        X: (n,) partition assignments for customer nodes
        coors: (n+1, 2) coordinates with depot at index 0
        demands: (n,) customer demands (NOT including depot)
        capacity: vehicle capacity
        K: number of partitions
        use_projection: if True, project to feasible before computing cost
        use_batched_sa: if True, use batched SA solver for sub-TSPs
        sa_steps: SA iterations (if using batched SA)
        penalty_lambda: penalty coefficient for capacity violations

    Returns:
        total_cost: scalar - total CVRP cost
        subtour_costs: (K,) - individual sub-TSP costs
        violation: scalar - capacity violation (0 if projected)
    """
    device = X.device

    # Compute violation before projection
    violation = compute_capacity_violation(X, demands, capacity, K)

    # Project to feasible if requested
    if use_projection:
        X = project_to_feasible(X, demands, capacity, K)
        violation = torch.tensor(0.0, device=device)

    # Convert to subtours
    subtours = partition_to_subtours(X, coors, K)

    # Compute subtour costs
    if use_batched_sa:
        # Import here to avoid circular dependency
        try:
            from utils.batched_tsp import solve_routes_by_bucket
            subtour_costs_list = solve_routes_by_bucket(
                subtours, n_steps=sa_steps, method='sa', device=device
            )
            subtour_costs = torch.stack([c if isinstance(c, torch.Tensor) else torch.tensor(c, device=device)
                                         for c in subtour_costs_list])
        except ImportError:
            # Fallback to simple solver
            subtour_costs = compute_subtour_costs_simple(subtours)
    else:
        subtour_costs = compute_subtour_costs_simple(subtours)

    # Total cost
    total_cost = subtour_costs.sum()

    # Add violation penalty if not projecting
    if not use_projection:
        total_cost = total_cost + penalty_lambda * violation

    return total_cost, subtour_costs, violation


class CVRPEnergy:
    """
    CVRP Energy function for diffusion training.

    Following DiffUCO's EnergyFunctions pattern:
    - calculate_Energy: returns (energy, violation_per_node, constraint_penalty)
    """

    def __init__(
        self,
        use_projection: bool = True,
        use_batched_sa: bool = False,
        sa_steps: int = 100,
        penalty_lambda: float = 10.0
    ):
        """
        Args:
            use_projection: project to feasible before computing cost
            use_batched_sa: use batched SA for sub-TSPs
            sa_steps: SA iterations
            penalty_lambda: capacity violation penalty coefficient
        """
        self.use_projection = use_projection
        self.use_batched_sa = use_batched_sa
        self.sa_steps = sa_steps
        self.penalty_lambda = penalty_lambda

    def calculate_Energy(
        self,
        X: torch.Tensor,
        coors: torch.Tensor,
        demands: torch.Tensor,
        capacity: float,
        K: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate CVRP energy (cost) for partition.

        Matches DiffUCO's calculate_Energy signature:
        Returns (Energy, violation_per_node, constraint_penalty)

        Args:
            X: (n,) partition assignments
            coors: (n+1, 2) coordinates with depot
            demands: (n,) customer demands
            capacity: vehicle capacity
            K: number of partitions
            node_graph_idx: (n,) graph index per node (for batched)
            n_graphs: number of graphs in batch

        Returns:
            energy: (n_graphs,) CVRP cost per graph
            violation_per_node: (n,) capacity violation indicator per node
            constraint_penalty: (n_graphs,) total constraint penalty per graph
        """
        device = X.device

        if node_graph_idx is None or n_graphs == 1:
            # Single graph case
            total_cost, subtour_costs, violation = compute_cvrp_cost(
                X, coors, demands, capacity, K,
                use_projection=self.use_projection,
                use_batched_sa=self.use_batched_sa,
                sa_steps=self.sa_steps,
                penalty_lambda=self.penalty_lambda
            )

            # Violation per node: 1 if node is in overloaded partition
            partition_demands = compute_partition_demands(X, demands, K)
            overloaded = partition_demands > capacity
            violation_per_node = overloaded[X].float()

            return (
                total_cost.unsqueeze(0),
                violation_per_node,
                violation.unsqueeze(0)
            )
        else:
            # Batched case: compute per graph
            energies = []
            violations = []
            violation_per_node_list = []

            for g in range(n_graphs):
                mask = (node_graph_idx == g)
                X_g = X[mask]
                n_g = X_g.size(0)

                # For CVRP, we need per-graph coors and demands
                # This assumes coors and demands are also batched consistently
                # In practice, we'd need separate coors/demands per graph
                # For now, assume single graph and this is a placeholder
                total_cost, subtour_costs, violation = compute_cvrp_cost(
                    X_g, coors, demands[mask] if demands.size(0) == X.size(0) else demands,
                    capacity, K,
                    use_projection=self.use_projection,
                    use_batched_sa=self.use_batched_sa,
                    sa_steps=self.sa_steps,
                    penalty_lambda=self.penalty_lambda
                )

                energies.append(total_cost)
                violations.append(violation)

                # Violation per node
                partition_demands = compute_partition_demands(X_g, demands[mask], K)
                overloaded = partition_demands > capacity
                violation_per_node_list.append(overloaded[X_g].float())

            return (
                torch.stack(energies),
                torch.cat(violation_per_node_list) if violation_per_node_list else torch.tensor([], device=device),
                torch.stack(violations)
            )


def create_energy_fn(
    coors: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    K: int,
    use_projection: bool = True,
    use_batched_sa: bool = False,
    penalty_lambda: float = 10.0
):
    """
    Create an energy function closure for use with trajectory collection.

    Returns a function with signature: energy_fn(X, node_graph_idx, n_graphs) -> energy

    This matches the interface expected by trajectory.py.

    Args:
        coors: (n+1, 2) coordinates with depot
        demands: (n,) customer demands
        capacity: vehicle capacity
        K: number of partitions
        use_projection: project to feasible
        use_batched_sa: use batched SA
        penalty_lambda: violation penalty

    Returns:
        energy_fn: callable(X, node_graph_idx, n_graphs) -> (n_graphs,) energy tensor
    """
    energy_calculator = CVRPEnergy(
        use_projection=use_projection,
        use_batched_sa=use_batched_sa,
        penalty_lambda=penalty_lambda
    )

    def energy_fn(X: torch.Tensor, node_graph_idx: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """Compute energy per graph."""
        energy, _, _ = energy_calculator.calculate_Energy(
            X, coors, demands, capacity, K, node_graph_idx, n_graphs
        )
        return energy

    return energy_fn


##############################################################################
# Tests
##############################################################################

def test_compute_K():
    """Test K computation."""
    print("Testing compute_K...")

    # Test case 1: demands sum to 100, capacity 30
    # K = ceil(100/30) + 1 = ceil(3.33) + 1 = 4 + 1 = 5
    demands = torch.tensor([0, 10, 20, 30, 40])  # depot + 4 customers, total = 100
    K = compute_K(demands, 30.0)
    assert K == 5, f"Expected K=5, got {K}"
    print(f"  demands={demands.tolist()}, capacity=30 -> K={K}")

    # Test case 2: demands sum to 45, capacity 20
    # K = ceil(45/20) + 1 = ceil(2.25) + 1 = 3 + 1 = 4
    demands = torch.tensor([0, 15, 15, 15])  # depot + 3 customers
    K = compute_K(demands, 20.0)
    assert K == 4, f"Expected K=4, got {K}"
    print(f"  demands={demands.tolist()}, capacity=20 -> K={K}")

    # Test case 3: exact fit
    # demands = 60, capacity = 20 -> K = ceil(60/20) + 1 = 3 + 1 = 4
    demands = torch.tensor([0, 20, 20, 20])
    K = compute_K(demands, 20.0)
    assert K == 4, f"Expected K=4, got {K}"
    print(f"  demands={demands.tolist()}, capacity=20 -> K={K}")

    print("[OK] compute_K tests passed\n")


def test_capacity_violation():
    """Test capacity violation computation."""
    print("Testing compute_capacity_violation...")

    # Test case 1: no violation
    X = torch.tensor([0, 1, 2])  # 3 customers in 3 partitions
    demands = torch.tensor([10.0, 10.0, 10.0])
    capacity = 20.0
    K = 3

    violation = compute_capacity_violation(X, demands, capacity, K)
    assert violation.item() == 0, f"Expected 0 violation, got {violation.item()}"
    print(f"  X={X.tolist()}, demands={demands.tolist()}, cap=20 -> violation={violation.item()}")

    # Test case 2: one partition overloaded
    X = torch.tensor([0, 0, 1])  # customers 0,1 in partition 0, customer 2 in partition 1
    demands = torch.tensor([15.0, 15.0, 10.0])  # partition 0 has 30, exceeds 20 by 10
    capacity = 20.0
    K = 2

    violation = compute_capacity_violation(X, demands, capacity, K)
    assert abs(violation.item() - 10.0) < 1e-5, f"Expected 10 violation, got {violation.item()}"
    print(f"  X={X.tolist()}, demands={demands.tolist()}, cap=20 -> violation={violation.item()}")

    # Test case 3: multiple partitions overloaded
    X = torch.tensor([0, 0, 1, 1])
    demands = torch.tensor([15.0, 15.0, 12.0, 12.0])  # partition 0: 30, partition 1: 24
    capacity = 20.0
    K = 2

    violation = compute_capacity_violation(X, demands, capacity, K)
    expected = (30 - 20) + (24 - 20)  # 10 + 4 = 14
    assert abs(violation.item() - expected) < 1e-5, f"Expected {expected} violation, got {violation.item()}"
    print(f"  X={X.tolist()}, demands={demands.tolist()}, cap=20 -> violation={violation.item()}")

    print("[OK] compute_capacity_violation tests passed\n")


def test_project_to_feasible():
    """Test feasibility projection."""
    print("Testing project_to_feasible...")

    # Test case 1: already feasible
    X = torch.tensor([0, 1, 2])
    demands = torch.tensor([10.0, 10.0, 10.0])
    capacity = 20.0
    K = 3

    X_proj = project_to_feasible(X, demands, capacity, K)
    assert (X_proj == X).all(), "Feasible solution should not change"
    print(f"  Already feasible: X={X.tolist()} -> X_proj={X_proj.tolist()}")

    # Test case 2: one overloaded partition
    X = torch.tensor([0, 0, 1])  # partition 0 has demand 30
    demands = torch.tensor([15.0, 15.0, 10.0])
    capacity = 20.0
    K = 3

    X_proj = project_to_feasible(X, demands, capacity, K)

    # Verify feasibility
    partition_demands = compute_partition_demands(X_proj, demands, K)
    assert (partition_demands <= capacity).all(), f"Projection not feasible: {partition_demands.tolist()}"
    print(f"  X={X.tolist()} -> X_proj={X_proj.tolist()}, partition_demands={partition_demands.tolist()}")

    # Test case 3: complex case
    X = torch.tensor([0, 0, 0, 1])  # partition 0 has 3 nodes
    demands = torch.tensor([10.0, 10.0, 10.0, 5.0])  # partition 0: 30, partition 1: 5
    capacity = 15.0
    K = 3

    X_proj = project_to_feasible(X, demands, capacity, K)
    partition_demands = compute_partition_demands(X_proj, demands, K)
    assert (partition_demands <= capacity).all(), f"Projection not feasible: {partition_demands.tolist()}"
    print(f"  X={X.tolist()} -> X_proj={X_proj.tolist()}, partition_demands={partition_demands.tolist()}")

    print("[OK] project_to_feasible tests passed\n")


def test_partition_to_subtours():
    """Test partition to subtour conversion."""
    print("Testing partition_to_subtours...")

    # 4 customers + depot
    coors = torch.tensor([
        [0.0, 0.0],  # depot
        [1.0, 0.0],  # customer 0
        [0.0, 1.0],  # customer 1
        [1.0, 1.0],  # customer 2
        [0.5, 0.5],  # customer 3
    ])

    X = torch.tensor([0, 0, 1, 2])  # customers 0,1 in partition 0, etc.
    K = 3

    subtours = partition_to_subtours(X, coors, K)

    assert len(subtours) == K, f"Expected {K} subtours, got {len(subtours)}"

    # Partition 0: depot + customers 0, 1
    assert subtours[0].shape[0] == 3, f"Partition 0 should have 3 nodes"
    assert (subtours[0][0] == coors[0]).all(), "First node should be depot"

    # Partition 1: depot + customer 2
    assert subtours[1].shape[0] == 2, f"Partition 1 should have 2 nodes"

    # Partition 2: depot + customer 3
    assert subtours[2].shape[0] == 2, f"Partition 2 should have 2 nodes"

    print(f"  Subtour sizes: {[s.shape[0] for s in subtours]}")

    # Test empty partition
    X = torch.tensor([0, 0, 0, 0])  # all in partition 0
    K = 3
    subtours = partition_to_subtours(X, coors, K)
    assert subtours[1].shape[0] == 1, "Empty partition should have just depot"
    assert subtours[2].shape[0] == 1, "Empty partition should have just depot"
    print(f"  Empty partitions: subtour sizes = {[s.shape[0] for s in subtours]}")

    print("[OK] partition_to_subtours tests passed\n")


def test_subtour_costs():
    """Test subtour cost computation."""
    print("Testing compute_subtour_costs_simple...")

    # Simple case: depot + 2 customers in a line
    # depot (0,0) - customer (1,0) - customer (2,0)
    # Optimal closed loop: 0->1->2->0, cost = 1 + 1 + 2 = 4
    coors = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    subtours = [coors]

    costs = compute_subtour_costs_simple(subtours)
    expected = 4.0  # greedy should find optimal here
    assert abs(costs[0].item() - expected) < 1e-5, f"Expected {expected}, got {costs[0].item()}"
    print(f"  Line tour: cost={costs[0].item()}")

    # Single customer
    coors_single = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    costs = compute_subtour_costs_simple([coors_single])
    expected = 2.0  # round trip
    assert abs(costs[0].item() - expected) < 1e-5, f"Expected {expected}, got {costs[0].item()}"
    print(f"  Single customer: cost={costs[0].item()}")

    # Just depot
    coors_depot = torch.tensor([[0.0, 0.0]])
    costs = compute_subtour_costs_simple([coors_depot])
    assert costs[0].item() == 0, f"Expected 0, got {costs[0].item()}"
    print(f"  Just depot: cost={costs[0].item()}")

    print("[OK] compute_subtour_costs_simple tests passed\n")


def test_cvrp_cost():
    """Test full CVRP cost computation."""
    print("Testing compute_cvrp_cost...")

    # Setup: 4 customers, 2 vehicles
    coors = torch.tensor([
        [0.0, 0.0],  # depot
        [1.0, 0.0],  # customer 0
        [2.0, 0.0],  # customer 1
        [0.0, 1.0],  # customer 2
        [0.0, 2.0],  # customer 3
    ])
    demands = torch.tensor([10.0, 10.0, 10.0, 10.0])  # 4 customers
    capacity = 25.0
    K = 2

    # Partition: customers 0,1 in partition 0 (demand 20), customers 2,3 in partition 1 (demand 20)
    X = torch.tensor([0, 0, 1, 1])

    total_cost, subtour_costs, violation = compute_cvrp_cost(
        X, coors, demands, capacity, K, use_projection=True
    )

    print(f"  Total cost: {total_cost.item():.4f}")
    print(f"  Subtour costs: {subtour_costs.tolist()}")
    print(f"  Violation: {violation.item()}")

    assert violation.item() == 0, "Should be feasible"
    assert total_cost.item() > 0, "Cost should be positive"

    # Test with violation
    X = torch.tensor([0, 0, 0, 1])  # partition 0 has demand 30, exceeds 25
    total_cost, subtour_costs, violation = compute_cvrp_cost(
        X, coors, demands, capacity, K, use_projection=False, penalty_lambda=10.0
    )

    expected_violation = 30 - 25  # 5
    assert abs(violation.item() - expected_violation) < 1e-5, f"Expected {expected_violation}, got {violation.item()}"
    print(f"  With violation: cost={total_cost.item():.4f}, violation={violation.item()}")

    print("[OK] compute_cvrp_cost tests passed\n")


def test_cvrp_energy_class():
    """Test CVRPEnergy class."""
    print("Testing CVRPEnergy class...")

    energy_calc = CVRPEnergy(use_projection=True)

    coors = torch.tensor([
        [0.0, 0.0],  # depot
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ])
    demands = torch.tensor([10.0, 10.0, 10.0, 10.0])
    capacity = 25.0
    K = 2

    X = torch.tensor([0, 0, 1, 1])

    energy, violation_per_node, constraint_penalty = energy_calc.calculate_Energy(
        X, coors, demands, capacity, K
    )

    print(f"  Energy: {energy}")
    print(f"  Violation per node: {violation_per_node}")
    print(f"  Constraint penalty: {constraint_penalty}")

    assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"
    assert energy.item() > 0, "Energy should be positive"

    print("[OK] CVRPEnergy class tests passed\n")


def test_create_energy_fn():
    """Test energy function factory."""
    print("Testing create_energy_fn...")

    coors = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
    ])
    demands = torch.tensor([10.0, 10.0, 10.0])
    capacity = 25.0
    K = 2

    energy_fn = create_energy_fn(coors, demands, capacity, K)

    X = torch.tensor([0, 0, 1])
    node_graph_idx = torch.tensor([0, 0, 0])
    n_graphs = 1

    energy = energy_fn(X, node_graph_idx, n_graphs)

    print(f"  Energy from factory: {energy}")
    assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"

    print("[OK] create_energy_fn tests passed\n")


def run_all_tests():
    """Run all CVRP energy tests."""
    print("=" * 60)
    print("CVRP Energy Function Tests")
    print("=" * 60 + "\n")

    test_compute_K()
    test_capacity_violation()
    test_project_to_feasible()
    test_partition_to_subtours()
    test_subtour_costs()
    test_cvrp_cost()
    test_cvrp_energy_class()
    test_create_energy_fn()

    print("=" * 60)
    print("All CVRP Energy tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
