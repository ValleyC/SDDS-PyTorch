"""
TSP Energy - DiffUCO Faithful Implementation

This module implements TSP tour length computation as energy for diffusion model training.
The diffusion model assigns each node to a position in the tour (0 to K-1).

Energy = Tour_Length + A * Position_Constraint_Penalty + A * Node_Constraint_Penalty

Reference: DIffUCO/EnergyFunctions/TSPEnergy.py
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import math

try:
    from .step_model import scatter_sum
except ImportError:
    from step_model import scatter_sum


class TSPEnergy:
    """
    TSP Energy computation matching DiffUCO's TSPEnergyClass.

    The model assigns each node to a position in the tour (class 0 to K-1).
    Energy components:
    - Tour length: sum of distances between consecutive tour positions
    - Position constraint: each position should have exactly one node
    - Node constraint: each node should be assigned exactly one position
    """

    def __init__(self, n_classes: int, penalty_coeff: float = 1.45):
        """
        Args:
            n_classes: Number of tour positions (typically = number of nodes per graph)
            penalty_coeff: Coefficient for constraint violation penalties (A in DiffUCO)
        """
        self.n_classes = n_classes
        self.penalty_coeff = penalty_coeff

        # Cyclic permutation matrix for getting "next position in tour"
        # This shifts indices by 1: position i -> position (i+1) mod K
        self.cycl_perm_mat = torch.zeros(n_classes, n_classes)
        for i in range(n_classes - 1):
            self.cycl_perm_mat[i + 1, i] = 1.0
        self.cycl_perm_mat[0, -1] = 1.0  # Wrap around

    def compute_energy_from_positions(
        self,
        positions: torch.Tensor,
        X_0_classes: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TSP energy from node positions (coordinate-based).

        This matches DiffUCO's calculate_Energy_per_instance exactly.

        Args:
            positions: Node coordinates (total_nodes, 2)
            X_0_classes: Tour position assignment per node (total_nodes,) with values 0 to K-1
            node_graph_idx: Graph index for each node (total_nodes,)
            n_graphs: Number of graphs in batch

        Returns:
            energy: Total energy per graph (n_graphs,)
            violations_per_node: Constraint violations per node (total_nodes,)
            penalty_per_graph: Constraint penalty per graph (n_graphs,)
        """
        device = positions.device

        # Move cyclic permutation matrix to device
        cycl_perm_mat = self.cycl_perm_mat.to(device)

        # Get number of nodes per graph
        n_node_per_graph = scatter_sum(
            torch.ones(X_0_classes.size(0), device=device),
            node_graph_idx, dim=0, dim_size=n_graphs
        )

        # Process each graph separately (since they may have different sizes)
        # For now, assume all graphs have same size = n_classes
        n_nodes_per_graph = self.n_classes

        # Reshape for per-graph computation
        # positions: (n_graphs, n_nodes_per_graph, 2)
        # X_0_classes: (n_graphs, n_nodes_per_graph)
        total_nodes = positions.size(0)
        positions_reshaped = positions.view(n_graphs, n_nodes_per_graph, 2)
        X_classes_reshaped = X_0_classes.view(n_graphs, n_nodes_per_graph)

        # Compute energy for each graph
        energies = []
        violations = []
        penalties = []

        for g in range(n_graphs):
            pos_g = positions_reshaped[g]  # (n_nodes, 2)
            X_g = X_classes_reshaped[g]  # (n_nodes,)

            energy_g, viol_g, penalty_g = self._compute_energy_single_graph(
                pos_g, X_g, cycl_perm_mat
            )
            energies.append(energy_g)
            violations.append(viol_g)
            penalties.append(penalty_g)

        energy = torch.stack(energies)  # (n_graphs,)
        violations_per_node = torch.cat(violations)  # (total_nodes,)
        penalty_per_graph = torch.stack(penalties)  # (n_graphs,)

        return energy, violations_per_node, penalty_per_graph

    def _compute_energy_single_graph(
        self,
        positions: torch.Tensor,
        X_0_classes: torch.Tensor,
        cycl_perm_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute energy for a single graph.

        Matches DiffUCO's calculate_Energy_per_instance (lines 87-108).

        Args:
            positions: Node coordinates (n_nodes, 2)
            X_0_classes: Tour position per node (n_nodes,)
            cycl_perm_mat: Cyclic permutation matrix (K, K)

        Returns:
            energy: Total energy (scalar)
            violations_per_node: Constraint violations per node (n_nodes,)
            penalty: Total constraint penalty (scalar)
        """
        n_nodes = positions.size(0)
        device = positions.device

        # One-hot encoding of tour positions
        # x_mat[i, k] = 1 if node i is at position k in the tour
        x_mat = F.one_hot(X_0_classes.long(), num_classes=self.n_classes).float()
        # x_mat: (n_nodes, K)

        # Distance matrix: pairwise Euclidean distances
        # distance_matrix[i, j] = ||pos_i - pos_j||
        diff = positions[:, None, :] - positions[None, :, :]  # (n_nodes, n_nodes, 2)
        distance_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)  # (n_nodes, n_nodes)

        # Cyclic shift: x_mat_cycl[i, k] = x_mat[i, (k-1) mod K]
        # This represents "node i is at position (k-1)", or equivalently
        # "the node at position k is followed by node at position (k+1)"
        x_mat_cycl = torch.matmul(x_mat, cycl_perm_mat)  # (n_nodes, K)

        # H_mat[i, j] = sum_k x_mat_cycl[i, k] * x_mat[j, k]
        # This is 1 if node i is at position (k-1) and node j is at position k
        # i.e., node j follows node i in the tour
        H_mat = torch.matmul(x_mat_cycl, x_mat.t())  # (n_nodes, n_nodes)

        # Tour length: sum of distances for edges in the tour
        tour_length = (H_mat * distance_matrix).sum()

        # Constraint 1: Each position should be used exactly once
        # Obj1[k] = (sum_i x_mat[i, k] - 1)^2
        pos_usage = x_mat.sum(dim=0)  # (K,) - how many nodes at each position
        Obj1 = ((pos_usage - 1) ** 2).sum()

        # Constraint 2: Each node should have exactly one position
        # Obj2[i] = (sum_k x_mat[i, k] - 1)^2
        node_assignment = x_mat.sum(dim=1)  # (n_nodes,) - positions per node
        Obj2 = ((node_assignment - 1) ** 2).sum()

        # Constraint violations per node (for debugging/analysis)
        # Check if each node's assignment collides with others
        X_expanded = X_0_classes[:, None]  # (n_nodes, 1)
        X_other = X_0_classes[None, :]  # (1, n_nodes)
        collision = (X_expanded == X_other).float()  # (n_nodes, n_nodes)
        violations_per_node = collision.sum(dim=1) - 1  # -1 to exclude self

        # Penalties
        penalty_pos = self.penalty_coeff * Obj1  # Position constraint penalty
        penalty_node = self.penalty_coeff * Obj2  # Node constraint penalty

        # Total energy
        energy = tour_length + penalty_pos + penalty_node

        return energy, violations_per_node, penalty_pos

    def compute_energy_from_distances(
        self,
        distance_matrix: torch.Tensor,
        X_0_classes: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TSP energy from pre-computed distance matrix.

        Alternative to coordinate-based computation when distances are given.

        Args:
            distance_matrix: Pairwise distances (n_nodes, n_nodes) for single graph
                            or list of distance matrices for batch
            X_0_classes: Tour position assignment per node (total_nodes,)
            node_graph_idx: Graph index for each node (total_nodes,)
            n_graphs: Number of graphs in batch

        Returns:
            energy: Total energy per graph (n_graphs,)
            violations_per_node: Constraint violations per node (total_nodes,)
            penalty_per_graph: Constraint penalty per graph (n_graphs,)
        """
        device = X_0_classes.device
        cycl_perm_mat = self.cycl_perm_mat.to(device)

        n_nodes_per_graph = self.n_classes
        X_classes_reshaped = X_0_classes.view(n_graphs, n_nodes_per_graph)

        energies = []
        violations = []
        penalties = []

        for g in range(n_graphs):
            X_g = X_classes_reshaped[g]

            # Get distance matrix for this graph
            if distance_matrix.dim() == 2:
                # Single distance matrix for all graphs (same structure)
                dist_g = distance_matrix
            else:
                # Batch of distance matrices
                dist_g = distance_matrix[g]

            energy_g, viol_g, penalty_g = self._compute_energy_from_dist_single(
                dist_g, X_g, cycl_perm_mat
            )
            energies.append(energy_g)
            violations.append(viol_g)
            penalties.append(penalty_g)

        energy = torch.stack(energies)
        violations_per_node = torch.cat(violations)
        penalty_per_graph = torch.stack(penalties)

        return energy, violations_per_node, penalty_per_graph

    def _compute_energy_from_dist_single(
        self,
        distance_matrix: torch.Tensor,
        X_0_classes: torch.Tensor,
        cycl_perm_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute energy for single graph from distance matrix.
        """
        n_nodes = X_0_classes.size(0)
        device = X_0_classes.device

        # One-hot encoding
        x_mat = F.one_hot(X_0_classes.long(), num_classes=self.n_classes).float()

        # Cyclic shift
        x_mat_cycl = torch.matmul(x_mat, cycl_perm_mat)

        # Tour adjacency matrix
        H_mat = torch.matmul(x_mat_cycl, x_mat.t())

        # Tour length
        tour_length = (H_mat * distance_matrix).sum()

        # Constraints
        pos_usage = x_mat.sum(dim=0)
        Obj1 = ((pos_usage - 1) ** 2).sum()

        node_assignment = x_mat.sum(dim=1)
        Obj2 = ((node_assignment - 1) ** 2).sum()

        # Violations
        X_expanded = X_0_classes[:, None]
        X_other = X_0_classes[None, :]
        collision = (X_expanded == X_other).float()
        violations_per_node = collision.sum(dim=1) - 1

        # Penalties
        penalty_pos = self.penalty_coeff * Obj1
        penalty_node = self.penalty_coeff * Obj2

        energy = tour_length + penalty_pos + penalty_node

        return energy, violations_per_node, penalty_pos


def create_tsp_energy_fn(
    n_classes: int,
    positions: Optional[torch.Tensor] = None,
    distance_matrix: Optional[torch.Tensor] = None,
    penalty_coeff: float = 1.45,
) -> Callable:
    """
    Create an energy function for TSP that can be passed to trajectory collection.

    Args:
        n_classes: Number of tour positions
        positions: Node coordinates (n_graphs * n_nodes, 2) - optional
        distance_matrix: Pairwise distances - optional
        penalty_coeff: Constraint penalty coefficient

    Returns:
        energy_fn: Function(X_0, node_graph_idx, n_graphs) -> (energy, violations, penalty)
    """
    tsp_energy = TSPEnergy(n_classes, penalty_coeff)

    if positions is not None:
        def energy_fn(X_0, node_graph_idx, n_graphs):
            return tsp_energy.compute_energy_from_positions(
                positions, X_0, node_graph_idx, n_graphs
            )
        return energy_fn
    elif distance_matrix is not None:
        def energy_fn(X_0, node_graph_idx, n_graphs):
            return tsp_energy.compute_energy_from_distances(
                distance_matrix, X_0, node_graph_idx, n_graphs
            )
        return energy_fn
    else:
        raise ValueError("Either positions or distance_matrix must be provided")


def compute_tour_length_only(
    positions: torch.Tensor,
    X_0_classes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute just the tour length (without constraint penalties).

    Useful for evaluating solution quality after training.

    Args:
        positions: Node coordinates (n_nodes, 2)
        X_0_classes: Tour position per node (n_nodes,)

    Returns:
        tour_length: Length of the tour
    """
    n_nodes = positions.size(0)

    # One-hot encoding
    x_mat = F.one_hot(X_0_classes.long(), num_classes=n_nodes).float()

    # Distance matrix
    diff = positions[:, None, :] - positions[None, :, :]
    distance_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)

    # Cyclic permutation matrix
    cycl_perm_mat = torch.zeros(n_nodes, n_nodes, device=positions.device)
    for i in range(n_nodes - 1):
        cycl_perm_mat[i + 1, i] = 1.0
    cycl_perm_mat[0, -1] = 1.0

    # Tour adjacency
    x_mat_cycl = torch.matmul(x_mat, cycl_perm_mat)
    H_mat = torch.matmul(x_mat_cycl, x_mat.t())

    # Tour length
    tour_length = (H_mat * distance_matrix).sum()

    return tour_length


def check_tour_validity(X_0_classes: torch.Tensor, n_classes: int) -> Tuple[bool, dict]:
    """
    Check if the assignment forms a valid TSP tour (permutation).

    Args:
        X_0_classes: Tour position per node (n_nodes,)
        n_classes: Number of positions (should equal n_nodes)

    Returns:
        is_valid: True if valid permutation
        info: Dictionary with violation details
    """
    n_nodes = X_0_classes.size(0)

    # Each position should appear exactly once
    position_counts = torch.bincount(X_0_classes.long(), minlength=n_classes)
    pos_violations = (position_counts != 1).sum().item()

    # Each node should have one position (always true if X_0_classes has one value per node)
    node_violations = 0

    is_valid = pos_violations == 0 and node_violations == 0

    info = {
        'is_valid': is_valid,
        'position_violations': pos_violations,
        'node_violations': node_violations,
        'position_counts': position_counts.tolist(),
    }

    return is_valid, info


def test_tsp_energy():
    """
    Test TSP energy computation.
    """
    print("=" * 60)
    print("Testing TSP Energy Computation")
    print("=" * 60)

    # Test parameters
    n_nodes = 5
    n_graphs = 2
    n_classes = n_nodes  # TSP: positions = nodes

    print(f"\nTest parameters:")
    print(f"  n_nodes per graph: {n_nodes}")
    print(f"  n_graphs: {n_graphs}")

    # Create random node positions (2D)
    torch.manual_seed(42)
    positions = torch.rand(n_graphs * n_nodes, 2)

    # Create valid tour assignment (permutation for each graph)
    # Graph 0: nodes assigned to positions [0, 1, 2, 3, 4]
    # Graph 1: nodes assigned to positions [2, 0, 4, 1, 3]
    X_0_classes = torch.tensor([0, 1, 2, 3, 4, 2, 0, 4, 1, 3])

    node_graph_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Create energy function
    tsp_energy = TSPEnergy(n_classes, penalty_coeff=1.45)

    # Compute energy
    energy, violations, penalty = tsp_energy.compute_energy_from_positions(
        positions, X_0_classes, node_graph_idx, n_graphs
    )

    print(f"\nResults for valid tours:")
    print(f"  Energy per graph: {energy.tolist()}")
    print(f"  Penalty per graph: {penalty.tolist()}")
    print(f"  Max violations per node: {violations.max().item()}")

    # Verify valid tour has zero violations
    valid_ok = violations.max().item() == 0
    print(f"\n[{'OK' if valid_ok else 'FAIL'}] Valid tour has zero violations: {valid_ok}")

    # Test with invalid tour (repeated positions)
    print("\nTesting invalid tour (position 0 used twice in graph 0):")
    X_0_invalid = torch.tensor([0, 0, 2, 3, 4, 2, 0, 4, 1, 3])  # First two nodes both at position 0

    energy_inv, violations_inv, penalty_inv = tsp_energy.compute_energy_from_positions(
        positions, X_0_invalid, node_graph_idx, n_graphs
    )

    print(f"  Energy per graph: {energy_inv.tolist()}")
    print(f"  Penalty per graph: {penalty_inv.tolist()}")
    print(f"  Max violations: {violations_inv.max().item()}")

    # Invalid tour should have higher penalty
    penalty_increased = penalty_inv[0] > penalty[0]
    print(f"\n[{'OK' if penalty_increased else 'FAIL'}] Invalid tour has higher penalty: {penalty_increased}")

    # Test tour length computation
    print("\nTesting pure tour length (no penalties):")
    pos_single = positions[:n_nodes]  # First graph only
    X_single = X_0_classes[:n_nodes]
    tour_len = compute_tour_length_only(pos_single, X_single)
    print(f"  Tour length: {tour_len.item():.4f}")

    # Verify tour length matches energy component (when tour is valid)
    # For valid tour, energy = tour_length + 0 (no penalties)
    tour_len_ok = abs(tour_len.item() - energy[0].item()) < 0.01
    print(f"[{'OK' if tour_len_ok else 'FAIL'}] Tour length matches energy for valid tour: {tour_len_ok}")

    # Test validity checker
    print("\nTesting tour validity checker:")
    is_valid, info = check_tour_validity(X_single, n_classes)
    print(f"  Valid tour: is_valid={is_valid}, position_violations={info['position_violations']}")

    is_invalid, info_inv = check_tour_validity(X_0_invalid[:n_nodes], n_classes)
    print(f"  Invalid tour: is_valid={is_invalid}, position_violations={info_inv['position_violations']}")

    validity_ok = is_valid and not is_invalid
    print(f"[{'OK' if validity_ok else 'FAIL'}] Validity checker works: {validity_ok}")

    # Test gradient flow
    print("\nTesting gradient flow:")
    positions_grad = positions.clone().requires_grad_(True)
    X_0_float = X_0_classes.float().requires_grad_(False)  # Discrete, no grad

    # Use soft assignment for gradient test
    soft_assignment = torch.randn(n_graphs * n_nodes, n_classes, requires_grad=True)
    soft_probs = F.softmax(soft_assignment, dim=-1)

    # Compute energy with soft assignment
    # This tests that the computation graph is differentiable
    x_mat = soft_probs.view(n_graphs, n_nodes, n_classes)

    # For first graph
    pos_g = positions_grad[:n_nodes]
    x_mat_g = x_mat[0]

    # Distance matrix
    diff = pos_g[:, None, :] - pos_g[None, :, :]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)

    # Cyclic perm
    cycl = tsp_energy.cycl_perm_mat.to(positions.device)
    x_cycl = torch.matmul(x_mat_g, cycl)
    H = torch.matmul(x_cycl, x_mat_g.t())

    tour_len_soft = (H * dist).sum()
    tour_len_soft.backward()

    grad_ok = soft_assignment.grad is not None and positions_grad.grad is not None
    print(f"  Soft assignment has gradient: {soft_assignment.grad is not None}")
    print(f"  Positions have gradient: {positions_grad.grad is not None}")
    print(f"[{'OK' if grad_ok else 'FAIL'}] Gradient flow works: {grad_ok}")

    # Summary
    all_ok = valid_ok and penalty_increased and tour_len_ok and validity_ok and grad_ok
    print(f"\n{'=' * 60}")
    print(f"[{'OK' if all_ok else 'FAIL'}] All TSP energy tests passed: {all_ok}")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    test_tsp_energy()
