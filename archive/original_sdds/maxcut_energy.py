"""
MaxCut Energy - DiffUCO Faithful Implementation

MaxCut is simpler than TSP:
- Binary variables (0/1) for each node
- Energy = negative cut value (we minimize, so negative of edges between partitions)
- No validity constraints

Reference: DIffUCO/EnergyFunctions/MIS_Energy.py (similar structure)
"""

import torch
from typing import Tuple, Callable

try:
    from .step_model import scatter_sum
except ImportError:
    from step_model import scatter_sum


def compute_maxcut_energy(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """
    Compute MaxCut energy (negative cut value).

    Cut value = sum of edge weights where endpoints are in different partitions.
    Energy = -cut_value (since we minimize energy, maximizing cut).

    Args:
        X: Node assignments (0 or 1) - shape (n_nodes,)
        edge_index: Edge list (2, n_edges)
        edge_weights: Edge weights (n_edges,) or (n_edges, 1)
        node_graph_idx: Graph index per node (n_nodes,)
        n_graphs: Number of graphs

    Returns:
        energy: Per-graph energy (n_graphs,) - negative cut value
    """
    device = X.device

    # Get edge endpoints
    src, dst = edge_index[0], edge_index[1]

    # Get assignments for each edge endpoint
    X_src = X[src]  # (n_edges,)
    X_dst = X[dst]  # (n_edges,)

    # Edge is in cut if endpoints differ
    in_cut = (X_src != X_dst).float()  # (n_edges,)

    # Weight the cut edges
    if edge_weights.dim() == 2:
        edge_weights = edge_weights.squeeze(-1)
    weighted_cut = in_cut * edge_weights  # (n_edges,)

    # Aggregate per graph
    # Map each edge to its graph (use source node's graph)
    edge_graph_idx = node_graph_idx[src]

    cut_per_graph = scatter_sum(weighted_cut, edge_graph_idx, dim=0, dim_size=n_graphs)

    # Energy = negative cut (we want to maximize cut, but minimize energy)
    energy = -cut_per_graph

    return energy


def compute_maxcut_cut_value(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cut value (positive, for reporting).

    Args:
        X: Node assignments (0 or 1)
        edge_index: Edge list
        edge_weights: Edge weights

    Returns:
        cut_value: Total cut value (scalar)
    """
    src, dst = edge_index[0], edge_index[1]
    X_src = X[src]
    X_dst = X[dst]

    in_cut = (X_src != X_dst).float()

    if edge_weights.dim() == 2:
        edge_weights = edge_weights.squeeze(-1)

    cut_value = (in_cut * edge_weights).sum()

    return cut_value


def create_maxcut_energy_fn(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
) -> Callable:
    """
    Create energy function for MaxCut.

    Args:
        edge_index: Graph edges (2, n_edges)
        edge_weights: Edge weights (n_edges,)

    Returns:
        energy_fn: Function(X, node_graph_idx, n_graphs) -> energy
    """
    def energy_fn(X, node_graph_idx, n_graphs):
        return compute_maxcut_energy(
            X, edge_index, edge_weights, node_graph_idx, n_graphs
        )
    return energy_fn


def generate_random_graph(
    n_nodes: int,
    edge_prob: float = 0.5,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random Erdos-Renyi graph.

    Args:
        n_nodes: Number of nodes
        edge_prob: Probability of edge between any two nodes
        device: Device

    Returns:
        edge_index: (2, n_edges)
        edge_weights: (n_edges,) - all ones
    """
    if device is None:
        device = torch.device('cpu')

    # Generate random adjacency
    senders = []
    receivers = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if torch.rand(1).item() < edge_prob:
                # Add both directions
                senders.extend([i, j])
                receivers.extend([j, i])

    if len(senders) == 0:
        # Ensure at least one edge
        senders = [0, 1]
        receivers = [1, 0]

    edge_index = torch.tensor([senders, receivers], dtype=torch.long, device=device)
    edge_weights = torch.ones(edge_index.size(1), device=device)

    return edge_index, edge_weights


def generate_batch_random_graphs(
    n_nodes: int,
    n_graphs: int,
    edge_prob: float = 0.5,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate batch of random graphs.

    Args:
        n_nodes: Nodes per graph
        n_graphs: Number of graphs
        edge_prob: Edge probability
        device: Device

    Returns:
        edge_index: (2, total_edges)
        edge_weights: (total_edges,)
        node_graph_idx: (total_nodes,)
    """
    if device is None:
        device = torch.device('cpu')

    all_senders = []
    all_receivers = []
    all_weights = []

    for g in range(n_graphs):
        offset = g * n_nodes

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if torch.rand(1).item() < edge_prob:
                    all_senders.extend([offset + i, offset + j])
                    all_receivers.extend([offset + j, offset + i])
                    all_weights.extend([1.0, 1.0])

        # Ensure at least one edge per graph
        if len(all_senders) == 0 or (len(all_senders) > 0 and all_senders[-1] < offset):
            all_senders.extend([offset, offset + 1])
            all_receivers.extend([offset + 1, offset])
            all_weights.extend([1.0, 1.0])

    edge_index = torch.tensor([all_senders, all_receivers], dtype=torch.long, device=device)
    edge_weights = torch.tensor(all_weights, device=device)
    node_graph_idx = torch.repeat_interleave(
        torch.arange(n_graphs, device=device), n_nodes
    )

    return edge_index, edge_weights, node_graph_idx


def test_maxcut_energy():
    """Test MaxCut energy computation."""
    print("=" * 60)
    print("Testing MaxCut Energy")
    print("=" * 60)

    # Simple 4-node graph: 0-1-2-3 (path)
    n_nodes = 4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ])
    edge_weights = torch.ones(6)
    node_graph_idx = torch.zeros(4, dtype=torch.long)

    # Test 1: All same partition (cut = 0)
    X_same = torch.tensor([0, 0, 0, 0])
    energy_same = compute_maxcut_energy(X_same, edge_index, edge_weights, node_graph_idx, 1)
    cut_same = compute_maxcut_cut_value(X_same, edge_index, edge_weights)

    print(f"\nAll same partition: X = {X_same.tolist()}")
    print(f"  Cut value: {cut_same.item()}")
    print(f"  Energy: {energy_same.item()}")
    assert cut_same.item() == 0, "Same partition should have 0 cut"
    print("  [OK]")

    # Test 2: Alternating (max cut for path)
    X_alt = torch.tensor([0, 1, 0, 1])
    energy_alt = compute_maxcut_energy(X_alt, edge_index, edge_weights, node_graph_idx, 1)
    cut_alt = compute_maxcut_cut_value(X_alt, edge_index, edge_weights)

    print(f"\nAlternating: X = {X_alt.tolist()}")
    print(f"  Cut value: {cut_alt.item()}")
    print(f"  Energy: {energy_alt.item()}")
    # Path 0-1-2-3 with alternating: all 3 edges are in cut
    # But we have bidirectional edges, so 6 edges total, 6 in cut
    assert cut_alt.item() == 6, f"Alternating should have cut=6, got {cut_alt.item()}"
    print("  [OK]")

    # Test 3: Two partitions
    X_half = torch.tensor([0, 0, 1, 1])
    energy_half = compute_maxcut_energy(X_half, edge_index, edge_weights, node_graph_idx, 1)
    cut_half = compute_maxcut_cut_value(X_half, edge_index, edge_weights)

    print(f"\nHalf-half: X = {X_half.tolist()}")
    print(f"  Cut value: {cut_half.item()}")
    print(f"  Energy: {energy_half.item()}")
    # Only edge 1-2 is in cut (bidirectional = 2 edges)
    assert cut_half.item() == 2, f"Half should have cut=2, got {cut_half.item()}"
    print("  [OK]")

    # Test 4: Batch of graphs
    print("\nTesting batch computation...")
    edge_index_batch, edge_weights_batch, node_graph_idx_batch = generate_batch_random_graphs(
        n_nodes=10, n_graphs=3, edge_prob=0.3
    )

    X_batch = torch.randint(0, 2, (30,))  # 3 graphs * 10 nodes
    energy_batch = compute_maxcut_energy(
        X_batch, edge_index_batch, edge_weights_batch, node_graph_idx_batch, 3
    )

    print(f"  Batch energy shape: {energy_batch.shape}")
    print(f"  Batch energies: {energy_batch.tolist()}")
    assert energy_batch.shape == (3,), "Batch energy should be (n_graphs,)"
    print("  [OK]")

    print("\n" + "=" * 60)
    print("All MaxCut tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_maxcut_energy()
