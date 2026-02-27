"""
Chip Placement Energy Function - DiffUCO Faithful Port

Energy = HPWL + overlap_weight * (overlap * HPWL) + boundary_weight * (boundary * HPWL)

Reference: DIffUCO/EnergyFunctions/ChipPlacementEnergy.py
"""

import torch
from typing import Callable, Optional, Tuple

try:
    from .step_model import scatter_sum
except ImportError:
    from step_model import scatter_sum


def _compute_hpwl(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """
    Compute Half-Perimeter Wirelength.

    For each edge (2-pin net): HPWL = |x_sender - x_receiver| + |y_sender - y_receiver|
    Aggregated per graph.

    Args:
        positions: (n_components, 2) component center positions
        edge_index: (2, n_edges) sender/receiver indices
        node_graph_idx: (n_components,) graph assignment
        n_graphs: number of graphs

    Returns:
        hpwl_per_graph: (n_graphs,)
    """
    senders = edge_index[0]
    receivers = edge_index[1]

    sender_pos = positions[senders]    # (n_edges, 2)
    receiver_pos = positions[receivers]  # (n_edges, 2)

    # Bounding box for 2-pin nets
    bbox_width = torch.abs(sender_pos[:, 0] - receiver_pos[:, 0])
    bbox_height = torch.abs(sender_pos[:, 1] - receiver_pos[:, 1])
    hpwl_per_edge = bbox_width + bbox_height  # (n_edges,)

    # Aggregate to graph level via sender node's graph
    edge_graph_idx = node_graph_idx[senders]
    hpwl_per_graph = scatter_sum(hpwl_per_edge, edge_graph_idx, dim=0, dim_size=n_graphs)

    return hpwl_per_graph


def _compute_overlap_penalty(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """
    Compute pairwise overlap penalty (O(n^2)).

    For each pair (i, j) in same graph with i < j:
        overlap_area = max(0, overlap_width) * max(0, overlap_height)

    Args:
        positions: (n_components, 2) center positions
        component_sizes: (n_components, 2) width, height
        node_graph_idx: (n_components,) graph assignment
        n_graphs: number of graphs

    Returns:
        overlap_per_graph: (n_graphs,)
    """
    num_components = positions.shape[0]
    device = positions.device

    # Bounding boxes from center + half-size
    half_sizes = component_sizes / 2.0
    x_min = positions[:, 0] - half_sizes[:, 0]  # (n,)
    y_min = positions[:, 1] - half_sizes[:, 1]
    x_max = positions[:, 0] + half_sizes[:, 0]
    y_max = positions[:, 1] + half_sizes[:, 1]

    # Pairwise overlap (vectorized)
    # Expand for broadcasting: (n, 1) vs (1, n)
    overlap_width = torch.clamp(
        torch.minimum(x_max.unsqueeze(1), x_max.unsqueeze(0)) -
        torch.maximum(x_min.unsqueeze(1), x_min.unsqueeze(0)),
        min=0.0
    )
    overlap_height = torch.clamp(
        torch.minimum(y_max.unsqueeze(1), y_max.unsqueeze(0)) -
        torch.maximum(y_min.unsqueeze(1), y_min.unsqueeze(0)),
        min=0.0
    )
    overlap_area = overlap_width * overlap_height  # (n, n)

    # Upper triangle mask (i < j) to count each pair once
    i_idx = torch.arange(num_components, device=device).unsqueeze(1)
    j_idx = torch.arange(num_components, device=device).unsqueeze(0)
    upper_mask = (i_idx < j_idx).float()

    # Same-graph mask
    same_graph_mask = (node_graph_idx.unsqueeze(1) == node_graph_idx.unsqueeze(0)).float()

    # Apply masks
    overlap_masked = overlap_area * upper_mask * same_graph_mask

    # Sum per component, then aggregate per graph
    overlap_per_component = overlap_masked.sum(dim=1)  # (n,)
    overlap_per_graph = scatter_sum(overlap_per_component, node_graph_idx, dim=0, dim_size=n_graphs)

    return overlap_per_graph


def _compute_boundary_penalty(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    canvas_x_min: float = -1.0,
    canvas_y_min: float = -1.0,
    canvas_width: float = 2.0,
    canvas_height: float = 2.0,
) -> torch.Tensor:
    """
    Compute boundary violation penalty.

    Penalty = sum of out-of-bounds areas weighted by component dimension.

    Args:
        positions: (n_components, 2) center positions
        component_sizes: (n_components, 2) width, height
        node_graph_idx: (n_components,) graph assignment
        n_graphs: number of graphs

    Returns:
        boundary_per_graph: (n_graphs,)
    """
    half_sizes = component_sizes / 2.0
    x_min = positions[:, 0] - half_sizes[:, 0]
    y_min = positions[:, 1] - half_sizes[:, 1]
    x_max = positions[:, 0] + half_sizes[:, 0]
    y_max = positions[:, 1] + half_sizes[:, 1]

    canvas_x_max = canvas_x_min + canvas_width
    canvas_y_max = canvas_y_min + canvas_height

    # Violations (how much extends beyond boundary)
    left_violation = torch.clamp(canvas_x_min - x_min, min=0.0)
    right_violation = torch.clamp(x_max - canvas_x_max, min=0.0)
    bottom_violation = torch.clamp(canvas_y_min - y_min, min=0.0)
    top_violation = torch.clamp(y_max - canvas_y_max, min=0.0)

    # Approximate area: violation_distance * perpendicular component dimension
    x_violation = (left_violation + right_violation) * component_sizes[:, 1]  # * height
    y_violation = (bottom_violation + top_violation) * component_sizes[:, 0]  # * width

    boundary_per_component = x_violation + y_violation
    boundary_per_graph = scatter_sum(boundary_per_component, node_graph_idx, dim=0, dim_size=n_graphs)

    return boundary_per_graph


def compute_chip_placement_energy(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    overlap_weight: float = 10.0,
    boundary_weight: float = 10.0,
    canvas_x_min: float = -1.0,
    canvas_y_min: float = -1.0,
    canvas_width: float = 2.0,
    canvas_height: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total chip placement energy.

    Energy = HPWL + overlap_weight * (overlap * HPWL) + boundary_weight * (boundary * HPWL)

    Faithful port of DIffUCO/EnergyFunctions/ChipPlacementEnergy.py

    Args:
        positions: (n_components, 2) component center positions
        component_sizes: (n_components, 2) width, height
        edge_index: (2, n_edges) netlist edges
        node_graph_idx: (n_components,) graph assignment
        n_graphs: number of graphs
        overlap_weight: penalty weight for overlap (multiplied by HPWL)
        boundary_weight: penalty weight for boundary (multiplied by HPWL)

    Returns:
        energy_per_graph: (n_graphs,) total energy
        hpwl_per_graph: (n_graphs,) HPWL component
        overlap_per_graph: (n_graphs,) raw overlap area
        boundary_per_graph: (n_graphs,) raw boundary violation
    """
    hpwl_per_graph = _compute_hpwl(positions, edge_index, node_graph_idx, n_graphs)

    overlap_per_graph = _compute_overlap_penalty(
        positions, component_sizes, node_graph_idx, n_graphs
    )

    boundary_per_graph = _compute_boundary_penalty(
        positions, component_sizes, node_graph_idx, n_graphs,
        canvas_x_min, canvas_y_min, canvas_width, canvas_height
    )

    # HPWL-normalized penalties (faithful to DIffUCO)
    normalized_overlap = overlap_per_graph * hpwl_per_graph
    normalized_boundary = boundary_per_graph * hpwl_per_graph

    energy_per_graph = (
        hpwl_per_graph
        + overlap_weight * normalized_overlap
        + boundary_weight * normalized_boundary
    )

    return energy_per_graph, hpwl_per_graph, overlap_per_graph, boundary_per_graph


def create_chip_placement_energy_fn(
    component_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    overlap_weight: float = 10.0,
    boundary_weight: float = 10.0,
    canvas_x_min: float = -1.0,
    canvas_y_min: float = -1.0,
    canvas_width: float = 2.0,
    canvas_height: float = 2.0,
) -> Callable:
    """
    Factory: create energy_fn matching trajectory.py interface.

    Returns:
        energy_fn(positions, node_graph_idx, n_graphs) -> (n_graphs,)
    """
    def energy_fn(positions, node_graph_idx, n_graphs):
        # Handle shape: positions may be (n_components,) flattened or (n_components, 2)
        if positions.dim() == 1:
            positions = positions.reshape(-1, 2)

        energy, _, _, _ = compute_chip_placement_energy(
            positions, component_sizes, edge_index,
            node_graph_idx, n_graphs,
            overlap_weight, boundary_weight,
            canvas_x_min, canvas_y_min, canvas_width, canvas_height
        )
        return energy

    return energy_fn


def test_chip_placement_energy():
    """Test energy computation with a simple hand-crafted instance."""
    print("=" * 60)
    print("Testing Chip Placement Energy")
    print("=" * 60)

    # 4 components in a single graph
    n_components = 4
    n_graphs = 1
    node_graph_idx = torch.zeros(n_components, dtype=torch.long)

    # Component sizes: all 0.4 x 0.4
    component_sizes = torch.full((n_components, 2), 0.4)

    # Netlist: 0-1, 1-2, 2-3 (chain)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                [1, 0, 2, 1, 3, 2]], dtype=torch.long)

    # Test 1: Non-overlapping placement
    positions_good = torch.tensor([
        [-0.5, -0.5],
        [0.0, -0.5],
        [0.0, 0.0],
        [0.5, 0.0],
    ], dtype=torch.float32)

    energy, hpwl, overlap, boundary = compute_chip_placement_energy(
        positions_good, component_sizes, edge_index, node_graph_idx, n_graphs
    )
    print(f"\nGood placement (spread out):")
    print(f"  HPWL: {hpwl.item():.4f}")
    print(f"  Overlap: {overlap.item():.4f}")
    print(f"  Boundary: {boundary.item():.4f}")
    print(f"  Energy: {energy.item():.4f}")

    # Test 2: All stacked at origin (degenerate)
    positions_stacked = torch.zeros(n_components, 2)
    energy_s, hpwl_s, overlap_s, boundary_s = compute_chip_placement_energy(
        positions_stacked, component_sizes, edge_index, node_graph_idx, n_graphs
    )
    print(f"\nStacked placement (all at origin):")
    print(f"  HPWL: {hpwl_s.item():.4f}")
    print(f"  Overlap: {overlap_s.item():.4f}")
    print(f"  Boundary: {boundary_s.item():.4f}")
    print(f"  Energy: {energy_s.item():.4f}")
    print(f"  NOTE: HPWL=0 makes overlap*HPWL=0 (known degenerate case)")

    # Test 3: Component outside boundary
    positions_oob = torch.tensor([
        [-1.2, 0.0],  # left of canvas
        [0.0, 0.0],
        [0.0, 1.2],   # top of canvas
        [0.5, 0.0],
    ], dtype=torch.float32)
    energy_b, hpwl_b, overlap_b, boundary_b = compute_chip_placement_energy(
        positions_oob, component_sizes, edge_index, node_graph_idx, n_graphs
    )
    print(f"\nOut-of-bounds placement:")
    print(f"  HPWL: {hpwl_b.item():.4f}")
    print(f"  Overlap: {overlap_b.item():.4f}")
    print(f"  Boundary: {boundary_b.item():.4f}")
    print(f"  Energy: {energy_b.item():.4f}")

    # Test 4: Gradient flow
    positions_grad = positions_good.clone().requires_grad_(True)
    energy_g, _, _, _ = compute_chip_placement_energy(
        positions_grad, component_sizes, edge_index, node_graph_idx, n_graphs
    )
    energy_g.sum().backward()
    grad_ok = positions_grad.grad is not None and not torch.isnan(positions_grad.grad).any()
    print(f"\nGradient flow: {'OK' if grad_ok else 'FAIL'}")

    # Test 5: Factory function interface
    energy_fn = create_chip_placement_energy_fn(component_sizes, edge_index)
    energy_factory = energy_fn(positions_good, node_graph_idx, n_graphs)
    factory_match = torch.allclose(energy_factory, energy, atol=1e-6)
    print(f"Factory function: {'OK' if factory_match else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Chip Placement Energy tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_chip_placement_energy()
