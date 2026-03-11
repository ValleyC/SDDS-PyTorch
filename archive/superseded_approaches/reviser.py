"""
Reviser — Local Improvement + Congestion

GLOP-style post-placement improvement:
  - local_improvement: HPWL-driven swap/shift local search (legality-preserving)
  - compute_congestion: RUDY-style density penalty
  - compute_cost: combined cost = HPWL + lambda * congestion

Separated from chip_placement_energy.py to keep reward code pure.
"""

import torch
from typing import Optional, Tuple

try:
    from .chip_placement_energy import (
        compute_chip_placement_energy, _compute_hpwl, scatter_sum
    )
except ImportError:
    from chip_placement_energy import (
        compute_chip_placement_energy, _compute_hpwl, scatter_sum
    )


def compute_congestion(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    grid_size: int = 8,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> torch.Tensor:
    """
    RUDY-style density/congestion penalty.

    Overlays a coarse G×G grid on the canvas. For each cell, sums the
    area contribution of all components overlapping it. Congestion is
    the sum of squared excess density above a uniform target.

    Args:
        positions: (N, 2) component center positions
        component_sizes: (N, 2) width, height
        node_graph_idx: (N,) graph assignment
        n_graphs: number of graphs
        grid_size: coarse grid resolution for density estimation
        canvas_min, canvas_max: canvas bounds

    Returns:
        congestion_per_graph: (n_graphs,) congestion penalty
    """
    device = positions.device
    N = positions.shape[0]
    G = grid_size
    cell_size = (canvas_max - canvas_min) / G
    cell_area = cell_size * cell_size

    # Grid cell centers
    cx = torch.linspace(
        canvas_min + cell_size / 2, canvas_max - cell_size / 2, G, device=device)
    cy = torch.linspace(
        canvas_min + cell_size / 2, canvas_max - cell_size / 2, G, device=device)
    # Cell bounds: (G,)
    cell_xmin = cx - cell_size / 2
    cell_xmax = cx + cell_size / 2
    cell_ymin = cy - cell_size / 2
    cell_ymax = cy + cell_size / 2

    # Component bounding boxes
    half_w = component_sizes[:, 0] / 2  # (N,)
    half_h = component_sizes[:, 1] / 2
    comp_xmin = positions[:, 0] - half_w
    comp_xmax = positions[:, 0] + half_w
    comp_ymin = positions[:, 1] - half_h
    comp_ymax = positions[:, 1] + half_h

    congestion_per_graph = torch.zeros(n_graphs, device=device)

    for g_idx in range(n_graphs):
        mask = (node_graph_idx == g_idx)
        if not mask.any():
            continue

        g_comp_xmin = comp_xmin[mask]  # (Vg,)
        g_comp_xmax = comp_xmax[mask]
        g_comp_ymin = comp_ymin[mask]
        g_comp_ymax = comp_ymax[mask]
        g_sizes = component_sizes[mask]

        # Total area for target density
        total_area = (g_sizes[:, 0] * g_sizes[:, 1]).sum()
        canvas_area = (canvas_max - canvas_min) ** 2
        target_density = total_area / canvas_area  # uniform target per unit area

        # Density grid: (G, G)
        density = torch.zeros(G, G, device=device)

        # For each component, compute overlap with each cell
        # Vectorized: (Vg, G) overlap along x and y
        Vg = g_comp_xmin.shape[0]

        # x overlap: (Vg, G)
        ov_x = torch.clamp(
            torch.minimum(g_comp_xmax.unsqueeze(1), cell_xmax.unsqueeze(0)) -
            torch.maximum(g_comp_xmin.unsqueeze(1), cell_xmin.unsqueeze(0)),
            min=0.0)
        # y overlap: (Vg, G)
        ov_y = torch.clamp(
            torch.minimum(g_comp_ymax.unsqueeze(1), cell_ymax.unsqueeze(0)) -
            torch.maximum(g_comp_ymin.unsqueeze(1), cell_ymin.unsqueeze(0)),
            min=0.0)

        # Area overlap: (Vg, G, G) via outer product of x and y overlaps
        # ov_x[:, j] * ov_y[:, i] for cell (i, j)
        ov_area = ov_x.unsqueeze(1) * ov_y.unsqueeze(2)  # (Vg, G_y, G_x)
        density = ov_area.sum(dim=0)  # (G, G) total area in each cell

        # Normalize to density (area fraction)
        density = density / cell_area

        # Congestion = sum of squared excess over target
        excess = torch.clamp(density - target_density, min=0.0)
        congestion_per_graph[g_idx] = (excess ** 2).sum()

    return congestion_per_graph


def compute_cost(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    edge_attr: torch.Tensor = None,
    congestion_weight: float = 0.1,
    congestion_grid: int = 8,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined cost: HPWL + lambda * congestion.

    Args:
        positions: (N, 2)
        component_sizes: (N, 2)
        edge_index: (2, E)
        node_graph_idx: (N,)
        n_graphs: int
        edge_attr: (E, 4) or None
        congestion_weight: lambda for congestion term
        congestion_grid: grid resolution for density estimation

    Returns:
        cost: (n_graphs,) combined cost
        hpwl: (n_graphs,) HPWL component
        congestion: (n_graphs,) congestion component
    """
    _, hpwl, overlap, boundary = compute_chip_placement_energy(
        positions, component_sizes, edge_index, node_graph_idx, n_graphs,
        overlap_weight=0.0, boundary_weight=0.0,
        edge_attr=edge_attr,
    )

    congestion = compute_congestion(
        positions, component_sizes, node_graph_idx, n_graphs,
        grid_size=congestion_grid, canvas_min=canvas_min, canvas_max=canvas_max,
    )

    cost = hpwl + congestion_weight * congestion
    return cost, hpwl, congestion


def local_improvement(
    positions: torch.Tensor,
    component_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    edge_attr: torch.Tensor = None,
    n_iters: int = 20,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> torch.Tensor:
    """
    HPWL-driven swap/shift local search, preserving legality.

    For n_iters iterations:
      1. Random pair swap: try swapping positions of two components
         in same graph. Accept if HPWL improves AND no new overlap.
      2. Random single shift: try moving one component by one grid cell.
         Accept if HPWL improves AND stays legal (no overlap, in bounds).

    Args:
        positions: (N, 2) initial legal placement
        component_sizes: (N, 2)
        edge_index: (2, E)
        node_graph_idx: (N,)
        n_graphs: int
        edge_attr: (E, 4) or None
        n_iters: number of local search iterations
        canvas_min, canvas_max: canvas bounds

    Returns:
        improved_positions: (N, 2) improved placement (detached)
    """
    pos = positions.clone().detach()
    N = pos.shape[0]
    device = pos.device

    if N < 2:
        return pos

    # Current HPWL
    _, current_hpwl, _, _ = compute_chip_placement_energy(
        pos, component_sizes, edge_index, node_graph_idx, n_graphs,
        overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr,
    )

    for iteration in range(n_iters):
        # Alternate between swap and shift
        if iteration % 2 == 0:
            # --- Pair swap ---
            pos = _try_swap(
                pos, component_sizes, edge_index, edge_attr,
                node_graph_idx, n_graphs, current_hpwl,
                canvas_min, canvas_max, device)
        else:
            # --- Single shift ---
            pos = _try_shift(
                pos, component_sizes, edge_index, edge_attr,
                node_graph_idx, n_graphs, current_hpwl,
                canvas_min, canvas_max, device)

        # Recompute HPWL after moves
        _, current_hpwl, _, _ = compute_chip_placement_energy(
            pos, component_sizes, edge_index, node_graph_idx, n_graphs,
            overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr,
        )

    return pos


def _try_swap(
    pos, component_sizes, edge_index, edge_attr,
    node_graph_idx, n_graphs, current_hpwl,
    canvas_min, canvas_max, device,
):
    """Try random pair swaps, accept if HPWL improves and stays legal."""
    N = pos.shape[0]
    n_tries = min(N, 10)  # Try a few swaps per iteration

    for _ in range(n_tries):
        # Pick two random nodes in same graph
        i = torch.randint(0, N, (1,), device=device).item()
        g = node_graph_idx[i].item()
        same_graph = torch.where(node_graph_idx == g)[0]
        if same_graph.shape[0] < 2:
            continue
        local_j = torch.randint(0, same_graph.shape[0], (1,), device=device).item()
        j = same_graph[local_j].item()
        if i == j:
            continue

        # Try swap
        pos_new = pos.clone()
        pos_new[i] = pos[j]
        pos_new[j] = pos[i]

        # Check: do swapped positions still fit in bounds?
        if not _check_boundary(pos_new[i], component_sizes[i], canvas_min, canvas_max):
            continue
        if not _check_boundary(pos_new[j], component_sizes[j], canvas_min, canvas_max):
            continue

        # Check: no new overlaps?
        if _has_overlap_with_others(pos_new, component_sizes, node_graph_idx, i, g):
            continue
        if _has_overlap_with_others(pos_new, component_sizes, node_graph_idx, j, g):
            continue

        # Check HPWL improvement
        _, new_hpwl, _, _ = compute_chip_placement_energy(
            pos_new, component_sizes, edge_index, node_graph_idx, n_graphs,
            overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr,
        )
        if new_hpwl[g] < current_hpwl[g]:
            pos = pos_new
            current_hpwl = new_hpwl

    return pos


def _try_shift(
    pos, component_sizes, edge_index, edge_attr,
    node_graph_idx, n_graphs, current_hpwl,
    canvas_min, canvas_max, device,
):
    """Try random single-component shifts, accept if HPWL improves."""
    N = pos.shape[0]
    n_tries = min(N, 10)
    # Shift amount: small fraction of canvas
    shift_amount = (canvas_max - canvas_min) / 32.0

    directions = torch.tensor([
        [shift_amount, 0.0],
        [-shift_amount, 0.0],
        [0.0, shift_amount],
        [0.0, -shift_amount],
    ], device=device)

    for _ in range(n_tries):
        i = torch.randint(0, N, (1,), device=device).item()
        g = node_graph_idx[i].item()
        d = torch.randint(0, 4, (1,), device=device).item()

        pos_new = pos.clone()
        pos_new[i] = pos[i] + directions[d]

        # Check boundary
        if not _check_boundary(pos_new[i], component_sizes[i], canvas_min, canvas_max):
            continue

        # Check overlap
        if _has_overlap_with_others(pos_new, component_sizes, node_graph_idx, i, g):
            continue

        # Check HPWL
        _, new_hpwl, _, _ = compute_chip_placement_energy(
            pos_new, component_sizes, edge_index, node_graph_idx, n_graphs,
            overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr,
        )
        if new_hpwl[g] < current_hpwl[g]:
            pos = pos_new
            current_hpwl = new_hpwl

    return pos


def _check_boundary(pos_i, size_i, canvas_min, canvas_max):
    """Check if component i is within canvas bounds."""
    half_w = size_i[0] / 2
    half_h = size_i[1] / 2
    return (pos_i[0] - half_w >= canvas_min and pos_i[0] + half_w <= canvas_max and
            pos_i[1] - half_h >= canvas_min and pos_i[1] + half_h <= canvas_max)


def _has_overlap_with_others(pos, sizes, node_graph_idx, idx, graph_idx):
    """Check if component idx overlaps with any other component in same graph."""
    same_graph = (node_graph_idx == graph_idx)
    N = pos.shape[0]

    cx_i, cy_i = pos[idx, 0], pos[idx, 1]
    hw_i, hh_i = sizes[idx, 0] / 2, sizes[idx, 1] / 2

    for j in range(N):
        if j == idx or not same_graph[j]:
            continue
        cx_j, cy_j = pos[j, 0], pos[j, 1]
        hw_j, hh_j = sizes[j, 0] / 2, sizes[j, 1] / 2

        if (abs(cx_i - cx_j) < hw_i + hw_j and
                abs(cy_i - cy_j) < hh_i + hh_j):
            return True
    return False
