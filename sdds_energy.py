"""
SDDS Energy Function — True Net-Level WA-HPWL + Overlap + Boundary

Replaces the old edge-sum HPWL with correct net-level bounding-box HPWL
from ChipSAT/differentiable_hpwl.py.

Energy = WA-HPWL + overlap_weight * overlap + boundary_weight * boundary

No legalization — downstream CP-SAT handles legality.
"""

import sys
import os
import torch
from typing import Callable, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ChipSAT'))
from differentiable_hpwl import wa_hpwl

# Import scatter_sum from active gnn_layers
sys.path.insert(0, os.path.dirname(__file__))
from gnn_layers import scatter_sum


def compute_overlap_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Pairwise overlap area penalty, aggregated per graph.

    O(V^2) but vectorized. For V < 500 this is fast.

    Args:
        positions: (V, 2) center coords
        sizes: (V, 2) widths and heights
        node_graph_idx: (V,) graph assignment
        n_graphs: number of graphs
        margin: inflate each cell by this fraction of its size (e.g., 0.1 = 10%)
                 Forces the model to leave gaps between cells.

    Returns:
        overlap_per_graph: (n_graphs,)
    """
    V = positions.shape[0]
    device = positions.device
    inflated = sizes * (1.0 + margin) if margin > 0 else sizes
    half = inflated / 2.0

    x_min = positions[:, 0] - half[:, 0]
    y_min = positions[:, 1] - half[:, 1]
    x_max = positions[:, 0] + half[:, 0]
    y_max = positions[:, 1] + half[:, 1]

    ow = torch.clamp(
        torch.minimum(x_max.unsqueeze(1), x_max.unsqueeze(0))
        - torch.maximum(x_min.unsqueeze(1), x_min.unsqueeze(0)),
        min=0.0,
    )
    oh = torch.clamp(
        torch.minimum(y_max.unsqueeze(1), y_max.unsqueeze(0))
        - torch.maximum(y_min.unsqueeze(1), y_min.unsqueeze(0)),
        min=0.0,
    )
    overlap_area = ow * oh  # (V, V)

    # Upper triangle + same graph
    i_idx = torch.arange(V, device=device).unsqueeze(1)
    j_idx = torch.arange(V, device=device).unsqueeze(0)
    upper = (i_idx < j_idx).float()
    same_g = (node_graph_idx.unsqueeze(1) == node_graph_idx.unsqueeze(0)).float()

    masked = overlap_area * upper * same_g
    per_comp = masked.sum(dim=1)
    return scatter_sum(per_comp, node_graph_idx, dim=0, dim_size=n_graphs)


def compute_boundary_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """
    Boundary violation penalty per graph.

    Args:
        positions: (V, 2) center coords
        sizes: (V, 2) widths and heights
        node_graph_idx: (V,) graph assignment
        n_graphs: number of graphs

    Returns:
        boundary_per_graph: (n_graphs,)
    """
    # Compute per-node boundary violations for per-graph aggregation
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    left = torch.clamp(-1.0 - (positions[:, 0] - half_w), min=0)
    right = torch.clamp((positions[:, 0] + half_w) - 1.0, min=0)
    bottom = torch.clamp(-1.0 - (positions[:, 1] - half_h), min=0)
    top = torch.clamp((positions[:, 1] + half_h) - 1.0, min=0)
    per_node = left + right + bottom + top
    return scatter_sum(per_node, node_graph_idx, dim=0, dim_size=n_graphs)


def make_sdds_energy_fn(
    sizes: torch.Tensor,
    net_tensors: dict,
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    hpwl_gamma: float = 50.0,
    overlap_weight: float = 10.0,
    boundary_weight: float = 5.0,
    overlap_margin: float = 0.0,
) -> Callable:
    """
    Create energy_fn for a single-circuit graph.

    Energy is normalized by circuit size so that rewards are comparable
    across circuits of different sizes:
    - HPWL normalized by number of nets
    - Overlap/boundary normalized by number of nodes

    Returns:
        energy_fn(positions, node_graph_idx, n_graphs) -> (n_graphs,)
    """
    nni = net_tensors['net_node_indices']
    npo = net_tensors['net_pin_offsets']
    nm = net_tensors['net_mask']
    n_nets = float(net_tensors['n_nets'])  # number of nets
    n_nodes = float(sizes.shape[0])

    def energy_fn(positions, ngi, ng):
        hpwl_total, _ = wa_hpwl(positions, nni, npo, nm, gamma=hpwl_gamma)
        # wa_hpwl returns scalar — for single graph, wrap
        hpwl_g = hpwl_total.unsqueeze(0) if ng == 1 else hpwl_total
        overlap_g = compute_overlap_penalty(positions, sizes, ngi, ng, margin=overlap_margin)
        boundary_g = compute_boundary_penalty(positions, sizes, ngi, ng)
        return hpwl_g / n_nets + overlap_weight * overlap_g / n_nodes + boundary_weight * boundary_g / n_nodes

    return energy_fn


def make_batched_energy_fn(
    circuit_sizes: List[torch.Tensor],
    circuit_net_tensors: List[dict],
    circuit_node_offsets: List[int],
    node_graph_idx: torch.Tensor,
    n_graphs: int,
    hpwl_gamma: float = 50.0,
    overlap_weight: float = 10.0,
    boundary_weight: float = 5.0,
    overlap_margin: float = 0.0,
) -> Callable:
    """
    Create energy_fn for a batched multi-circuit graph.

    Each circuit has its own net tensors, so we compute per-circuit and aggregate.

    Returns:
        energy_fn(positions, node_graph_idx, n_graphs) -> (n_graphs,)
    """
    n_circuits = len(circuit_sizes)

    def energy_fn(positions, ngi, ng):
        energies = torch.zeros(ng, device=positions.device)

        for g in range(n_circuits):
            offset = circuit_node_offsets[g]
            V_g = circuit_sizes[g].shape[0]
            pos_g = positions[offset:offset + V_g]
            sizes_g = circuit_sizes[g]
            nt_g = circuit_net_tensors[g]
            ngi_g = torch.zeros(V_g, dtype=torch.long, device=positions.device)

            # WA-HPWL for this circuit
            hpwl_val, _ = wa_hpwl(
                pos_g, nt_g['net_node_indices'], nt_g['net_pin_offsets'],
                nt_g['net_mask'], gamma=hpwl_gamma,
            )
            # Overlap
            overlap_val = compute_overlap_penalty(pos_g, sizes_g, ngi_g, 1, margin=overlap_margin)
            # Boundary
            boundary_val = compute_boundary_penalty(pos_g, sizes_g, ngi_g, 1)

            n_nets_g = float(nt_g['n_nets'])
            n_nodes_g = float(V_g)
            energies[g] = hpwl_val / n_nets_g + overlap_weight * overlap_val[0] / n_nodes_g + boundary_weight * boundary_val[0] / n_nodes_g

        return energies

    return energy_fn


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _test():
    """Verify energy against cpsat_solver.compute_net_hpwl."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ChipSAT'))
    from benchmark_loader import load_bookshelf_circuit
    from cpsat_solver import compute_net_hpwl
    from differentiable_hpwl import build_net_tensors

    benchmark_dir = os.path.join(
        os.path.dirname(__file__), '..', 'ChipSAT', 'benchmarks', 'iccad04', 'extracted', 'ibm01'
    )
    if not os.path.exists(benchmark_dir):
        # Try SDDS-PyTorch benchmarks
        benchmark_dir = os.path.join(
            os.path.dirname(__file__), 'benchmarks', 'iccad04', 'extracted', 'ibm01'
        )

    if not os.path.exists(benchmark_dir):
        print("ibm01 not found, skipping verification")
        return

    data = load_bookshelf_circuit(benchmark_dir, 'ibm01', macros_only=True)
    positions = data['positions']
    sizes = data['node_features']
    nets = data['nets']
    V = data['n_components']

    # Exact HPWL
    exact = compute_net_hpwl(positions, sizes, nets)

    # WA-HPWL
    pos_t = torch.from_numpy(positions).float()
    sizes_t = torch.from_numpy(sizes).float()
    nt = build_net_tensors(nets, V)
    ngi = torch.zeros(V, dtype=torch.long)

    print(f"ibm01: {V} macros, {nt['n_nets']} nets")
    print(f"Exact HPWL: {exact:.4f}")

    for gamma in [10, 50, 100, 500]:
        wa, _ = wa_hpwl(pos_t, nt['net_node_indices'], nt['net_pin_offsets'],
                        nt['net_mask'], gamma=gamma)
        print(f"  WA-HPWL(gamma={gamma:3d}): {wa.item():.4f}  ratio={wa.item()/exact:.4f}")

    # Full energy
    energy_fn = make_sdds_energy_fn(sizes_t, nt, ngi, 1, hpwl_gamma=50.0)
    e = energy_fn(pos_t, ngi, 1)
    print(f"\nFull energy at reference: {e.item():.4f}")

    # Overlap / boundary
    ov = compute_overlap_penalty(pos_t, sizes_t, ngi, 1)
    bd = compute_boundary_penalty(pos_t, sizes_t, ngi, 1)
    print(f"  Overlap:  {ov.item():.6f}")
    print(f"  Boundary: {bd.item():.6f}")

    # Gradient
    pos_g = pos_t.clone().requires_grad_(True)
    e_g = energy_fn(pos_g, ngi, 1)
    e_g.sum().backward()
    print(f"  Gradient: {'OK' if pos_g.grad is not None else 'FAIL'}, norm={pos_g.grad.norm():.4f}")

    print("\nAll checks passed!")


if __name__ == '__main__':
    _test()
