"""
Chip Placement Dataset Generator - DiffUCO Faithful Port

3-step generation:
1. Place components legally (greedy, no overlap)
2. Generate proximity-based netlist (k-NN)
3. Randomize positions (training input)

Reference: DIffUCO/DatasetCreator/loadGraphDatasets/ChipDatasetGenerator_Unsupervised.py
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

try:
    from .step_model import scatter_sum
except ImportError:
    from step_model import scatter_sum


# Scale presets matching DIffUCO chip_placement_config.py
SCALE_PRESETS = {
    "Chip_5_components": {
        "max_pool": 10, "long_size_range": (0.5, 0.8),
        "aspect_range": (0.4, 1.0), "target_density": (0.5, 0.7),
    },
    "Chip_10_components": {
        "max_pool": 25, "long_size_range": (0.4, 0.75),
        "aspect_range": (0.4, 1.0), "target_density": (0.6, 0.8),
    },
    "Chip_20_components": {
        "max_pool": 60, "long_size_range": (0.2, 0.5),
        "aspect_range": (0.4, 1.0), "target_density": (0.65, 0.85),
    },
    "Chip_50_components": {
        "max_pool": 120, "long_size_range": (0.1, 0.35),
        "aspect_range": (0.4, 1.0), "target_density": (0.7, 0.9),
    },
}


def _place_components_legal(
    x_sizes: np.ndarray,
    y_sizes: np.ndarray,
    stop_density: float,
    canvas: float = 1.0,
    max_attempts: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Place components greedily without overlap until target density reached.

    Args:
        x_sizes: (pool_size,) width of each candidate component
        y_sizes: (pool_size,) height
        stop_density: target placement density
        canvas: half-size of canvas (canvas is [-canvas, canvas]^2)
        max_attempts: max random position attempts per component

    Returns:
        positions: (V, 2) placed component centers
        sizes: (V, 2) placed component (width, height)
        density: actual density achieved
    """
    canvas_area = (2 * canvas) ** 2
    total_area = 0.0
    placed_positions = []
    placed_sizes = []

    # Sort by area (largest first for better packing)
    areas = x_sizes * y_sizes
    order = np.argsort(-areas)

    rng = np.random.default_rng()

    for idx in order:
        if total_area / canvas_area >= stop_density:
            break

        w, h = float(x_sizes[idx]), float(y_sizes[idx])

        # Valid position range (center must keep component inside canvas)
        lo_x, hi_x = -canvas + w / 2, canvas - w / 2
        lo_y, hi_y = -canvas + h / 2, canvas - h / 2

        if lo_x >= hi_x or lo_y >= hi_y:
            continue  # Component too large

        placed = False
        for _ in range(max_attempts):
            cx = rng.uniform(lo_x, hi_x)
            cy = rng.uniform(lo_y, hi_y)

            # Check overlap with all placed components
            overlap = False
            for (px, py), (pw, ph) in zip(placed_positions, placed_sizes):
                if (abs(cx - px) < (w + pw) / 2 and
                        abs(cy - py) < (h + ph) / 2):
                    overlap = True
                    break

            if not overlap:
                placed_positions.append((cx, cy))
                placed_sizes.append((w, h))
                total_area += w * h
                placed = True
                break

        if not placed:
            continue  # Skip this component

    if len(placed_positions) < 2:
        # Fallback: place at least 2 components
        placed_positions = [(0.0, 0.0), (0.5, 0.0)]
        placed_sizes = [(0.2, 0.2), (0.2, 0.2)]
        total_area = 0.08

    positions = np.array(placed_positions, dtype=np.float32)
    sizes = np.array(placed_sizes, dtype=np.float32)
    density = total_area / canvas_area

    return positions, sizes, density


def _generate_netlist_proximity(
    positions: np.ndarray,
    sizes: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate netlist based on spatial proximity (k-NN).

    Args:
        positions: (V, 2) component centers
        sizes: (V, 2) component (width, height)
        k: number of nearest neighbors to connect

    Returns:
        edge_index: (2, E) int array
        edge_attr: (E, 4) float array [src_dx, src_dy, dst_dx, dst_dy] terminal offsets
    """
    V = positions.shape[0]
    k = min(k, V - 1)

    rng = np.random.default_rng()

    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (V, V, 2)
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (V, V)
    np.fill_diagonal(dists, np.inf)

    edges_src, edges_dst = [], []
    edge_attrs = []

    for i in range(V):
        nn_idx = np.argsort(dists[i])[:k]
        for j in nn_idx:
            edges_src.append(i)
            edges_dst.append(j)

            # Random terminal offsets within component bounds
            half_w_i, half_h_i = sizes[i, 0] / 2, sizes[i, 1] / 2
            half_w_j, half_h_j = sizes[j, 0] / 2, sizes[j, 1] / 2

            src_dx = rng.uniform(-half_w_i, half_w_i)
            src_dy = rng.uniform(-half_h_i, half_h_i)
            dst_dx = rng.uniform(-half_w_j, half_w_j)
            dst_dy = rng.uniform(-half_h_j, half_h_j)

            edge_attrs.append([src_dx, src_dy, dst_dx, dst_dy])

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)

    return edge_index, edge_attr


def generate_chip_instance(
    dataset_name: str = "Chip_20_components",
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate a single chip placement instance.

    Returns dict with:
        'positions': (V, 2) randomized positions (training input / initial X_T)
        'legal_positions': (V, 2) legal placement (ground truth)
        'sizes': (V, 2) component (width, height)
        'edge_index': (2, E) netlist
        'edge_attr': (E, 4) terminal offsets
        'density': float
        'n_components': int
    """
    if seed is not None:
        np.random.seed(seed)

    preset = SCALE_PRESETS.get(dataset_name, SCALE_PRESETS["Chip_20_components"])
    max_pool = preset["max_pool"]
    lo_size, hi_size = preset["long_size_range"]
    lo_asp, hi_asp = preset["aspect_range"]
    lo_dens, hi_dens = preset["target_density"]

    rng = np.random.default_rng(seed)

    # Sample component sizes
    long_sizes = rng.uniform(lo_size, hi_size, size=max_pool).astype(np.float32)
    aspect_ratios = rng.uniform(lo_asp, hi_asp, size=max_pool).astype(np.float32)
    short_sizes = long_sizes * aspect_ratios

    # Random orientation
    long_x = rng.random(max_pool) > 0.5
    x_sizes = np.where(long_x, long_sizes, short_sizes)
    y_sizes = np.where(long_x, short_sizes, long_sizes)

    # Target density
    stop_density = rng.uniform(lo_dens, hi_dens)

    # Step 1: Place legally
    legal_positions, sizes, density = _place_components_legal(
        x_sizes, y_sizes, stop_density
    )

    # Step 2: Generate proximity netlist
    edge_index, edge_attr = _generate_netlist_proximity(legal_positions, sizes)

    # Step 3: Randomize positions (uniform in [-1, 1])
    V = sizes.shape[0]
    positions = rng.uniform(-1.0, 1.0, size=(V, 2)).astype(np.float32)

    return {
        "positions": positions,
        "legal_positions": legal_positions,
        "sizes": sizes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "density": density,
        "n_components": V,
    }


def generate_chip_batch(
    n_graphs: int,
    dataset_name: str = "Chip_20_components",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of chip instances with combined graph structure.

    Args:
        n_graphs: number of instances
        dataset_name: scale preset name
        seed: random seed
        device: torch device

    Returns:
        node_features: (total_nodes, 2) component sizes
        edge_index: (2, total_edges) with global node indices
        edge_attr: (total_edges, 4) terminal offsets
        node_graph_idx: (total_nodes,) graph assignment
        initial_positions: (total_nodes, 2) randomized starting positions
        legal_positions: (total_nodes, 2) ground truth positions
    """
    if device is None:
        device = torch.device("cpu")

    all_sizes = []
    all_edges_src = []
    all_edges_dst = []
    all_edge_attr = []
    all_positions = []
    all_legal = []
    all_graph_idx = []

    node_offset = 0

    for g in range(n_graphs):
        s = seed + g if seed is not None else None
        instance = generate_chip_instance(dataset_name, seed=s)

        V = instance["n_components"]
        all_sizes.append(instance["sizes"])
        all_positions.append(instance["positions"])
        all_legal.append(instance["legal_positions"])
        all_graph_idx.append(np.full(V, g, dtype=np.int64))

        # Offset edge indices for batching
        ei = instance["edge_index"]
        all_edges_src.append(ei[0] + node_offset)
        all_edges_dst.append(ei[1] + node_offset)
        all_edge_attr.append(instance["edge_attr"])

        node_offset += V

    node_features = torch.tensor(np.concatenate(all_sizes, axis=0), dtype=torch.float32, device=device)
    edge_index = torch.tensor(
        np.array([np.concatenate(all_edges_src), np.concatenate(all_edges_dst)]),
        dtype=torch.long, device=device
    )
    edge_attr = torch.tensor(np.concatenate(all_edge_attr, axis=0), dtype=torch.float32, device=device)
    node_graph_idx = torch.tensor(np.concatenate(all_graph_idx), dtype=torch.long, device=device)
    initial_positions = torch.tensor(np.concatenate(all_positions, axis=0), dtype=torch.float32, device=device)
    legal_positions = torch.tensor(np.concatenate(all_legal, axis=0), dtype=torch.float32, device=device)

    return node_features, edge_index, edge_attr, node_graph_idx, initial_positions, legal_positions


def test_chip_placement_data():
    """Test dataset generation."""
    print("=" * 60)
    print("Testing Chip Placement Dataset Generator")
    print("=" * 60)

    for name in ["Chip_5_components", "Chip_10_components", "Chip_20_components"]:
        print(f"\n--- {name} ---")
        instance = generate_chip_instance(name, seed=42)
        V = instance["n_components"]
        E = instance["edge_index"].shape[1]
        print(f"  Components: {V}")
        print(f"  Edges: {E}")
        print(f"  Density: {instance['density']:.3f}")
        print(f"  Size range: [{instance['sizes'].min():.3f}, {instance['sizes'].max():.3f}]")
        print(f"  Position range: [{instance['positions'].min():.3f}, {instance['positions'].max():.3f}]")

    # Test batching
    print("\n--- Batch generation (4 graphs) ---")
    node_feat, edge_idx, edge_att, ng_idx, pos, legal = generate_chip_batch(
        4, "Chip_10_components", seed=42
    )
    print(f"  node_features: {node_feat.shape}")
    print(f"  edge_index: {edge_idx.shape}")
    print(f"  edge_attr: {edge_att.shape}")
    print(f"  node_graph_idx: {ng_idx.shape}, unique graphs: {ng_idx.unique().tolist()}")
    print(f"  positions: {pos.shape}")

    # Verify no cross-graph edges
    src_graphs = ng_idx[edge_idx[0]]
    dst_graphs = ng_idx[edge_idx[1]]
    cross_graph = (src_graphs != dst_graphs).sum().item()
    print(f"  Cross-graph edges: {cross_graph} {'(OK)' if cross_graph == 0 else '(FAIL)'}")

    print("\n" + "=" * 60)
    print("Dataset generator tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_chip_placement_data()
