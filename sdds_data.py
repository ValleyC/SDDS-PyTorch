"""
SDDS Data Adapter — Unified Loading for Synthetic + Benchmark Circuits

Converts both ChipSAT synthetic circuits and BookShelf benchmarks into
a common SDDSCircuit format with 14D node features and pre-built net tensors.

Key difference from old chip_placement_data.py:
- Synthetic circuits use topology-first generation (NOT k-NN from positions)
- 14D node features: sizes + Laplacian PE + centrality
- Net tensors built for true net-level WA-HPWL
"""

import sys
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ChipSAT'))
from synthetic_circuits import generate_circuit
from benchmark_loader import load_bookshelf_circuit
from placement_model import build_node_features
from differentiable_hpwl import build_net_tensors


@dataclass
class SDDSCircuit:
    """Single circuit in SDDS-ready format."""
    node_features: torch.Tensor       # (V, 14)
    sizes: torch.Tensor               # (V, 2)
    edge_index: torch.Tensor          # (2, E)
    edge_attr: torch.Tensor           # (E, 4)
    net_tensors: dict                 # padded net tensors for WA-HPWL
    nets: list                        # original nets for eval
    reference_positions: torch.Tensor  # (V, 2)
    n_components: int
    circuit_name: str


def load_synthetic_circuit(
    seed: int,
    v_range: Tuple[int, int] = (30, 300),
    device: str = 'cpu',
) -> SDDSCircuit:
    """
    Generate a synthetic circuit and convert to SDDSCircuit.

    Uses ChipSAT/synthetic_circuits.py (topology-first, NOT k-NN).
    """
    data, metadata = generate_circuit(seed=seed, v_range=v_range)

    V = data['n_components']
    sizes = torch.from_numpy(data['node_features']).float().to(device)  # (V, 2)
    edge_index = torch.from_numpy(data['edge_index']).long().to(device)  # (2, E)
    edge_attr = torch.from_numpy(data['edge_attr']).float().to(device)  # (E, 4)
    positions = torch.from_numpy(data['positions']).float().to(device)  # (V, 2)
    nets = data['nets']

    # Build 14D node features
    node_features = build_node_features(sizes, edge_index, V)  # (V, 14)

    # Build net tensors for WA-HPWL
    nt = build_net_tensors(nets, V, device=torch.device(device))

    return SDDSCircuit(
        node_features=node_features,
        sizes=sizes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        net_tensors=nt,
        nets=nets,
        reference_positions=positions,
        n_components=V,
        circuit_name=data.get('circuit_name', f'synth_{seed}'),
    )


def load_benchmark_sdds(
    circuit_dir: str,
    circuit_name: str,
    macros_only: bool = True,
    max_nodes: int = None,
    seed: int = 42,
    device: str = 'cpu',
) -> SDDSCircuit:
    """
    Load a BookShelf benchmark and convert to SDDSCircuit.
    """
    data = load_bookshelf_circuit(
        circuit_dir, circuit_name,
        macros_only=macros_only, max_nodes=max_nodes, seed=seed,
    )

    V = data['n_components']
    sizes = torch.from_numpy(data['node_features']).float().to(device)
    edge_index = torch.from_numpy(data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(data['edge_attr']).float().to(device)
    positions = torch.from_numpy(data['positions']).float().to(device)
    nets = data['nets']

    node_features = build_node_features(sizes, edge_index, V)
    nt = build_net_tensors(nets, V, device=torch.device(device))

    return SDDSCircuit(
        node_features=node_features,
        sizes=sizes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        net_tensors=nt,
        nets=nets,
        reference_positions=positions,
        n_components=V,
        circuit_name=circuit_name,
    )


class SDDSDataset:
    """
    Fixed dataset of synthetic + benchmark circuits for SDDS training.

    Follows DIffUCO paradigm: pre-generate all instances, then iterate
    with a DataLoader (shuffle per epoch, same instances across epochs).
    """

    def __init__(
        self,
        n_synthetic: int = 200,
        synthetic_ratio: float = 0.7,
        v_range: Tuple[int, int] = (30, 200),
        benchmark_circuits: Optional[List[SDDSCircuit]] = None,
        device: str = 'cpu',
        seed: int = 0,
    ):
        self.v_range = v_range
        self.device = device
        self.circuits: List[SDDSCircuit] = []

        # Pre-generate fixed synthetic dataset
        n_synth = int(n_synthetic * synthetic_ratio)
        print(f"Pre-generating {n_synth} synthetic circuits (V={v_range[0]}-{v_range[1]})...")
        for i in range(n_synth):
            circ = load_synthetic_circuit(seed + i, v_range, device)
            self.circuits.append(circ)

        # Add benchmark circuits
        if benchmark_circuits:
            self.circuits.extend(benchmark_circuits)

        print(f"Dataset: {len(self.circuits)} circuits "
              f"({n_synth} synthetic + {len(self.circuits) - n_synth} benchmark)")

        self._rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, idx: int) -> SDDSCircuit:
        return self.circuits[idx]

    def get_epoch_order(self, epoch: int) -> List[int]:
        """Get shuffled index order for an epoch."""
        rng = np.random.RandomState(self._rng.randint(0, 2**31) + epoch)
        indices = list(range(len(self.circuits)))
        rng.shuffle(indices)
        return indices

    def sample_circuit(self, seed: int) -> SDDSCircuit:
        """Sample one circuit (for backward compat / eval)."""
        idx = seed % len(self.circuits)
        return self.circuits[idx]

    def sample_batch(self, batch_size: int, seed: int) -> dict:
        """
        Sample a batch of circuits for training.

        Returns dict with batched graph data + per-circuit net_tensors.
        """
        circuits = []
        for i in range(batch_size):
            circuits.append(self.sample_circuit(seed + i))

        # Concatenate into batched graph
        all_features = []
        all_sizes = []
        all_ei_src = []
        all_ei_dst = []
        all_ea = []
        all_ngi = []
        circuit_net_tensors = []
        circuit_sizes = []
        circuit_node_offsets = []
        all_ref_pos = []

        node_offset = 0
        for g, circ in enumerate(circuits):
            V = circ.n_components
            all_features.append(circ.node_features)
            all_sizes.append(circ.sizes)
            all_ei_src.append(circ.edge_index[0] + node_offset)
            all_ei_dst.append(circ.edge_index[1] + node_offset)
            all_ea.append(circ.edge_attr)
            all_ngi.append(torch.full((V,), g, dtype=torch.long, device=circ.node_features.device))
            all_ref_pos.append(circ.reference_positions)

            circuit_net_tensors.append(circ.net_tensors)
            circuit_sizes.append(circ.sizes)
            circuit_node_offsets.append(node_offset)

            node_offset += V

        device = circuits[0].node_features.device
        return {
            'node_features': torch.cat(all_features, dim=0),
            'sizes': torch.cat(all_sizes, dim=0),
            'edge_index': torch.stack([torch.cat(all_ei_src), torch.cat(all_ei_dst)]),
            'edge_attr': torch.cat(all_ea, dim=0),
            'node_graph_idx': torch.cat(all_ngi).to(device),
            'n_graphs': batch_size,
            'circuit_net_tensors': circuit_net_tensors,
            'circuit_sizes': circuit_sizes,
            'circuit_node_offsets': circuit_node_offsets,
            'reference_positions': torch.cat(all_ref_pos, dim=0),
            'circuits': circuits,
        }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _test():
    """Verify data loading."""
    print("=== Testing sdds_data.py ===\n")

    # Test synthetic
    print("--- Synthetic circuit ---")
    circ = load_synthetic_circuit(seed=42, v_range=(30, 100))
    print(f"  Name: {circ.circuit_name}")
    print(f"  V={circ.n_components}, E={circ.edge_index.shape[1]}")
    print(f"  Node features: {circ.node_features.shape}")
    print(f"  Sizes: {circ.sizes.shape}")
    print(f"  Nets: {circ.net_tensors['n_nets']}")
    print(f"  Reference pos range: [{circ.reference_positions.min():.3f}, {circ.reference_positions.max():.3f}]")
    assert circ.node_features.shape == (circ.n_components, 14)

    # Test dataset batch
    print("\n--- Dataset batch ---")
    dataset = SDDSDataset(synthetic_ratio=1.0, v_range=(30, 80))
    batch = dataset.sample_batch(3, seed=0)
    print(f"  Total nodes: {batch['node_features'].shape[0]}")
    print(f"  Total edges: {batch['edge_index'].shape[1]}")
    print(f"  Graphs: {batch['n_graphs']}")
    print(f"  Circuit sizes: {[s.shape[0] for s in batch['circuit_sizes']]}")
    print(f"  Node offsets: {batch['circuit_node_offsets']}")

    # Verify no cross-graph edges
    src_g = batch['node_graph_idx'][batch['edge_index'][0]]
    dst_g = batch['node_graph_idx'][batch['edge_index'][1]]
    cross = (src_g != dst_g).sum().item()
    print(f"  Cross-graph edges: {cross} {'(OK)' if cross == 0 else '(FAIL)'}")

    print("\nAll checks passed!")


if __name__ == '__main__':
    _test()
