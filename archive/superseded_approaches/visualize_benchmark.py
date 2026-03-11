"""
Visualize benchmark data to verify correctness and compatibility with the training pipeline.

Produces plots:
  1. Component placement with bounding boxes and edges (reference placement)
  2. Size distribution histogram
  3. Comparison: benchmark data vs synthetic data
  4. Edge connectivity statistics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection

from benchmark_loader import load_iccad04_circuit, load_bookshelf_circuit
from chip_placement_data import generate_chip_instance
from chip_placement_energy import compute_chip_placement_energy

import torch


def plot_placement(ax, positions, sizes, edge_index=None, edge_attr=None,
                   title="", canvas=(-1, 1, -1, 1), highlight_pins=False):
    """
    Plot component placement with bounding boxes and optional edges.

    Args:
        positions: (V, 2) center positions
        sizes: (V, 2) component (width, height)
        edge_index: (2, E) optional edges
        edge_attr: (E, 4) optional pin offsets
        title: plot title
        canvas: (x_min, x_max, y_min, y_max)
    """
    V = positions.shape[0]

    # Draw canvas boundary
    cx_min, cx_max, cy_min, cy_max = canvas
    canvas_rect = patches.Rectangle(
        (cx_min, cy_min), cx_max - cx_min, cy_max - cy_min,
        linewidth=2, edgecolor='black', facecolor='#f0f0f0', zorder=0
    )
    ax.add_patch(canvas_rect)

    # Draw components as rectangles
    rects = []
    colors = []
    for i in range(V):
        cx, cy = positions[i]
        w, h = sizes[i]
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=0.5, edgecolor='#333333'
        )
        rects.append(rect)
        # Color by relative size
        area = w * h
        colors.append(area)

    pc = PatchCollection(rects, alpha=0.7, edgecolor='#333333', linewidth=0.5)
    pc.set_array(np.array(colors))
    pc.set_cmap('YlOrRd')
    ax.add_collection(pc)

    # Draw edges
    if edge_index is not None and edge_index.shape[1] > 0:
        # Only draw unique edges (skip reverse duplicates)
        seen = set()
        lines = []
        pin_src_pts = []
        pin_dst_pts = []
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                continue
            seen.add(pair)

            if edge_attr is not None and edge_attr.shape[0] > e:
                # Pin-to-pin
                src_pin = positions[src] + edge_attr[e, :2]
                dst_pin = positions[dst] + edge_attr[e, 2:4]
            else:
                src_pin = positions[src]
                dst_pin = positions[dst]

            lines.append([src_pin, dst_pin])
            pin_src_pts.append(src_pin)
            pin_dst_pts.append(dst_pin)

        if lines:
            lc = LineCollection(lines, colors='#4488cc', linewidths=0.3, alpha=0.4, zorder=1)
            ax.add_collection(lc)

            if highlight_pins and pin_src_pts:
                src_arr = np.array(pin_src_pts)
                dst_arr = np.array(pin_dst_pts)
                ax.scatter(src_arr[:, 0], src_arr[:, 1], s=1, c='blue', alpha=0.3, zorder=3)
                ax.scatter(dst_arr[:, 0], dst_arr[:, 1], s=1, c='red', alpha=0.3, zorder=3)

    ax.set_xlim(cx_min - 0.1, cx_max + 0.1)
    ax.set_ylim(cy_min - 0.1, cy_max + 0.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)


def main():
    benchmark_base = os.path.join(os.path.dirname(__file__), 'benchmarks')
    test_path = os.path.join(benchmark_base, 'iccad04', 'extracted', 'ibm01', 'ibm01.nodes')
    if not os.path.exists(test_path):
        print("ICCAD04 benchmarks not found in benchmarks/. Run download first.")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), 'viz_output')
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: Full ibm01 placement ──────────────────────────────────────
    print("Loading ibm01 (full)...")
    data_full = load_iccad04_circuit(benchmark_base, "ibm01")
    pos_full = data_full['positions']
    sizes_full = data_full['node_features']
    ei_full = data_full['edge_index']
    ea_full = data_full['edge_attr']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"ibm01 — Full Circuit ({data_full['n_components']} components, "
                 f"{ei_full.shape[1]} edges)", fontsize=13)

    plot_placement(axes[0], pos_full, sizes_full, ei_full, ea_full,
                   title="Reference Placement (with edges)")
    plot_placement(axes[1], pos_full, sizes_full, title="Reference Placement (no edges)")

    # Size distribution
    areas = sizes_full[:, 0] * sizes_full[:, 1]
    axes[2].hist(areas, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[2].set_xlabel('Component Area (normalized)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Size Distribution')
    axes[2].set_yscale('log')

    plt.tight_layout()
    path1 = os.path.join(out_dir, 'ibm01_full.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig)

    # ── Figure 2: Subsampled ibm01 (BFS) ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ibm01 — BFS Subsampled (different sizes)", fontsize=13)

    for i, max_n in enumerate([50, 100, 200]):
        data_sub = load_iccad04_circuit(benchmark_base, "ibm01", max_nodes=max_n, seed=42)
        plot_placement(
            axes[i], data_sub['positions'], data_sub['node_features'],
            data_sub['edge_index'], data_sub['edge_attr'],
            title=f"max_nodes={max_n} ({data_sub['n_components']}n, {data_sub['edge_index'].shape[1]}e)",
            highlight_pins=True,
        )

    plt.tight_layout()
    path2 = os.path.join(out_dir, 'ibm01_subsampled.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close(fig)

    # ── Figure 3: Benchmark vs Synthetic comparison ─────────────────────────
    print("Generating synthetic data for comparison...")
    data_sub = load_iccad04_circuit(benchmark_base, "ibm01", max_nodes=50, seed=42)
    synth = generate_chip_instance("Chip_20_components", seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Benchmark vs Synthetic Data Comparison", fontsize=13)

    plot_placement(
        axes[0], data_sub['positions'], data_sub['node_features'],
        data_sub['edge_index'], data_sub['edge_attr'],
        title=f"ICCAD04 ibm01 (subsampled {data_sub['n_components']}n)",
    )
    plot_placement(
        axes[1], synth['legal_positions'], synth['sizes'],
        synth['edge_index'], synth['edge_attr'],
        title=f"Synthetic Chip_20 ({synth['n_components']}n)",
    )

    plt.tight_layout()
    path3 = os.path.join(out_dir, 'benchmark_vs_synthetic.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")
    plt.close(fig)

    # ── Figure 4: Data compatibility check ──────────────────────────────────
    print("\nData compatibility check:")
    data100 = load_iccad04_circuit(benchmark_base, "ibm01", max_nodes=100, seed=42)

    pos_t = torch.tensor(data100['positions'], dtype=torch.float32)
    sizes_t = torch.tensor(data100['node_features'], dtype=torch.float32)
    ei_t = torch.tensor(data100['edge_index'], dtype=torch.long)
    ea_t = torch.tensor(data100['edge_attr'], dtype=torch.float32)
    ngi_t = torch.zeros(data100['n_components'], dtype=torch.long)

    energy, hpwl, overlap, boundary = compute_chip_placement_energy(
        pos_t, sizes_t, ei_t, ngi_t, n_graphs=1, edge_attr=ea_t,
    )

    print(f"  V={data100['n_components']}, E={ei_t.shape[1]}")
    print(f"  positions range: [{pos_t.min():.3f}, {pos_t.max():.3f}]")
    print(f"  sizes range:     [{sizes_t.min():.6f}, {sizes_t.max():.6f}]")
    print(f"  edge_attr range: [{ea_t.min():.6f}, {ea_t.max():.6f}]")
    print(f"  Energy:    {energy.item():.4f}")
    print(f"  HPWL:      {hpwl.item():.4f}")
    print(f"  Overlap:   {overlap.item():.6f}")
    print(f"  Boundary:  {boundary.item():.6f}")

    # Check for NaN/Inf
    has_nan = (torch.isnan(pos_t).any() or torch.isnan(sizes_t).any() or
               torch.isnan(ei_t.float()).any() or torch.isnan(ea_t).any())
    has_inf = (torch.isinf(pos_t).any() or torch.isinf(sizes_t).any() or
               torch.isinf(ea_t).any())
    print(f"  NaN: {'FAIL' if has_nan else 'OK'}")
    print(f"  Inf: {'FAIL' if has_inf else 'OK'}")

    # Edge index bounds check
    max_idx = ei_t.max().item()
    print(f"  edge_index max={max_idx}, V={data100['n_components']}: "
          f"{'OK' if max_idx < data100['n_components'] else 'FAIL'}")

    # Gradient flow check
    pos_grad = pos_t.clone().requires_grad_(True)
    e_grad, _, _, _ = compute_chip_placement_energy(
        pos_grad, sizes_t, ei_t, ngi_t, n_graphs=1, edge_attr=ea_t,
    )
    e_grad.sum().backward()
    grad_ok = pos_grad.grad is not None and not torch.isnan(pos_grad.grad).any()
    print(f"  Gradient flow: {'OK' if grad_ok else 'FAIL'}")

    # ── Figure 5: Multiple IBM circuits ─────────────────────────────────────
    print("\nLoading multiple IBM circuits...")
    circuits = ["ibm01", "ibm02", "ibm03", "ibm04"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("ICCAD04 Circuits (subsampled to 100 nodes)", fontsize=13)

    for i, cname in enumerate(circuits):
        try:
            d = load_iccad04_circuit(benchmark_base, cname, max_nodes=100, seed=42)
            plot_placement(
                axes[i], d['positions'], d['node_features'],
                d['edge_index'], d['edge_attr'],
                title=f"{cname} ({d['n_components']}n, {d['edge_index'].shape[1]}e)",
            )
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center',
                        transform=axes[i].transAxes, fontsize=8)
            axes[i].set_title(cname)

    plt.tight_layout()
    path5 = os.path.join(out_dir, 'multi_circuits.png')
    fig.savefig(path5, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path5}")
    plt.close(fig)

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
