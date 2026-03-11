"""
Large Neighborhood Search (LNS) Solver for Chip Placement

Orchestrates iterative CP-SAT subproblem solving:
  1. Select a neighborhood (subset of macros to re-place)
  2. Solve subset with CP-SAT (NoOverlap2D + minimize net-level HPWL)
  3. Accept/reject based on full cost recomputation
  4. Adaptive window/subset sizing

Acceptance policy:
  - Phase 1: improvement-only (greedy descent)
  - Phase 2: simulated annealing (activated after plateau)
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from collections import deque

from cpsat_solver import (
    legalize, solve_subset, compute_net_hpwl, check_overlap, check_boundary,
)


def compute_density_np(
    positions: np.ndarray,
    sizes: np.ndarray,
    grid_size: int = 8,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> float:
    """Cell-area density (deprecated, use compute_rudy_np for congestion)."""
    G = grid_size
    cell_size = (canvas_max - canvas_min) / G
    cell_area = cell_size * cell_size
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    canvas_area = (canvas_max - canvas_min) ** 2
    target_density = total_macro_area / canvas_area
    edges = np.linspace(canvas_min, canvas_max, G + 1)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    comp_xmin = positions[:, 0] - half_w
    comp_xmax = positions[:, 0] + half_w
    comp_ymin = positions[:, 1] - half_h
    comp_ymax = positions[:, 1] + half_h
    density = np.zeros((G, G))
    for gx in range(G):
        for gy in range(G):
            cx_lo, cx_hi = edges[gx], edges[gx + 1]
            cy_lo, cy_hi = edges[gy], edges[gy + 1]
            ox = np.maximum(0, np.minimum(comp_xmax, cx_hi) - np.maximum(comp_xmin, cx_lo))
            oy = np.maximum(0, np.minimum(comp_ymax, cy_hi) - np.maximum(comp_ymin, cy_lo))
            density[gx, gy] = (ox * oy).sum() / cell_area
    excess = np.maximum(0, density - target_density)
    return float((excess ** 2).sum())


def compute_rudy_np(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    grid_size: int = 32,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> Dict:
    """
    Net-based RUDY (Rectangular Uniform wire DensitY) congestion.

    For each net, distributes wire density weight = (w_eff + h_eff) / (w_eff * h_eff)
    uniformly over the net bounding box, accumulated on a grid.
    Overflow = max(0, rudy_map - capacity) where capacity = mean(rudy_map).

    Returns dict with: cost, rudy_map, rudy_max, rudy_p95, rudy_p99, overflow_sum
    """
    G = grid_size
    tile_w = (canvas_max - canvas_min) / G
    tile_h = tile_w
    edges = np.linspace(canvas_min, canvas_max, G + 1)

    # Fixed minimum bbox extent — independent of grid_size for invariance.
    # A net's wire must span at least this fraction of the canvas.
    min_extent = (canvas_max - canvas_min) / 16.0

    # Precompute net bboxes and weights — vectorized over pins per net
    net_bboxes = []  # (min_x, max_x, min_y, max_y)
    net_weights = []

    for net in nets:
        if len(net) < 2:
            continue
        pin_xs = np.array([positions[n, 0] + dx for n, dx, dy in net])
        pin_ys = np.array([positions[n, 1] + dy for n, dx, dy in net])
        x_lo, x_hi = float(pin_xs.min()), float(pin_xs.max())
        y_lo, y_hi = float(pin_ys.min()), float(pin_ys.max())
        bbox_w = x_hi - x_lo
        bbox_h = y_hi - y_lo
        # Degenerate clamping: expand bbox to fixed min_extent (grid-size invariant)
        w_eff = max(bbox_w, min_extent)
        h_eff = max(bbox_h, min_extent)
        # Expand bbox symmetrically if degenerate
        if bbox_w < min_extent:
            cx = (x_lo + x_hi) / 2
            x_lo = cx - w_eff / 2
            x_hi = cx + w_eff / 2
        if bbox_h < min_extent:
            cy = (y_lo + y_hi) / 2
            y_lo = cy - h_eff / 2
            y_hi = cy + h_eff / 2
        weight = (w_eff + h_eff) / (w_eff * h_eff)
        net_bboxes.append((x_lo, x_hi, y_lo, y_hi))
        net_weights.append(weight)

    rudy_map = np.zeros((G, G), dtype=np.float64)

    if len(net_bboxes) == 0:
        return {
            'cost': 0.0,
            'rudy_map': rudy_map,
            'rudy_max': 0.0,
            'rudy_p95': 0.0,
            'rudy_p99': 0.0,
            'overflow_sum': 0.0,
        }

    # Vectorized: (M, 4) bboxes, (M,) weights
    bboxes = np.array(net_bboxes, dtype=np.float64)  # (M, 4)
    weights = np.array(net_weights, dtype=np.float64)  # (M,)

    # Accumulate RUDY on grid — vectorized over nets per tile
    # Normalize by tile area so values are density (wire demand per unit area),
    # making the map grid-size invariant.
    tile_area = tile_w * tile_h
    for gx in range(G):
        for gy in range(G):
            tx_lo, tx_hi = edges[gx], edges[gx + 1]
            ty_lo, ty_hi = edges[gy], edges[gy + 1]
            # Overlap of each net bbox with this tile
            ox = np.maximum(0, np.minimum(bboxes[:, 1], tx_hi) - np.maximum(bboxes[:, 0], tx_lo))
            oy = np.maximum(0, np.minimum(bboxes[:, 3], ty_hi) - np.maximum(bboxes[:, 2], ty_lo))
            overlap_area = ox * oy  # (M,)
            rudy_map[gx, gy] = (weights * overlap_area).sum() / tile_area

    # Overflow-style cost.
    # capacity = mean(rudy_map): a heuristic proxy for uniform routing capacity.
    # This is NOT calibrated to actual routing resources; it represents the
    # average wire density that would result from perfectly uniform distribution.
    # Suitable as an optimization proxy; real evaluation requires global routing.
    capacity = rudy_map.mean()
    overflow = np.maximum(0, rudy_map - capacity)

    # Multiply overflow by tile_area to get a spatial integral (grid-size invariant).
    # Without this, sum(overflow) scales with number of tiles.
    overflow_integral = float((overflow * tile_area).sum())

    flat = rudy_map.flatten()
    return {
        'cost': overflow_integral,
        'rudy_map': rudy_map,
        'rudy_max': float(flat.max()),
        'rudy_p95': float(np.percentile(flat, 95)),
        'rudy_p99': float(np.percentile(flat, 99)),
        'overflow_sum': overflow_integral,
    }


def compute_per_macro_rudy(
    positions: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[Tuple[int, float, float]]],
    macro_nets: List[List[int]],
    grid_size: int = 32,
    canvas_min: float = -1.0,
    canvas_max: float = 1.0,
) -> np.ndarray:
    """
    Per-macro RUDY score: mean RUDY density over tiles touched by the macro's nets.

    Averaged over incident net count to avoid bias toward high-degree macros.
    This measures "how congested is this macro's routing neighborhood" rather
    than "how much total wire passes through this macro."

    Returns:
        (N,) array of per-macro RUDY scores
    """
    N = positions.shape[0]
    G = grid_size
    tile_w = (canvas_max - canvas_min) / G

    # First compute full RUDY map
    rudy_info = compute_rudy_np(positions, sizes, nets, grid_size, canvas_min, canvas_max)
    rudy_map = rudy_info['rudy_map']

    # For each macro, average RUDY over tiles touched by its nets
    scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        valid_nets = 0
        macro_rudy = 0.0
        for net_idx in macro_nets[i]:
            net = nets[net_idx]
            if len(net) < 2:
                continue
            pin_xs = [positions[n, 0] + dx for n, dx, dy in net]
            pin_ys = [positions[n, 1] + dy for n, dx, dy in net]
            x_lo, x_hi = min(pin_xs), max(pin_xs)
            y_lo, y_hi = min(pin_ys), max(pin_ys)
            # Find overlapping tiles
            gx_lo = max(0, int((x_lo - canvas_min) / tile_w))
            gx_hi = min(G - 1, int((x_hi - canvas_min) / tile_w))
            gy_lo = max(0, int((y_lo - canvas_min) / tile_w))
            gy_hi = min(G - 1, int((y_hi - canvas_min) / tile_w))
            region = rudy_map[gx_lo:gx_hi + 1, gy_lo:gy_hi + 1]
            n_tiles = max(region.size, 1)
            macro_rudy += region.sum() / n_tiles  # mean density per net
            valid_nets += 1
        # Average over incident nets to reduce degree bias
        if valid_nets > 0:
            scores[i] = macro_rudy / valid_nets

    return scores


class LNSSolver:
    """Large Neighborhood Search for chip macro placement."""

    def __init__(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        nets: List[List[Tuple[int, float, float]]],
        edge_index: np.ndarray,
        congestion_weight: float = 0.1,
        subset_size: int = 30,
        window_fraction: float = 0.15,
        cpsat_time_limit: float = 5.0,
        plateau_threshold: int = 20,
        adapt_threshold: int = 30,
        min_subset: int = 10,
        min_window: float = 0.05,
        max_window: float = 0.4,
        sa_t_init: float = 0.5,
        sa_cooling: float = 0.995,
        seed: int = 42,
    ):
        self.N = positions.shape[0]
        self.sizes = sizes.copy()
        self.nets = nets
        self.edge_index = edge_index
        self.congestion_weight = congestion_weight
        self.rng = np.random.default_rng(seed)

        # Current and best solutions
        self.current_pos = positions.copy()
        self.best_pos = positions.copy()
        self.current_hpwl = compute_net_hpwl(positions, sizes, nets)
        self.best_hpwl = self.current_hpwl
        self.current_cost = self._compute_cost(positions, self.current_hpwl)
        self.best_cost = self.current_cost

        # Adaptive parameters
        self.subset_size = subset_size
        self.window_fraction = window_fraction
        self.cpsat_time_limit = cpsat_time_limit
        self.min_subset = min_subset
        self.min_window = min_window
        self.max_window = max_window
        self.plateau_threshold = plateau_threshold
        self.adapt_threshold = adapt_threshold

        # Simulated annealing
        self.sa_t_init = sa_t_init
        self.sa_cooling = sa_cooling
        self.sa_temperature = sa_t_init
        self.sa_active = False

        # Tracking
        self.stagnation_count = 0
        self.iteration = 0
        self.n_accepted = 0
        self.n_improved = 0
        self.n_infeasible = 0

        # Per-strategy tracking
        self.strategies = ['random', 'worst_hpwl', 'congestion', 'connected']
        self.strategy_attempts = {s: 0 for s in self.strategies}
        self.strategy_successes = {s: 0 for s in self.strategies}

        # Precompute per-macro HPWL contribution and net adjacency
        self._precompute_macro_data()

        # Build adjacency from edge_index for connected strategy
        self.adj = [[] for _ in range(self.N)]
        if edge_index is not None and edge_index.shape[1] > 0:
            for e in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                if src < self.N and dst < self.N:
                    self.adj[src].append(dst)

    def _compute_cost(self, positions: np.ndarray, hpwl: float) -> float:
        """Compute full cost: HPWL + congestion_weight * RUDY_overflow."""
        if self.congestion_weight > 0:
            rudy = compute_rudy_np(positions, self.sizes, self.nets)
            return hpwl + self.congestion_weight * rudy['cost']
        return hpwl

    def _precompute_macro_data(self):
        """Compute per-macro HPWL contribution for neighborhood selection."""
        self.macro_hpwl = np.zeros(self.N)
        self.macro_nets = [[] for _ in range(self.N)]  # which nets each macro belongs to

        for net_idx, net in enumerate(self.nets):
            if len(net) < 2:
                continue
            xs = [self.current_pos[n, 0] + dx for n, dx, dy in net]
            ys = [self.current_pos[n, 1] + dy for n, dx, dy in net]
            net_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))

            for (node_idx, _, _) in net:
                self.macro_hpwl[node_idx] += net_hpwl
                self.macro_nets[node_idx].append(net_idx)

    def _update_macro_hpwl(self):
        """Recompute per-macro HPWL from current positions."""
        self.macro_hpwl[:] = 0
        for net_idx, net in enumerate(self.nets):
            if len(net) < 2:
                continue
            xs = [self.current_pos[n, 0] + dx for n, dx, dy in net]
            ys = [self.current_pos[n, 1] + dy for n, dx, dy in net]
            net_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            for (node_idx, _, _) in net:
                self.macro_hpwl[node_idx] += net_hpwl

    def select_strategy(self) -> str:
        """Epsilon-greedy strategy selection based on success rates."""
        epsilon = 0.3
        if self.rng.random() < epsilon or self.iteration < 20:
            return self.rng.choice(self.strategies)

        # Weight by success rate
        weights = []
        for s in self.strategies:
            attempts = max(self.strategy_attempts[s], 1)
            weights.append(self.strategy_successes[s] / attempts + 0.01)
        weights = np.array(weights)
        weights /= weights.sum()
        return self.rng.choice(self.strategies, p=weights)

    def get_neighborhood(self, strategy: str, k: int) -> np.ndarray:
        """Select k macros according to the given strategy."""
        k = min(k, self.N)

        if strategy == 'random':
            return self.rng.choice(self.N, size=k, replace=False)

        elif strategy == 'worst_hpwl':
            # Top-k macros by per-macro HPWL contribution
            indices = np.argsort(-self.macro_hpwl)[:k]
            return indices

        elif strategy == 'congestion':
            # Macros in highest-RUDY regions (net-based wire density)
            macro_rudy = compute_per_macro_rudy(
                self.current_pos, self.sizes, self.nets, self.macro_nets,
            )
            # Add noise for diversity
            max_val = macro_rudy.max()
            if max_val > 0:
                macro_rudy += self.rng.uniform(0, max_val * 0.1, size=self.N)
            indices = np.argsort(-macro_rudy)[:k]
            return indices

        elif strategy == 'connected':
            # BFS from random seed along netlist edges
            seed = self.rng.integers(0, self.N)
            visited = set([seed])
            frontier = [seed]
            while len(visited) < k and frontier:
                self.rng.shuffle(frontier)
                next_frontier = []
                for node in frontier:
                    for neighbor in self.adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                            if len(visited) >= k:
                                break
                    if len(visited) >= k:
                        break
                frontier = next_frontier
            # If BFS didn't reach k, pad with random
            if len(visited) < k:
                remaining = list(set(range(self.N)) - visited)
                self.rng.shuffle(remaining)
                for r in remaining[:k - len(visited)]:
                    visited.add(r)
            return np.array(list(visited), dtype=int)

        else:
            return self.rng.choice(self.N, size=k, replace=False)

    def accept(self, delta_cost: float) -> bool:
        """Accept or reject a move based on cost change."""
        if delta_cost < -1e-8:
            return True

        if self.sa_active:
            if self.sa_temperature > 1e-10:
                prob = np.exp(-delta_cost / self.sa_temperature)
                return self.rng.random() < prob
            return False

        return False

    def step(self) -> Dict:
        """Run one LNS iteration."""
        self.iteration += 1
        t0 = time.time()

        # Select strategy and neighborhood
        strategy = self.select_strategy()
        subset = self.get_neighborhood(strategy, self.subset_size)
        self.strategy_attempts[strategy] += 1

        # Solve CP-SAT subproblem
        new_positions = solve_subset(
            self.current_pos, self.sizes, self.nets, subset,
            time_limit=self.cpsat_time_limit,
            window_fraction=self.window_fraction,
        )

        dt = time.time() - t0
        accepted = False
        improved = False
        delta_cost = 0.0

        if new_positions is None:
            # Infeasible or timeout — reject
            self.n_infeasible += 1
        else:
            # Full cost recomputation (HPWL + congestion)
            new_hpwl = compute_net_hpwl(new_positions, self.sizes, self.nets)
            new_cost = self._compute_cost(new_positions, new_hpwl)
            delta_cost = new_cost - self.current_cost

            if self.accept(delta_cost):
                accepted = True
                self.n_accepted += 1
                self.current_pos = new_positions
                self.current_hpwl = new_hpwl
                self.current_cost = new_cost
                self._update_macro_hpwl()  # keep macro_hpwl in sync with current_pos

                if new_cost < self.best_cost - 1e-8:
                    improved = True
                    self.n_improved += 1
                    self.best_pos = new_positions.copy()
                    self.best_hpwl = new_hpwl
                    self.best_cost = new_cost
                    self.stagnation_count = 0

                    # Shrink window/subset after improvement
                    self.window_fraction = max(
                        self.window_fraction * 0.8, self.min_window)
                    self.subset_size = max(
                        self.subset_size - 5, self.min_subset)

                    # Deactivate SA after improvement
                    if self.sa_active:
                        self.sa_active = False
                        self.sa_temperature = self.sa_t_init

                    self.strategy_successes[strategy] += 1

        if not improved:
            self.stagnation_count += 1

            # Activate SA after plateau
            if not self.sa_active and self.stagnation_count >= self.plateau_threshold:
                self.sa_active = True
                self.sa_temperature = self.sa_t_init

            # Grow window/subset after longer stagnation
            if self.stagnation_count >= self.adapt_threshold:
                self.window_fraction = min(
                    self.window_fraction * 1.5, self.max_window)
                self.subset_size = min(
                    self.subset_size + 10, self.N // 2)

        # Cool SA temperature
        if self.sa_active:
            self.sa_temperature *= self.sa_cooling

        return {
            'iteration': self.iteration,
            'strategy': strategy,
            'accepted': accepted,
            'improved': improved,
            'delta_cost': delta_cost,
            'current_hpwl': self.current_hpwl,
            'best_hpwl': self.best_hpwl,
            'current_cost': self.current_cost,
            'best_cost': self.best_cost,
            'subset_size': self.subset_size,
            'window_fraction': self.window_fraction,
            'sa_active': self.sa_active,
            'sa_temperature': self.sa_temperature if self.sa_active else 0.0,
            'stagnation': self.stagnation_count,
            'time': dt,
        }

    def solve(
        self,
        n_iterations: int = 500,
        log_every: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full LNS optimization.

        Returns:
            dict with best_positions, best_hpwl, best_cost, history
        """
        history = []

        if verbose:
            print(f"\nLNS: {self.N} macros, {len(self.nets)} nets")
            print(f"  Initial HPWL: {self.current_hpwl:.4f}")
            overlap, n_ov = check_overlap(self.current_pos, self.sizes)
            boundary = check_boundary(self.current_pos, self.sizes)
            print(f"  Initial overlap: {overlap:.6f} ({n_ov} pairs)")
            print(f"  Initial boundary: {boundary:.6f}")
            print(f"  subset_size={self.subset_size}, window={self.window_fraction:.2f}")
            print()

        for i in range(n_iterations):
            metrics = self.step()
            history.append(metrics)

            if verbose and (i % log_every == 0 or i == n_iterations - 1 or metrics['improved']):
                mode = "SA" if metrics['sa_active'] else "GD"
                status = "IMPROVED" if metrics['improved'] else (
                    "accepted" if metrics['accepted'] else "rejected")
                print(
                    f"  [{i+1:4d}/{n_iterations}] [{mode}] "
                    f"best={metrics['best_hpwl']:.4f} "
                    f"cur={metrics['current_hpwl']:.4f} "
                    f"delta={metrics['delta_cost']:+.4f} "
                    f"strat={metrics['strategy']:12s} "
                    f"{status:8s} "
                    f"w={metrics['window_fraction']:.2f} "
                    f"k={metrics['subset_size']:3d} "
                    f"stag={metrics['stagnation']:3d} "
                    f"({metrics['time']:.1f}s)"
                )

        # Compute RUDY stats on best placement
        rudy_stats = compute_rudy_np(self.best_pos, self.sizes, self.nets)

        # Final summary
        if verbose:
            print(f"\nLNS complete after {n_iterations} iterations:")
            print(f"  Best HPWL: {self.best_hpwl:.4f}")
            overlap, n_ov = check_overlap(self.best_pos, self.sizes)
            boundary = check_boundary(self.best_pos, self.sizes)
            print(f"  Overlap: {overlap:.6f} ({n_ov} pairs)")
            print(f"  Boundary: {boundary:.6f}")
            print(f"  RUDY: max={rudy_stats['rudy_max']:.4f} "
                  f"p95={rudy_stats['rudy_p95']:.4f} "
                  f"p99={rudy_stats['rudy_p99']:.4f} "
                  f"overflow={rudy_stats['overflow_sum']:.4f}")
            print(f"  Accepted: {self.n_accepted}/{n_iterations} "
                  f"({self.n_accepted/max(n_iterations,1)*100:.1f}%)")
            print(f"  Improved: {self.n_improved}")
            print(f"  Infeasible: {self.n_infeasible}")
            print(f"  Strategy success rates:")
            for s in self.strategies:
                attempts = self.strategy_attempts[s]
                successes = self.strategy_successes[s]
                rate = successes / max(attempts, 1) * 100
                print(f"    {s:15s}: {successes}/{attempts} ({rate:.1f}%)")

        return {
            'best_positions': self.best_pos,
            'best_hpwl': self.best_hpwl,
            'best_cost': self.best_cost,
            'rudy_stats': rudy_stats,
            'history': history,
        }
