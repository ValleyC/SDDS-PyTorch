"""
ML-Guided Large Neighborhood Search (LNS) Solver

Wraps the core LNS loop with a NeighborhoodGNN that learns:
  - WHICH macros to include in the CP-SAT subproblem (subset selection)

CP-SAT handles WHERE macros go (exact optimization within window).
ML handles WHAT to optimize next (learned destroy operator).

Supports:
  - Learned subset selection with confidence-gated heuristic fallback
  - Experience collection for offline warm-start + online REINFORCE
  - Exploration via mixed random/learned subsets during training
"""

import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any

from cpsat_solver import (
    solve_subset, compute_net_hpwl,
    check_overlap, check_boundary,
)
from lns_solver import compute_rudy_np, compute_per_macro_rudy, LNSSolver
from trust_region_model import NeighborhoodGNN, build_node_features
from experience_buffer import ExperienceBuffer


class MLGuidedLNSSolver:
    """
    LNS solver with learned neighborhood (subset) selection.

    When model is None or confidence is low, falls back to heuristic
    strategies (random, worst_hpwl, congestion, connected).
    """

    def __init__(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        nets: List[List[Tuple[int, float, float]]],
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        # ML model
        model: Optional[NeighborhoodGNN] = None,
        device: torch.device = torch.device('cpu'),
        use_learned_subset: bool = True,
        confidence_threshold: float = 0.05,
        explore_temperature: float = 1.0,
        explore: bool = False,
        # Experience collection
        experience_buffer: Optional[ExperienceBuffer] = None,
        circuit_id: str = 'unknown',
        # LNS params (same as LNSSolver)
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
        n_iterations: int = 500,
        seed: int = 42,
    ):
        self.N = positions.shape[0]
        self.sizes = sizes.copy()
        self.nets = nets
        self.edge_index_np = edge_index
        self.edge_attr_np = edge_attr
        self.circuit_id = circuit_id
        self.n_total_iterations = n_iterations

        # ML
        self.model = model
        self.device = device
        self.use_learned_subset = use_learned_subset and model is not None
        self.confidence_threshold = confidence_threshold
        self.explore_temperature = explore_temperature
        self.explore = explore
        self.experience_buffer = experience_buffer

        # LNS state (replicating LNSSolver)
        self.rng = np.random.default_rng(seed)
        self.congestion_weight = congestion_weight

        self.current_pos = positions.copy()
        self.best_pos = positions.copy()
        self.current_hpwl = compute_net_hpwl(positions, sizes, nets)
        self.best_hpwl = self.current_hpwl
        self.current_cost = self._compute_cost(positions, self.current_hpwl)
        self.best_cost = self.current_cost

        self.subset_size = subset_size
        self.window_fraction = window_fraction
        self.cpsat_time_limit = cpsat_time_limit
        self.min_subset = min_subset
        self.min_window = min_window
        self.max_window = max_window
        self.plateau_threshold = plateau_threshold
        self.adapt_threshold = adapt_threshold

        self.sa_t_init = sa_t_init
        self.sa_cooling = sa_cooling
        self.sa_temperature = sa_t_init
        self.sa_active = False

        self.stagnation_count = 0
        self.iteration = 0
        self.n_accepted = 0
        self.n_improved = 0
        self.n_infeasible = 0
        self.n_learned_subsets = 0
        self.n_fallback_subsets = 0

        self.strategies = ['random', 'worst_hpwl', 'congestion', 'connected']
        self.strategy_attempts = {s: 0 for s in self.strategies}
        self.strategy_successes = {s: 0 for s in self.strategies}

        # Per-macro HPWL and adjacency (same as LNSSolver)
        self.macro_hpwl = np.zeros(self.N)
        self.macro_nets = [[] for _ in range(self.N)]
        self._precompute_macro_data()

        self.adj = [[] for _ in range(self.N)]
        if edge_index is not None and edge_index.shape[1] > 0:
            for e in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                if src < self.N and dst < self.N:
                    self.adj[src].append(dst)

        # Track last iteration's displacement and subset for node features
        self._last_delta = np.zeros((self.N, 2), dtype=np.float32)
        self._last_subset_mask = np.zeros(self.N, dtype=np.float32)

        # Prepare edge tensors once (graph structure doesn't change)
        self._edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        self._edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    def _compute_cost(self, positions: np.ndarray, hpwl: float) -> float:
        if self.congestion_weight > 0:
            rudy = compute_rudy_np(positions, self.sizes, self.nets)
            return hpwl + self.congestion_weight * rudy['cost']
        return hpwl

    def _precompute_macro_data(self):
        self.macro_hpwl[:] = 0
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
        self.macro_hpwl[:] = 0
        for net in self.nets:
            if len(net) < 2:
                continue
            xs = [self.current_pos[n, 0] + dx for n, dx, dy in net]
            ys = [self.current_pos[n, 1] + dy for n, dx, dy in net]
            net_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            for (node_idx, _, _) in net:
                self.macro_hpwl[node_idx] += net_hpwl

    # --- Strategy selection and neighborhood ---

    def select_strategy(self) -> str:
        epsilon = 0.3
        if self.rng.random() < epsilon or self.iteration < 20:
            return self.rng.choice(self.strategies)
        weights = []
        for s in self.strategies:
            attempts = max(self.strategy_attempts[s], 1)
            weights.append(self.strategy_successes[s] / attempts + 0.01)
        weights = np.array(weights)
        weights /= weights.sum()
        return self.rng.choice(self.strategies, p=weights)

    def get_neighborhood(self, strategy: str, k: int) -> np.ndarray:
        k = min(k, self.N)
        if strategy == 'random':
            return self.rng.choice(self.N, size=k, replace=False)
        elif strategy == 'worst_hpwl':
            return np.argsort(-self.macro_hpwl)[:k]
        elif strategy == 'congestion':
            macro_rudy = compute_per_macro_rudy(
                self.current_pos, self.sizes, self.nets, self.macro_nets,
            )
            max_val = macro_rudy.max()
            if max_val > 0:
                macro_rudy += self.rng.uniform(0, max_val * 0.1, size=self.N)
            return np.argsort(-macro_rudy)[:k]
        elif strategy == 'connected':
            seed_node = self.rng.integers(0, self.N)
            visited = set([seed_node])
            frontier = [seed_node]
            while len(visited) < k and frontier:
                self.rng.shuffle(frontier)
                next_frontier = []
                for node in frontier:
                    for nb in self.adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.append(nb)
                            if len(visited) >= k:
                                break
                    if len(visited) >= k:
                        break
                frontier = next_frontier
            if len(visited) < k:
                remaining = list(set(range(self.N)) - visited)
                self.rng.shuffle(remaining)
                for r in remaining[:k - len(visited)]:
                    visited.add(r)
            return np.array(list(visited), dtype=int)
        else:
            return self.rng.choice(self.N, size=k, replace=False)

    def accept(self, delta_cost: float) -> bool:
        if delta_cost < -1e-8:
            return True
        if self.sa_active and self.sa_temperature > 1e-10:
            prob = np.exp(-delta_cost / self.sa_temperature)
            return self.rng.random() < prob
        return False

    # --- Node features (always computed, needed for experience collection) ---

    def _compute_node_features(self) -> np.ndarray:
        """Compute 14-dim node features for current placement state."""
        macro_congestion = np.zeros(self.N, dtype=np.float32)
        if self.congestion_weight > 0:
            raw_rudy = compute_per_macro_rudy(
                self.current_pos, self.sizes, self.nets, self.macro_nets,
            )
            # Robust normalization: divide by p95 + eps, clip to [0, 5]
            p95 = np.percentile(raw_rudy, 95) if self.N > 1 else (raw_rudy.max() + 1e-8)
            macro_congestion = np.clip(raw_rudy / (p95 + 1e-8), 0.0, 5.0).astype(np.float32)

        return build_node_features(
            positions=self.current_pos,
            sizes=self.sizes,
            macro_hpwl=self.macro_hpwl,
            macro_congestion=macro_congestion,
            last_delta=self._last_delta,
            in_last_subset=self._last_subset_mask,
            stagnation_frac=self.stagnation_count / max(self.adapt_threshold, 1),
            iteration_frac=self.iteration / max(self.n_total_iterations, 1),
            window_fraction=self.window_fraction,
            subset_size_frac=self.subset_size / max(self.N, 1),
            sa_temperature_norm=self.sa_temperature / max(self.sa_t_init, 1e-8) if self.sa_active else 0.0,
        )

    # --- Subset selection (learned or heuristic) ---

    def _select_subset_learned(self, node_features: np.ndarray, k: int) -> Tuple[np.ndarray, str, float]:
        """
        Use GNN to select subset. Falls back to heuristic if confidence is low.

        Returns:
            subset: (k,) macro indices
            strategy: 'learned' or heuristic name
            confidence: model confidence score
        """
        node_features_t = torch.tensor(node_features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                node_features_t,
                self._edge_index_t,
                self._edge_attr_t,
            )

        scores = outputs['subset_scores']
        conf = self.model.confidence(scores, k)

        # Confidence-gated fallback
        if conf < self.confidence_threshold:
            self.n_fallback_subsets += 1
            strategy = self.select_strategy()
            subset = self.get_neighborhood(strategy, k)
            self.strategy_attempts[strategy] += 1
            return subset, strategy, conf

        # Use learned selection
        self.n_learned_subsets += 1
        indices, _ = self.model.select_subset(
            scores, k,
            temperature=self.explore_temperature,
            explore=self.explore,
        )
        subset = indices.cpu().numpy()
        return subset, 'learned', conf

    # --- Post-solve state update (shared by step() and external REINFORCE loop) ---

    def update_state(
        self,
        new_positions: Optional[np.ndarray],
        subset: np.ndarray,
        strategy: str,
    ) -> Dict[str, Any]:
        """
        Update solver state after a CP-SAT solve. Handles acceptance,
        best tracking, SA activation/cooling, and window/subset adaptation.

        Returns dict with: accepted, improved, feasible, delta_cost, new_hpwl, new_cost
        """
        accepted = False
        improved = False
        delta_cost = 0.0
        feasible = new_positions is not None

        if not feasible:
            self.n_infeasible += 1
        else:
            new_hpwl = compute_net_hpwl(new_positions, self.sizes, self.nets)
            new_cost = self._compute_cost(new_positions, new_hpwl)
            delta_cost = new_cost - self.current_cost

            if self.accept(delta_cost):
                accepted = True
                self.n_accepted += 1

                # Track displacement for node features
                self._last_delta = new_positions - self.current_pos
                self._last_subset_mask = np.zeros(self.N, dtype=np.float32)
                for idx in subset:
                    self._last_subset_mask[int(idx)] = 1.0

                self.current_pos = new_positions
                self.current_hpwl = new_hpwl
                self.current_cost = new_cost
                self._update_macro_hpwl()

                if new_cost < self.best_cost - 1e-8:
                    improved = True
                    self.n_improved += 1
                    self.best_pos = new_positions.copy()
                    self.best_hpwl = new_hpwl
                    self.best_cost = new_cost
                    self.stagnation_count = 0
                    self.window_fraction = max(self.window_fraction * 0.8, self.min_window)
                    self.subset_size = max(self.subset_size - 5, self.min_subset)
                    if self.sa_active:
                        self.sa_active = False
                        self.sa_temperature = self.sa_t_init
                    if strategy in self.strategy_successes:
                        self.strategy_successes[strategy] += 1

        if not improved:
            self.stagnation_count += 1
            if not self.sa_active and self.stagnation_count >= self.plateau_threshold:
                self.sa_active = True
                self.sa_temperature = self.sa_t_init
            if self.stagnation_count >= self.adapt_threshold:
                self.window_fraction = min(self.window_fraction * 1.5, self.max_window)
                self.subset_size = min(self.subset_size + 10, self.N // 2)

        if self.sa_active:
            self.sa_temperature *= self.sa_cooling

        self.iteration += 1

        return {
            'accepted': accepted,
            'improved': improved,
            'feasible': feasible,
            'delta_cost': delta_cost,
            'new_hpwl': new_hpwl if feasible else self.current_hpwl,
            'new_cost': new_cost if feasible else self.current_cost,
        }

    # --- Main step ---

    def step(self) -> Dict[str, Any]:
        """Run one ML-guided LNS iteration."""
        t0 = time.time()

        # Compute node features (always, for experience collection)
        node_features = self._compute_node_features()

        # Select subset
        k = min(self.subset_size, self.N)
        if self.use_learned_subset:
            subset, strategy, confidence = self._select_subset_learned(node_features, k)
        else:
            strategy = self.select_strategy()
            subset = self.get_neighborhood(strategy, k)
            self.strategy_attempts[strategy] += 1
            confidence = 0.0

        # Solve CP-SAT (always use standard solve_subset, no hints)
        new_positions = solve_subset(
            self.current_pos, self.sizes, self.nets, subset,
            time_limit=self.cpsat_time_limit,
            window_fraction=self.window_fraction,
        )

        dt = time.time() - t0

        # Update state (acceptance, adaptation, SA)
        state = self.update_state(new_positions, subset, strategy)
        accepted = state['accepted']
        improved = state['improved']
        feasible = state['feasible']
        delta_cost = state['delta_cost']

        # Collect experience
        if self.experience_buffer is not None:
            # Store pre_positions correctly (before the solve changed them)
            pre_pos = self.current_pos if not accepted else (
                new_positions - self._last_delta if feasible else self.current_pos)
            if accepted and feasible:
                pre_pos = new_positions - self._last_delta

            experience = {
                'node_features': node_features,
                'edge_index': self.edge_index_np,
                'edge_attr': self.edge_attr_np,
                'subset_indices': subset,
                'pre_positions': pre_pos,
                'post_positions': new_positions,
                'delta_cost': -delta_cost,  # positive = improvement
                'solve_time': dt,
                'feasible': feasible,
                'accepted': accepted,
                'circuit_id': self.circuit_id,
                'strategy': strategy,
                'confidence': confidence,
            }
            self.experience_buffer.add(experience)

        metrics = {
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
            'feasible': feasible,
            'confidence': confidence,
        }
        return metrics

    def solve(
        self,
        n_iterations: int = 500,
        log_every: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run full ML-guided LNS optimization."""
        self.n_total_iterations = n_iterations
        history = []

        mode_label = "learned_subset" if self.use_learned_subset else "heuristic_LNS"
        if self.explore:
            mode_label += "+explore"

        if verbose:
            print(f"\nML-Guided LNS [{mode_label}]: {self.N} macros, {len(self.nets)} nets")
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
                status_str = "IMPROVED" if metrics['improved'] else (
                    "accepted" if metrics['accepted'] else (
                        "infeasible" if not metrics['feasible'] else "rejected"))
                conf_str = f"c={metrics['confidence']:.2f}" if self.use_learned_subset else ""
                print(
                    f"  [{i+1:4d}/{n_iterations}] [{mode}] "
                    f"best={metrics['best_hpwl']:.4f} "
                    f"cur={metrics['current_hpwl']:.4f} "
                    f"delta={metrics['delta_cost']:+.4f} "
                    f"strat={metrics['strategy']:12s} "
                    f"{status_str:10s} "
                    f"w={metrics['window_fraction']:.2f} "
                    f"k={metrics['subset_size']:3d} "
                    f"stag={metrics['stagnation']:3d} "
                    f"{conf_str} "
                    f"({metrics['time']:.1f}s)"
                )

        if verbose:
            print(f"\nML-Guided LNS [{mode_label}] complete after {n_iterations} iterations:")
            print(f"  Best HPWL: {self.best_hpwl:.4f}")
            overlap, n_ov = check_overlap(self.best_pos, self.sizes)
            boundary = check_boundary(self.best_pos, self.sizes)
            print(f"  Overlap: {overlap:.6f} ({n_ov} pairs)")
            print(f"  Boundary: {boundary:.6f}")
            print(f"  Accepted: {self.n_accepted}/{n_iterations} "
                  f"({self.n_accepted/max(n_iterations,1)*100:.1f}%)")
            print(f"  Improved: {self.n_improved}")
            print(f"  Infeasible: {self.n_infeasible}")
            if self.use_learned_subset:
                total = self.n_learned_subsets + self.n_fallback_subsets
                print(f"  Learned subsets: {self.n_learned_subsets}/{total} "
                      f"({self.n_learned_subsets/max(total,1)*100:.0f}%), "
                      f"fallbacks: {self.n_fallback_subsets}")

            if self.experience_buffer is not None:
                stats = self.experience_buffer.stats()
                print(f"  Experience buffer: {stats['size']} entries, "
                      f"feasible_rate={stats['feasible_rate']:.2f}")

        return {
            'best_positions': self.best_pos,
            'best_hpwl': self.best_hpwl,
            'best_cost': self.best_cost,
            'history': history,
        }
