"""
REINFORCE Trainer for Heatmap-based Chip Placement

Self-critical REINFORCE with:
  - Per-graph validity (not per-sample)
  - Self-critical baseline (greedy decode cost, matching policy repair_mode)
  - Congestion reward (HPWL + lambda * congestion)
  - Normalized log-probs (per-step average, not sum over all components)
  - Masked entropy bonus
  - Warm-start imitation pretraining
  - Per-graph reviser (only applies to valid graphs)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

try:
    from .heatmap_model import ChipHeatmapModel
    from .greedy_placer import GreedyPlacer, positions_to_cell_indices
    from .reviser import compute_cost, local_improvement
    from .chip_placement_energy import compute_chip_placement_energy
except ImportError:
    from heatmap_model import ChipHeatmapModel
    from greedy_placer import GreedyPlacer, positions_to_cell_indices
    from reviser import compute_cost, local_improvement
    from chip_placement_energy import compute_chip_placement_energy


class REINFORCETrainer:
    """Self-critical REINFORCE trainer for heatmap chip placement."""

    def __init__(
        self,
        model: ChipHeatmapModel,
        placer: GreedyPlacer,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
        entropy_weight: float = 0.01,
        congestion_weight: float = 0.1,
        congestion_grid: int = 8,
        reviser_iters: int = 20,
        invalid_penalty: float = 1000.0,
    ):
        self.model = model
        self.placer = placer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.entropy_weight = entropy_weight
        self.congestion_weight = congestion_weight
        self.congestion_grid = congestion_grid
        self.reviser_iters = reviser_iters
        self.invalid_penalty = invalid_penalty

    def _revise_per_graph(
        self,
        positions_2d: torch.Tensor,
        valid_per_graph: torch.Tensor,
        component_sizes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """Run reviser, selectively applying results to valid graphs only.

        Args:
            positions_2d: (N, 2) positions for one sample
            valid_per_graph: (n_graphs,) bool — which graphs are valid
            component_sizes, edge_index, edge_attr, node_graph_idx: graph data
            n_graphs: int

        Returns:
            revised_positions: (N, 2) — revised for valid graphs, original for invalid
        """
        revised = local_improvement(
            positions_2d.clone(), component_sizes, edge_index,
            node_graph_idx, n_graphs, edge_attr=edge_attr,
            n_iters=self.reviser_iters)

        # Only keep revisions for valid graphs
        result = positions_2d.clone()
        for g in range(n_graphs):
            if valid_per_graph[g]:
                g_mask = (node_graph_idx == g)
                result[g_mask] = revised[g_mask]
        return result

    def train_step(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        component_sizes: torch.Tensor,
        n_samples: int = 16,
        repair_mode: bool = True,
        use_reviser: bool = True,
    ) -> Dict:
        """
        One REINFORCE training step.

        1. GNN forward → logits
        2. Greedy decode (baseline, matching policy repair_mode) → greedy_cost
        3. Sample K placements → positions, log_probs, entropies, valid_flags
        4. Reviser: per-graph local_improvement on valid placements
        5. Compute cost per (graph, sample) — per-graph validity
        6. Self-critical advantage: cost - greedy_cost
        7. Normalized REINFORCE loss + masked entropy bonus
        """
        device = node_features.device
        self.model.train()

        # Count components per graph (for log-prob normalization)
        n_comp = torch.zeros(n_graphs, device=device)
        for g in range(n_graphs):
            n_comp[g] = (node_graph_idx == g).sum().float()

        # 1. Single GNN forward
        embeddings, logits = self.model(node_features, edge_index, edge_attr)

        # 2. Self-critical baseline: greedy decode (no grad)
        # Baseline always repairs — provides a reachable target cost even when
        # policy uses strict mode. If baseline also failed, advantage = 0 and
        # gradient would vanish, killing learning in strict phase.
        with torch.no_grad():
            greedy_pos, _, _, greedy_valid, greedy_repaired = \
                self.placer.place_components(
                    self.model, logits.detach(), component_sizes,
                    node_graph_idx, n_graphs, n_samples=1, greedy=True,
                    repair_mode=True)

            # Per-graph reviser for greedy baseline
            greedy_pos_2d = greedy_pos[:, 0, :]
            if use_reviser and self.reviser_iters > 0:
                greedy_pos_2d = self._revise_per_graph(
                    greedy_pos_2d, greedy_valid[:, 0],
                    component_sizes, edge_index, edge_attr,
                    node_graph_idx, n_graphs)

            greedy_cost, greedy_hpwl, greedy_cong = compute_cost(
                greedy_pos_2d, component_sizes, edge_index,
                node_graph_idx, n_graphs, edge_attr=edge_attr,
                congestion_weight=self.congestion_weight,
                congestion_grid=self.congestion_grid)

        # 3. Sample K placements
        positions, log_probs, entropies, valid_flags, repaired_flags = \
            self.placer.place_components(
                self.model, logits, component_sizes,
                node_graph_idx, n_graphs, n_samples=n_samples, greedy=False,
                repair_mode=repair_mode)

        # 4. Reviser: per-graph local improvement on each sample
        # FIX #2: Only apply reviser to valid graphs (was gating on ALL graphs)
        if use_reviser and self.reviser_iters > 0:
            with torch.no_grad():
                for s in range(n_samples):
                    positions[:, s, :] = self._revise_per_graph(
                        positions[:, s, :], valid_flags[:, s],
                        component_sizes, edge_index, edge_attr,
                        node_graph_idx, n_graphs)

        # 5. Compute cost per (graph, sample) — PER-GRAPH validity
        cost_matrix = torch.full(
            (n_graphs, n_samples), self.invalid_penalty, device=device)
        hpwl_matrix = torch.full(
            (n_graphs, n_samples), self.invalid_penalty, device=device)
        cong_matrix = torch.zeros(n_graphs, n_samples, device=device)

        for s in range(n_samples):
            pos_s = positions[:, s, :]

            _, hpwl_s, overlap_s, boundary_s = compute_chip_placement_energy(
                pos_s, component_sizes, edge_index,
                node_graph_idx, n_graphs,
                overlap_weight=0.0, boundary_weight=0.0,
                edge_attr=edge_attr)

            cost_s, _, cong_s = compute_cost(
                pos_s, component_sizes, edge_index,
                node_graph_idx, n_graphs, edge_attr=edge_attr,
                congestion_weight=self.congestion_weight,
                congestion_grid=self.congestion_grid)

            for g in range(n_graphs):
                if valid_flags[g, s]:
                    cost_matrix[g, s] = cost_s[g]
                    hpwl_matrix[g, s] = hpwl_s[g]
                    cong_matrix[g, s] = cong_s[g]

        # 6. Self-critical advantage per (graph, sample)
        advantage = cost_matrix - greedy_cost.unsqueeze(1)

        # FIX #4: Normalize log_probs and entropies by components per graph
        # Prevents scale explosion: raw sum over 246 steps → per-step average
        norm_factor = n_comp.unsqueeze(1).clamp(min=1)  # (n_graphs, 1)
        norm_log_probs = log_probs / norm_factor
        norm_entropies = entropies / norm_factor

        # 7. REINFORCE loss with normalized log-probs
        reinforce_loss = (advantage.detach() * norm_log_probs).mean()

        # Entropy bonus (normalized per step)
        entropy_bonus = norm_entropies.mean()
        total_loss = reinforce_loss - self.entropy_weight * entropy_bonus

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()

        # Metrics
        valid_mask_flat = valid_flags.flatten()
        valid_hpwl = hpwl_matrix.flatten()[valid_mask_flat]

        metrics = {
            'loss': total_loss.item(),
            'reinforce_loss': reinforce_loss.item(),
            'entropy': entropy_bonus.item(),
            'greedy_hpwl': greedy_hpwl.mean().item(),
            'greedy_cost': greedy_cost.mean().item(),
            'greedy_congestion': greedy_cong.mean().item(),
            'mean_hpwl': valid_hpwl.mean().item() if valid_hpwl.numel() > 0 else self.invalid_penalty,
            'best_hpwl': hpwl_matrix.min(dim=1).values.mean().item(),
            'mean_congestion': cong_matrix[valid_flags].mean().item() if valid_flags.any() else 0.0,
            'invalid_ratio': (~valid_flags).float().mean().item(),
            'repaired_ratio': repaired_flags.float().mean().item(),
            'advantage_std': advantage.std().item(),
            'advantage_mean': advantage.mean().item(),
            'mean_log_prob': norm_log_probs.mean().item(),
            'raw_log_prob': log_probs.mean().item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'occ_alpha': self.model.current_alpha,
        }

        return metrics

    def pretrain_imitation(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        reference_positions: torch.Tensor,
        grid_size: int,
        n_epochs: int = 50,
        log_every: int = 10,
    ) -> None:
        """
        Warm-start: cross-entropy pretraining on reference placement cell labels.

        Stabilizes heatmap head before RL exploration begins.
        """
        self.model.train()
        reference_cells = positions_to_cell_indices(
            reference_positions, grid_size,
            canvas_min=self.placer.canvas_min,
            canvas_max=self.placer.canvas_max,
        ).to(node_features.device)

        for epoch in range(n_epochs):
            _, logits = self.model(node_features, edge_index, edge_attr)
            loss = self.model.imitation_loss(logits, reference_cells)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == 0:
                with torch.no_grad():
                    pred_cells = logits.argmax(dim=-1)
                    accuracy = (pred_cells == reference_cells).float().mean().item()
                print(f"  Pretrain epoch {epoch+1}/{n_epochs}: "
                      f"loss={loss.item():.4f}, acc={accuracy:.3f}")

    @torch.no_grad()
    def evaluate(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: torch.Tensor,
        n_graphs: int,
        component_sizes: torch.Tensor,
        n_eval_samples: int = 64,
        overlap_threshold: float = 0.01,
        boundary_threshold: float = 0.01,
        use_reviser: bool = True,
    ) -> Dict:
        """
        Evaluate placement quality.

        FIX #5: Uses strict mode (repair_mode=False) for sampled placements
        to report true policy feasibility. Reports both:
          - strict_feasibility: fraction placed without termination
          - legal_ratio: among feasible, fraction passing overlap/boundary thresholds
        """
        self.model.eval()
        device = node_features.device

        # GNN forward
        _, logits = self.model(node_features, edge_index, edge_attr)

        # 1. Greedy placement (repair for best-effort greedy metric)
        greedy_pos, _, _, greedy_valid, _ = self.placer.place_components(
            self.model, logits, component_sizes,
            node_graph_idx, n_graphs, n_samples=1, greedy=True,
            repair_mode=True)
        greedy_pos_2d = greedy_pos[:, 0, :]

        if use_reviser and self.reviser_iters > 0:
            greedy_pos_2d = self._revise_per_graph(
                greedy_pos_2d, greedy_valid[:, 0],
                component_sizes, edge_index, edge_attr,
                node_graph_idx, n_graphs)

        _, greedy_hpwl, greedy_overlap, greedy_boundary = compute_chip_placement_energy(
            greedy_pos_2d, component_sizes, edge_index,
            node_graph_idx, n_graphs,
            overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr)

        # 2. Sampled placements — STRICT mode for true feasibility
        sampled_pos, _, _, sampled_valid, sampled_repaired = \
            self.placer.place_components(
                self.model, logits, component_sizes,
                node_graph_idx, n_graphs, n_samples=n_eval_samples, greedy=False,
                repair_mode=False)  # FIX #5: strict mode

        best_hpwl_per_graph = torch.full((n_graphs,), float('inf'), device=device)
        best_legal_hpwl_per_graph = torch.full((n_graphs,), float('inf'), device=device)
        n_strict_valid = 0  # (graph, sample) pairs placed without termination
        n_legal = 0          # of those, pairs passing overlap/boundary thresholds
        n_total = n_graphs * n_eval_samples

        for s in range(n_eval_samples):
            pos_s = sampled_pos[:, s, :].clone()

            # Per-graph reviser (only valid graphs)
            if use_reviser and self.reviser_iters > 0:
                pos_s = self._revise_per_graph(
                    pos_s, sampled_valid[:, s],
                    component_sizes, edge_index, edge_attr,
                    node_graph_idx, n_graphs)

            _, hpwl_s, overlap_s, boundary_s = compute_chip_placement_energy(
                pos_s, component_sizes, edge_index,
                node_graph_idx, n_graphs,
                overlap_weight=0.0, boundary_weight=0.0, edge_attr=edge_attr)

            for g in range(n_graphs):
                if not sampled_valid[g, s]:
                    continue  # Skip terminated (graph, sample) pairs

                n_strict_valid += 1

                if hpwl_s[g] < best_hpwl_per_graph[g]:
                    best_hpwl_per_graph[g] = hpwl_s[g]

                is_legal = (overlap_s[g] < overlap_threshold and
                            boundary_s[g] < boundary_threshold)
                if is_legal:
                    n_legal += 1
                    if hpwl_s[g] < best_legal_hpwl_per_graph[g]:
                        best_legal_hpwl_per_graph[g] = hpwl_s[g]

        strict_feasibility = n_strict_valid / max(n_total, 1)
        legal_ratio = n_legal / max(n_strict_valid, 1)

        return {
            'greedy_hpwl': greedy_hpwl.mean().item(),
            'greedy_overlap': greedy_overlap.mean().item(),
            'greedy_boundary': greedy_boundary.mean().item(),
            'best_sampled_hpwl': best_hpwl_per_graph.mean().item(),
            'best_legal_hpwl': best_legal_hpwl_per_graph.mean().item()
                if best_legal_hpwl_per_graph.isfinite().any() else float('inf'),
            'strict_feasibility': strict_feasibility,
            'legal_ratio': legal_ratio,
            'n_eval_samples': n_eval_samples,
        }
