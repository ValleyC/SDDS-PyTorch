"""
Neighborhood GNN for Learned Subset Selection in LNS

GNN backbone encodes placement graph -> per-macro subset scores.
At each LNS iteration, top-k macros by score form the destroy neighborhood.
CP-SAT solves the subproblem exactly. ML decides WHICH macros to optimize,
not WHERE they go.

Architecture:
    EncodeProcessDecode backbone (from step_model.py)
    -> Subset head: per-macro logit ("should this macro be in the destroy set?")
    -> Value head: per-graph scalar (REINFORCE baseline)

Training:
    Stage B (warm-start): Weighted BCE on subset membership from improving traces
    Stage C (online): REINFORCE with reward = max(0, delta_cost) / solve_time
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict

from step_model import EncodeProcessDecode, ReluMLP, ValueMLP, scatter_sum


class NeighborhoodGNN(nn.Module):
    """
    GNN that scores macros for subset selection in LNS.

    Given placement state (node features + graph), outputs per-macro scores
    indicating how beneficial it would be to include each macro in the next
    CP-SAT subproblem.
    """

    def __init__(
        self,
        input_dim: int = 14,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_message_passes: int = 5,
        mean_aggr: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared GNN backbone
        self.backbone = EncodeProcessDecode(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_message_passes=n_message_passes,
            mean_aggr=mean_aggr,
        )

        # Subset scoring head: hidden_dim -> 1 logit per macro
        self.subset_head = nn.Sequential(
            ReluMLP([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize near zero so initial scores are ~0.5 after sigmoid
        nn.init.xavier_normal_(self.subset_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.subset_head[-1].bias)

        # Value head (per-graph baseline for REINFORCE)
        self.value_mlp = ValueMLP([hidden_dim, 120, 64, 1])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: (V, input_dim)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
            node_graph_idx: (V,) graph assignment
            n_graphs: batch size

        Returns dict with:
            subset_logits: (V,) raw logits for subset selection
            subset_scores: (V,) sigmoid(logits), P(macro should be in subset)
            value: (n_graphs,) per-graph value estimate
            node_embeddings: (V, hidden_dim)
        """
        node_embeddings = self.backbone(node_features, edge_index, edge_attr)

        # Subset scores
        subset_logits = self.subset_head(node_embeddings).squeeze(-1)  # (V,)
        subset_scores = torch.sigmoid(subset_logits)  # (V,)

        # Value head: aggregate node embeddings per graph
        if node_graph_idx is None:
            node_graph_idx = torch.zeros(node_features.size(0), dtype=torch.long,
                                         device=node_features.device)

        graph_embed = scatter_sum(node_embeddings, node_graph_idx, dim=0,
                                  dim_size=n_graphs)
        nodes_per_graph = scatter_sum(
            torch.ones(node_features.size(0), 1, device=node_features.device),
            node_graph_idx, dim=0, dim_size=n_graphs
        )
        graph_embed = graph_embed / torch.sqrt(torch.clamp(nodes_per_graph, min=1.0))
        value = self.value_mlp(graph_embed).squeeze(-1)  # (n_graphs,)

        return {
            'subset_logits': subset_logits,
            'subset_scores': subset_scores,
            'value': value,
            'node_embeddings': node_embeddings,
        }

    def select_subset(
        self,
        subset_scores: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        explore: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k macros by score, with optional Gumbel exploration.

        Args:
            subset_scores: (V,) per-macro scores in [0,1]
            k: number of macros to select
            temperature: Gumbel noise scale for exploration
            explore: if True, add Gumbel noise

        Returns:
            indices: (k,) selected macro indices
            log_prob: scalar, log P(selected subset) under Bernoulli model
        """
        V = subset_scores.size(0)
        k = min(k, V)

        if explore and temperature > 0:
            # Gumbel-top-k: add noise to log-odds for exploration
            eps = 1e-8
            logits = torch.log(subset_scores / (1 - subset_scores + eps) + eps)
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
            perturbed = logits + gumbel * temperature
            indices = torch.topk(perturbed, k).indices
        else:
            indices = torch.topk(subset_scores, k).indices

        # Log-prob under independent Bernoulli model:
        # log P(S) = sum_{i in S} log(p_i) + sum_{j not in S} log(1 - p_j)
        selection_mask = torch.zeros(V, dtype=torch.bool, device=subset_scores.device)
        selection_mask[indices] = True

        eps = 1e-8
        log_prob = (
            torch.log(subset_scores[selection_mask] + eps).sum()
            + torch.log(1 - subset_scores[~selection_mask] + eps).sum()
        )

        return indices, log_prob

    def confidence(self, subset_scores: torch.Tensor, k: int) -> float:
        """
        Confidence metric: how decisive is the model about top-k selection?
        High confidence = clear separation between top-k and rest.
        Returns value in [0, 1].
        """
        k = min(k, subset_scores.size(0))
        sorted_scores, _ = torch.sort(subset_scores, descending=True)
        top_k_min = sorted_scores[k - 1]
        rest_max = sorted_scores[k] if k < subset_scores.size(0) else torch.tensor(0.0)
        # Gap between weakest selected and strongest not-selected
        gap = (top_k_min - rest_max).clamp(0, 1).item()
        return gap


def build_node_features(
    positions: np.ndarray,
    sizes: np.ndarray,
    macro_hpwl: Optional[np.ndarray] = None,
    macro_congestion: Optional[np.ndarray] = None,
    last_delta: Optional[np.ndarray] = None,
    in_last_subset: Optional[np.ndarray] = None,
    stagnation_frac: float = 0.0,
    iteration_frac: float = 0.0,
    window_fraction: float = 0.15,
    subset_size_frac: float = 0.1,
    sa_temperature_norm: float = 0.0,
) -> np.ndarray:
    """
    Build the 14-dim node feature vector for NeighborhoodGNN.

    Args:
        positions: (V, 2) current centers in [-1, 1]
        sizes: (V, 2) macro sizes in [-1, 1]
        macro_hpwl: (V,) per-macro HPWL contribution, normalized. None -> zeros.
        macro_congestion: (V,) per-macro congestion score. None -> zeros.
        last_delta: (V, 2) displacement from last CP-SAT solve. None -> zeros.
        in_last_subset: (V,) binary mask. None -> zeros.
        stagnation_frac: scalar, broadcast to all nodes.
        iteration_frac: scalar in [0, 1].
        window_fraction: current window fraction.
        subset_size_frac: current subset size / total macros.
        sa_temperature_norm: SA temperature / initial temp.

    Returns:
        features: (V, 14) numpy array
    """
    V = positions.shape[0]

    if macro_hpwl is None:
        macro_hpwl = np.zeros(V, dtype=np.float32)
    else:
        # Normalize to [0, 1]
        hmax = macro_hpwl.max()
        if hmax > 1e-8:
            macro_hpwl = macro_hpwl / hmax

    if macro_congestion is None:
        macro_congestion = np.zeros(V, dtype=np.float32)
    else:
        cmax = macro_congestion.max()
        if cmax > 1e-8:
            macro_congestion = macro_congestion / cmax

    if last_delta is None:
        last_delta = np.zeros((V, 2), dtype=np.float32)

    if in_last_subset is None:
        in_last_subset = np.zeros(V, dtype=np.float32)

    features = np.column_stack([
        positions.astype(np.float32),                         # 2: position_xy
        sizes.astype(np.float32),                             # 2: size_wh
        macro_hpwl.astype(np.float32).reshape(-1, 1),         # 1: macro_hpwl_norm
        macro_congestion.astype(np.float32).reshape(-1, 1),   # 1: macro_congestion
        last_delta.astype(np.float32),                        # 2: last_delta_xy
        in_last_subset.astype(np.float32).reshape(-1, 1),     # 1: in_last_subset
        np.full((V, 1), stagnation_frac, dtype=np.float32),   # 1: stagnation_frac
        np.full((V, 1), iteration_frac, dtype=np.float32),    # 1: iter_frac
        np.full((V, 1), window_fraction, dtype=np.float32),   # 1: window_frac
        np.full((V, 1), subset_size_frac, dtype=np.float32),  # 1: subset_frac
        np.full((V, 1), sa_temperature_norm, dtype=np.float32),  # 1: sa_temp
    ])

    assert features.shape == (V, 14), f"Expected (V, 14), got {features.shape}"
    return features
