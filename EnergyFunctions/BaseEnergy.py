"""
Base class for energy functions in combinatorial optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class BaseEnergyClass(ABC):
    """
    Abstract base class for energy functions in combinatorial optimization.

    Energy functions define the objective and constraints for different
    CO problems (TSP, MIS, MaxCut, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the energy class.

        Args:
            config: Configuration dictionary containing:
                - n_bernoulli_features: Number of output classes (2 for binary)
                - penalty_weight: Weight for constraint violations
        """
        self.config = config
        self.n_bernoulli_features = config.get("n_bernoulli_features", 2)
        self.penalty_weight = config.get("penalty_weight", 1.0)

    @abstractmethod
    def calculate_Energy(
        self,
        graph_data: Any,
        x_0: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the energy (cost) for a given solution.

        Args:
            graph_data: Graph data (coordinates, edges, etc.)
            x_0: Solution tensor (e.g., edge adjacency matrix)
            node_graph_idx: Index mapping nodes to graphs in batch

        Returns:
            Tuple of:
                - energy: Total energy per graph (objective + penalties)
                - violations_per_node: Constraint violations per node
                - constraint_violation_per_graph: Total constraint violation per graph
        """
        pass

    @abstractmethod
    def calculate_relaxed_Energy(
        self,
        graph_data: Any,
        p: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate relaxed energy using soft probabilities.

        Args:
            graph_data: Graph data
            p: Probability tensor (soft assignments)
            node_graph_idx: Index mapping nodes to graphs

        Returns:
            Relaxed energy tensor
        """
        pass

    def calculate_Energy_per_node(
        self,
        graph_data: Any,
        x_0: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate energy per node (wrapper for batched computation).

        Args:
            graph_data: Graph data
            x_0: Solution tensor
            node_graph_idx: Index mapping nodes to graphs

        Returns:
            Tuple of (energy, constraint_violations)
        """
        energy, _, hb_per_graph = self.calculate_Energy(graph_data, x_0, node_graph_idx)
        return energy, hb_per_graph

    def calculate_Energy_loss(
        self,
        graph_data: Any,
        logits: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate energy-based loss from model logits.

        Args:
            graph_data: Graph data
            logits: Model output logits
            node_graph_idx: Index mapping nodes to graphs

        Returns:
            Energy loss tensor
        """
        # Convert logits to probabilities
        p = torch.softmax(logits, dim=-1)[..., 1]  # Probability of edge being present
        return self.calculate_relaxed_Energy(graph_data, p, node_graph_idx)

    def calculate_Energy_feasible(
        self,
        graph_data: Any,
        x_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate energy only for feasible solutions.

        Args:
            graph_data: Graph data
            x_0: Solution tensor

        Returns:
            Tuple of (feasible_energy, is_feasible, violations)
        """
        energy, violations_per_node, hb_per_graph = self.calculate_Energy(graph_data, x_0)

        # Check feasibility (no constraint violations)
        is_feasible = (hb_per_graph == 0).float()

        # Mask infeasible solutions with large energy
        feasible_energy = torch.where(
            is_feasible.bool(),
            energy,
            torch.full_like(energy, float('inf'))
        )

        return feasible_energy, is_feasible, violations_per_node

    def calculate_tour_length(
        self,
        coords: torch.Tensor,
        edge_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the total tour length from an edge adjacency matrix.
        (Specific to TSP, can be overridden)

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            edge_matrix: Edge adjacency matrix (batch, n_nodes, n_nodes)

        Returns:
            Tour length per graph
        """
        # Compute pairwise distances
        # coords: (batch, n, 2)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, n, n, 2)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # (batch, n, n)

        # Compute tour length as sum of selected edges
        # For undirected tour, divide by 2 (each edge counted twice)
        tour_length = torch.sum(edge_matrix * dist_matrix, dim=(-2, -1)) / 2

        return tour_length

    def calculate_degree_violations(
        self,
        edge_matrix: torch.Tensor,
        target_degree: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate degree constraint violations.
        (For TSP, each node should have exactly degree 2)

        Args:
            edge_matrix: Edge adjacency matrix (batch, n_nodes, n_nodes)
            target_degree: Target degree for each node

        Returns:
            Tuple of (violations_per_node, total_violations_per_graph)
        """
        # Compute degree of each node
        degree = torch.sum(edge_matrix, dim=-1)  # (batch, n_nodes)

        # Violations: deviation from target degree
        violations_per_node = torch.abs(degree - target_degree)

        # Total violations per graph
        total_violations = torch.sum(violations_per_node, dim=-1)

        return violations_per_node, total_violations

    @staticmethod
    def apply_conditional_expectation(
        graph_data: Any,
        p: torch.Tensor,
        energy_func: callable,
        n_classes: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Conditional Expectation (CE) post-processing to improve solutions.

        This iteratively fixes each variable to its best value given the others.

        Args:
            graph_data: Graph data
            p: Soft probability assignments
            energy_func: Energy calculation function
            n_classes: Number of classes

        Returns:
            Tuple of (best_solution, best_energy)
        """
        # Convert soft to hard assignments
        x_0 = (p > 0.5).float()

        # Get initial energy
        best_energy, _, _ = energy_func(graph_data, x_0)
        best_x = x_0.clone()

        # Iteratively improve
        n_nodes = x_0.shape[-2] if x_0.dim() > 1 else x_0.shape[0]

        for i in range(n_nodes):
            for c in range(n_classes):
                # Try setting node i to class c
                x_test = best_x.clone()
                if x_test.dim() == 2:
                    x_test[i] = c
                else:
                    x_test[:, i] = c

                # Compute energy
                test_energy, _, _ = energy_func(graph_data, x_test)

                # Keep if better
                improved = test_energy < best_energy
                if improved.any():
                    best_x = torch.where(improved.unsqueeze(-1), x_test, best_x)
                    best_energy = torch.where(improved, test_energy, best_energy)

        return best_x, best_energy
