"""
TSP Energy function for binary edge representation.

For TSP, a valid solution is a Hamiltonian cycle where:
- Each node has exactly degree 2
- The tour is a single connected cycle (no subtours)

The energy is the sum of edge distances in the tour.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .BaseEnergy import BaseEnergyClass


class TSPEnergyClass(BaseEnergyClass):
    """
    Energy function for TSP with binary edge representation.
    
    The adjacency matrix A[i,j] = 1 means edge (i,j) is in the tour.
    For an undirected tour, A should be symmetric.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TSP energy class.
        
        Args:
            config: Configuration dictionary containing:
                - penalty_weight: Weight for constraint violations (default: 1.0)
                - n_bernoulli_features: Number of output classes (default: 2)
        """
        super().__init__(config)
        self.penalty_weight = config.get("penalty_weight", 1.0)
    
    def calculate_Energy(
        self,
        coords: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate TSP energy (tour length + constraint penalties).
        
        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            adj_matrix: Edge adjacency matrix (batch, n_nodes, n_nodes)
            node_graph_idx: Not used for dense representation
            
        Returns:
            Tuple of:
                - energy: Total energy per graph (tour length + penalties)
                - degree_violations: Degree violations per node
                - total_violations: Total constraint violations per graph
        """
        # Make adjacency symmetric (undirected)
        adj_sym = (adj_matrix + adj_matrix.transpose(-2, -1)) / 2
        
        # Calculate tour length
        tour_length = self.calculate_tour_length(coords, adj_sym)
        
        # Calculate degree violations
        degree_violations, total_degree_violations = self.calculate_degree_violations(
            adj_sym, target_degree=2
        )
        
        # Calculate subtour penalty (approximation using connectivity check)
        subtour_penalty = self.calculate_subtour_penalty(adj_sym)
        
        # Total constraint violations
        total_violations = total_degree_violations + subtour_penalty
        
        # Total energy = objective + penalty * violations
        energy = tour_length + self.penalty_weight * total_violations
        
        return energy, degree_violations, total_violations
    
    def calculate_relaxed_Energy(
        self,
        coords: torch.Tensor,
        p: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate relaxed energy using soft probabilities.
        
        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            p: Edge probabilities (batch, n_nodes, n_nodes)
            node_graph_idx: Not used
            
        Returns:
            Relaxed energy (differentiable)
        """
        # Make probabilities symmetric
        p_sym = (p + p.transpose(-2, -1)) / 2
        
        # Relaxed tour length (expected tour length)
        tour_length = self.calculate_tour_length(coords, p_sym)
        
        # Soft degree constraint
        degree = torch.sum(p_sym, dim=-1)  # (batch, n_nodes)
        degree_penalty = torch.sum((degree - 2) ** 2, dim=-1)  # L2 penalty
        
        # Total relaxed energy
        energy = tour_length + self.penalty_weight * degree_penalty
        
        return energy
    
    def calculate_subtour_penalty(
        self,
        adj_matrix: torch.Tensor,
        n_iterations: int = 10
    ) -> torch.Tensor:
        """
        Calculate subtour penalty using iterative connectivity check.
        
        This approximates connectivity by checking if all nodes are reachable
        from node 0 using power iteration on the adjacency matrix.
        
        Args:
            adj_matrix: Adjacency matrix (batch, n_nodes, n_nodes)
            n_iterations: Number of iterations for connectivity check
            
        Returns:
            Subtour penalty per graph
        """
        batch_size, n_nodes, _ = adj_matrix.shape
        
        # Start from node 0
        reached = torch.zeros(batch_size, n_nodes, device=adj_matrix.device)
        reached[:, 0] = 1.0
        
        # Propagate through adjacency
        for _ in range(n_iterations):
            reached = torch.clamp(reached + torch.bmm(reached.unsqueeze(1), adj_matrix).squeeze(1), max=1.0)
        
        # Count unreached nodes
        unreached = n_nodes - torch.sum(reached, dim=-1)
        
        return unreached
    
    def get_reward(
        self,
        coords: torch.Tensor,
        adj_matrix: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get reward for RL training (negative energy).
        
        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            adj_matrix: Adjacency matrix (batch, n_nodes, n_nodes)
            normalize: Whether to normalize by number of nodes
            
        Returns:
            Reward per graph
        """
        energy, _, violations = self.calculate_Energy(coords, adj_matrix)
        
        # Negative energy as reward (lower energy = higher reward)
        reward = -energy
        
        if normalize:
            n_nodes = coords.shape[1]
            reward = reward / n_nodes
        
        return reward
