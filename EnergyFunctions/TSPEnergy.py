"""
TSP Energy function using permutation matrix representation.

For TSP, the solution is represented as a permutation matrix where:
- X[i, p] = 1 means node i is at position p in the tour
- Each row sums to 1 (each node in exactly one position)
- Each column sums to 1 (each position has exactly one node)

The tour visits positions 0 -> 1 -> 2 -> ... -> N-1 -> 0.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .BaseEnergy import BaseEnergyClass


class TSPEnergyClass(BaseEnergyClass):
    """
    Energy function for TSP with permutation matrix representation.

    The state X_0 has shape (batch, n_nodes) where X_0[b, i] is the position
    of node i in the tour (integer from 0 to n_nodes-1).

    This is converted to a permutation matrix x_mat[i, p] = 1 if node i at position p.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TSP energy class.

        Args:
            config: Configuration dictionary containing:
                - n_nodes: Number of nodes in TSP
                - penalty_weight: Weight for constraint violations (default: 1.45)
        """
        super().__init__(config)
        self.n_nodes = config.get("n_nodes", 20)
        self.penalty_weight = config.get("penalty_weight", 1.45)

        # Create cyclic permutation matrix for computing tour length
        # This matrix shifts position p to position (p+1) mod N
        self._init_cyclic_perm_matrix(self.n_nodes)

    def _init_cyclic_perm_matrix(self, n_nodes: int):
        """Initialize the cyclic permutation matrix."""
        # cycl_perm_mat[p, q] = 1 if q = (p+1) mod N
        # i.e., it maps position p to position p+1
        cycl_perm_mat = torch.zeros(n_nodes, n_nodes)
        for i in range(n_nodes - 1):
            cycl_perm_mat[i, i + 1] = 1.0
        cycl_perm_mat[n_nodes - 1, 0] = 1.0  # Wrap around

        self.register_buffer('cycl_perm_mat', cycl_perm_mat)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer (will be moved to correct device)."""
        setattr(self, name, tensor)

    def calculate_Energy(
        self,
        coords: torch.Tensor,
        X_0_classes: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate TSP energy using permutation matrix formulation.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            X_0_classes: Position assignments (batch, n_nodes) with values 0 to n_nodes-1
            node_graph_idx: Not used for dense representation

        Returns:
            Tuple of:
                - energy: Total energy per graph (tour length + penalties)
                - violations_per_node: Constraint violations per node
                - constraint_penalty: Total constraint penalty (HB) per graph
        """
        batch_size, n_nodes = X_0_classes.shape[:2]
        device = coords.device

        # Ensure cyclic permutation matrix is on correct device
        if self.cycl_perm_mat.device != device:
            self.cycl_perm_mat = self.cycl_perm_mat.to(device)

        # Handle case where X_0_classes might have extra dimension
        if X_0_classes.dim() == 3:
            X_0_classes = X_0_classes.squeeze(-1)

        # Convert to long for one_hot
        X_0_classes = X_0_classes.long()

        # Convert position indices to one-hot permutation matrix
        # x_mat[b, i, p] = 1 if node i is at position p
        x_mat = F.one_hot(X_0_classes, num_classes=n_nodes).float()  # (batch, n_nodes, n_nodes)

        # Compute distance matrix
        # dist[b, i, j] = distance between node i and node j
        dist_matrix = torch.sqrt(
            torch.sum((coords[:, :, None, :] - coords[:, None, :, :]) ** 2, dim=-1)
        )  # (batch, n_nodes, n_nodes)

        # Compute tour length using cyclic permutation
        # x_mat_cycl[b, i, p] = x_mat[b, i, (p+1) mod N]
        # This tells us: if node i is at position (p+1), x_mat_cycl[b, i, p] = 1
        x_mat_cycl = torch.matmul(x_mat, self.cycl_perm_mat)  # (batch, n_nodes, n_nodes)

        # H_mat[b, i, j] = sum_p x_mat[b, i, p] * x_mat_cycl[b, j, p]
        #                = sum_p x_mat[b, i, p] * x_mat[b, j, (p+1) mod N]
        # This equals 1 if node i is at position p and node j is at position p+1
        # i.e., if edge (i, j) is in the tour
        H_mat = torch.bmm(x_mat_cycl, x_mat.transpose(-2, -1))  # (batch, n_nodes, n_nodes)

        # Tour length = sum of distances for edges in the tour
        tour_length = torch.sum(H_mat * dist_matrix, dim=(-2, -1))  # (batch,)

        # Constraint 1: Each position has exactly one node
        # Obj1 = A * sum_p (sum_i x_mat[i, p] - 1)^2
        col_sum = torch.sum(x_mat, dim=1)  # (batch, n_nodes) - sum over nodes for each position
        Obj1 = torch.sum((col_sum - 1) ** 2, dim=-1)  # (batch,)

        # Constraint 2: Each node is at exactly one position
        # Obj2 = A * sum_i (sum_p x_mat[i, p] - 1)^2
        row_sum = torch.sum(x_mat, dim=2)  # (batch, n_nodes) - sum over positions for each node
        Obj2 = torch.sum((row_sum - 1) ** 2, dim=-1)  # (batch,)

        # Constraint penalty
        A = self.penalty_weight
        HB = A * Obj1 + A * Obj2  # (batch,)

        # Total energy
        energy = tour_length + HB  # (batch,)

        # Per-node violations: check if each node appears exactly once
        violations_per_node = torch.abs(row_sum - 1)  # (batch, n_nodes)

        return energy, violations_per_node, HB

    def calculate_relaxed_Energy(
        self,
        coords: torch.Tensor,
        p_mat: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate relaxed energy using soft probabilities.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            p_mat: Position probabilities (batch, n_nodes, n_nodes)
                   p_mat[b, i, p] = probability that node i is at position p
            node_graph_idx: Not used

        Returns:
            Relaxed energy (differentiable)
        """
        batch_size, n_nodes, _ = p_mat.shape
        device = coords.device

        # Ensure cyclic permutation matrix is on correct device
        if self.cycl_perm_mat.device != device:
            self.cycl_perm_mat = self.cycl_perm_mat.to(device)

        # Compute distance matrix
        dist_matrix = torch.sqrt(
            torch.sum((coords[:, :, None, :] - coords[:, None, :, :]) ** 2, dim=-1) + 1e-8
        )

        # Relaxed tour length
        p_mat_cycl = torch.matmul(p_mat, self.cycl_perm_mat)
        H_mat = torch.bmm(p_mat_cycl, p_mat.transpose(-2, -1))
        tour_length = torch.sum(H_mat * dist_matrix, dim=(-2, -1))

        # Soft constraints
        col_sum = torch.sum(p_mat, dim=1)
        row_sum = torch.sum(p_mat, dim=2)

        A = self.penalty_weight
        Obj1 = A * torch.sum((col_sum - 1) ** 2, dim=-1)
        Obj2 = A * torch.sum((row_sum - 1) ** 2, dim=-1)

        return tour_length + Obj1 + Obj2

    def decode_tour(
        self,
        X_0_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode position assignments to tour order.

        Args:
            X_0_classes: Position assignments (batch, n_nodes)

        Returns:
            tour: Node order (batch, n_nodes) where tour[b, p] is the node at position p
        """
        if X_0_classes.dim() == 3:
            X_0_classes = X_0_classes.squeeze(-1)

        batch_size, n_nodes = X_0_classes.shape
        device = X_0_classes.device

        # Sort by position to get tour order
        # tour[b, p] = node at position p
        # Use -1 as sentinel for empty slots (node 0 is valid!)
        tour = torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=device)
        filled = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)

        for b in range(batch_size):
            for node in range(n_nodes):
                pos = X_0_classes[b, node].item()
                # Handle duplicate positions by finding first empty slot
                pos = pos % n_nodes
                start_pos = pos
                while filled[b, pos]:
                    pos = (pos + 1) % n_nodes
                    if pos == start_pos:
                        # All positions filled (shouldn't happen with valid permutation)
                        break
                tour[b, pos] = node
                filled[b, pos] = True

        return tour

    def calculate_tour_length_from_tour(
        self,
        coords: torch.Tensor,
        tour: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate tour length from explicit tour order.

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            tour: Tour order (batch, n_nodes)

        Returns:
            Tour length per graph
        """
        batch_size, n_nodes = tour.shape

        # Get coordinates in tour order
        batch_idx = torch.arange(batch_size, device=tour.device).unsqueeze(1).expand(-1, n_nodes)
        tour_coords = coords[batch_idx, tour]  # (batch, n_nodes, 2)

        # Calculate distances between consecutive nodes
        next_coords = torch.roll(tour_coords, -1, dims=1)
        distances = torch.sqrt(torch.sum((tour_coords - next_coords) ** 2, dim=-1))

        return torch.sum(distances, dim=-1)

    def calculate_tour_length(
        self,
        coords: torch.Tensor,
        X_0_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate only the tour length (without constraint penalties).

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            X_0_classes: Position assignments (batch, n_nodes)

        Returns:
            Tour length per graph
        """
        batch_size, n_nodes = X_0_classes.shape[:2]
        device = coords.device

        # Ensure cyclic permutation matrix is on correct device
        if self.cycl_perm_mat.device != device:
            self.cycl_perm_mat = self.cycl_perm_mat.to(device)

        # Handle extra dimension
        if X_0_classes.dim() == 3:
            X_0_classes = X_0_classes.squeeze(-1)

        X_0_classes = X_0_classes.long()

        # Convert to one-hot
        x_mat = F.one_hot(X_0_classes, num_classes=n_nodes).float()

        # Distance matrix
        dist_matrix = torch.sqrt(
            torch.sum((coords[:, :, None, :] - coords[:, None, :, :]) ** 2, dim=-1)
        )

        # Tour length using cyclic permutation
        x_mat_cycl = torch.matmul(x_mat, self.cycl_perm_mat)
        H_mat = torch.bmm(x_mat_cycl, x_mat.transpose(-2, -1))
        tour_length = torch.sum(H_mat * dist_matrix, dim=(-2, -1))

        return tour_length

    def calculate_constraint_violations(
        self,
        X_0_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate constraint violations for the permutation matrix.

        Args:
            X_0_classes: Position assignments (batch, n_nodes)

        Returns:
            Tuple of (row_violations, col_violations) per graph
        """
        if X_0_classes.dim() == 3:
            X_0_classes = X_0_classes.squeeze(-1)

        batch_size, n_nodes = X_0_classes.shape
        X_0_classes = X_0_classes.long()

        # Convert to one-hot
        x_mat = F.one_hot(X_0_classes, num_classes=n_nodes).float()

        # Row sum violations: each node should be at exactly one position
        row_sum = torch.sum(x_mat, dim=2)  # (batch, n_nodes)
        row_violations = torch.sum((row_sum - 1) ** 2, dim=-1)  # (batch,)

        # Column sum violations: each position should have exactly one node
        col_sum = torch.sum(x_mat, dim=1)  # (batch, n_nodes)
        col_violations = torch.sum((col_sum - 1) ** 2, dim=-1)  # (batch,)

        return row_violations, col_violations

    def get_reward(
        self,
        coords: torch.Tensor,
        X_0_classes: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get reward for RL training (negative energy).

        Args:
            coords: Node coordinates (batch, n_nodes, 2)
            X_0_classes: Position assignments (batch, n_nodes)
            normalize: Whether to normalize by number of nodes

        Returns:
            Reward per graph
        """
        energy, _, violations = self.calculate_Energy(coords, X_0_classes)

        # Negative energy as reward
        reward = -energy

        if normalize:
            n_nodes = coords.shape[1]
            reward = reward / n_nodes

        return reward
