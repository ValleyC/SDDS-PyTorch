"""
Categorical noise distribution for discrete diffusion over positions.

This implements the forward and reverse diffusion process for categorical variables
(e.g., position assignments in TSP permutation formulation).

For TSP, each node has a position variable X[i] in {0, 1, ..., N-1}.
The forward process randomly flips positions with probability gamma_t.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from .BaseNoise import BaseNoiseDistribution


class BernoulliNoise(BaseNoiseDistribution):
    """
    Categorical noise distribution for position-based discrete diffusion.

    The forward process gradually randomizes positions:
        p(X_t | X_{t-1}) = (1-gamma_t) if X_t == X_{t-1} else gamma_t / (N-1)

    This means:
        - With probability (1-gamma_t): stay at the same position
        - With probability gamma_t: flip to a uniform random OTHER position
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize categorical noise distribution.

        Args:
            config: Configuration dictionary containing:
                - n_diffusion_steps: Number of diffusion steps
                - diff_schedule: Type of noise schedule
                - n_nodes: Number of nodes (positions) for TSP
                - n_bernoulli_features: Number of categories (= n_nodes for TSP)
        """
        super().__init__(config)
        self.n_categories = config.get("n_bernoulli_features", config.get("n_nodes", 20))

    def combine_losses(
        self,
        noise_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        T: float = 1.0
    ) -> torch.Tensor:
        """
        Combine different loss components.

        Args:
            noise_loss: Loss from noise prediction
            energy_loss: Loss from energy/reward
            entropy_loss: Entropy regularization loss
            T: Temperature parameter for entropy bonus

        Returns:
            Combined loss
        """
        return noise_loss + energy_loss - T * entropy_loss

    def calculate_noise_distr_reward(
        self,
        noise_distr_step: torch.Tensor,
        entropy_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the reward contribution from the noise distribution.

        Args:
            noise_distr_step: Noise distribution step value
            entropy_reward: Entropy reward

        Returns:
            Reward tensor
        """
        return -(noise_distr_step - entropy_reward)

    def calc_noise_step(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse diffusion process.

        Given the model's prediction logits, sample X_{t-1} from the posterior.

        Args:
            logits: Model output logits (batch, n_nodes, n_positions)
            X_t: Current position assignments at timestep t (batch, n_nodes)
            t_idx: Current timestep index (0 to T-1, where 0 is cleanest)
            generator: Random generator

        Returns:
            Tuple of (X_{t-1}, log_prob per sample)
        """
        batch_size = logits.shape[0]
        n_nodes = logits.shape[1]
        n_positions = logits.shape[2]
        device = logits.device

        # Handle extra dimension in X_t
        if X_t.dim() == 3:
            X_t = X_t.squeeze(-1)

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Get predicted probabilities for X_0 (clean state)
        p_x0 = F.softmax(logits, dim=-1)  # (batch, n_nodes, n_positions)
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

        if t_idx == 0:
            # At t=0, sample directly from the predicted distribution
            X_prev = torch.multinomial(
                p_x0.view(-1, n_positions),
                num_samples=1,
                generator=generator
            ).view(batch_size, n_nodes)

            # Compute log probability
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_nodes)
            node_idx = torch.arange(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
            log_prob_per_node = torch.log(p_x0[batch_idx, node_idx, X_prev] + 1e-8)
            log_prob = log_prob_per_node.sum(dim=-1)  # Sum over nodes

            return X_prev, log_prob

        # For t > 0, compute posterior q(X_{t-1} | X_t, p_x0)
        gamma_t = self.get_gamma_t(t_idx)
        gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else torch.tensor(0.0)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(device)
        else:
            gamma_t = torch.tensor(gamma_t, device=device)

        if isinstance(gamma_t_m1, torch.Tensor):
            gamma_t_m1 = gamma_t_m1.to(device)
        else:
            gamma_t_m1 = torch.tensor(gamma_t_m1, device=device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)
        gamma_t_m1 = torch.clamp(gamma_t_m1, min=1e-8, max=1-1e-8)

        # Compute posterior for each position
        # p(X_{t-1} = j | X_t, X_0) propto p(X_t | X_{t-1}=j) * p(X_{t-1}=j | X_0)
        # Simplified: blend predicted X_0 with noise
        # With prob (1-gamma_{t-1}), X_{t-1} = X_0
        # With prob gamma_{t-1} / (N-1), X_{t-1} is uniform random

        # For computational simplicity, use:
        # p_sample = (1 - gamma_{t-1}) * p_x0 + gamma_{t-1} * uniform
        uniform = torch.ones_like(p_x0) / n_positions
        p_sample = (1 - gamma_t_m1) * p_x0 + gamma_t_m1 * uniform

        # Ensure valid probabilities
        p_sample = torch.clamp(p_sample, min=1e-8, max=1-1e-8)
        p_sample = p_sample / p_sample.sum(dim=-1, keepdim=True)

        # Sample from posterior
        X_prev = torch.multinomial(
            p_sample.view(-1, n_positions),
            num_samples=1,
            generator=generator
        ).view(batch_size, n_nodes)

        # Compute log probability
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_nodes)
        node_idx = torch.arange(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        log_prob_per_node = torch.log(p_sample[batch_idx, node_idx, X_prev] + 1e-8)
        log_prob = log_prob_per_node.sum(dim=-1)  # Sum over nodes

        return X_prev, log_prob

    def compute_action_log_prob(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        action: torch.Tensor,
        t_idx: int
    ) -> torch.Tensor:
        """
        Compute log probability of a given action under the posterior distribution.

        Args:
            logits: Model output logits (batch, n_nodes, n_positions)
            X_t: Current state at timestep t (batch, n_nodes)
            action: The action (X_{t-1}) to compute log prob for (batch, n_nodes)
            t_idx: Current timestep index

        Returns:
            Log probability per sample (batch,)
        """
        batch_size = logits.shape[0]
        n_nodes = logits.shape[1]
        n_positions = logits.shape[2]
        device = logits.device

        # Handle extra dimension
        if action.dim() == 3:
            action = action.squeeze(-1)

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Get predicted probabilities for X_0
        p_x0 = F.softmax(logits, dim=-1)
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

        if t_idx == 0:
            # At t=0, use p_x0 directly
            p_sample = p_x0
        else:
            # For t > 0, compute posterior
            gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else torch.tensor(0.0)

            if isinstance(gamma_t_m1, torch.Tensor):
                gamma_t_m1 = gamma_t_m1.to(device)
            else:
                gamma_t_m1 = torch.tensor(gamma_t_m1, device=device)

            gamma_t_m1 = torch.clamp(gamma_t_m1, min=1e-8, max=1-1e-8)

            # Same posterior formula as calc_noise_step
            uniform = torch.ones_like(p_x0) / n_positions
            p_sample = (1 - gamma_t_m1) * p_x0 + gamma_t_m1 * uniform

        # Ensure valid probabilities
        p_sample = torch.clamp(p_sample, min=1e-8, max=1-1e-8)
        p_sample = p_sample / p_sample.sum(dim=-1, keepdim=True)

        # Compute log probability of the given action
        action_long = action.long()
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_nodes)
        node_idx = torch.arange(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        log_prob_per_node = torch.log(p_sample[batch_idx, node_idx, action_long] + 1e-8)

        # Sum over all nodes
        return log_prob_per_node.sum(dim=-1)

    def sample_forward_diff_process(
        self,
        X_t_m1: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Generator]:
        """
        Sample from the forward diffusion process q(X_t | X_{t-1}).

        With probability gamma_t, flip to a random position.
        With probability (1-gamma_t), stay at the same position.

        Args:
            X_t_m1: State at timestep t-1 (batch, n_nodes)
            t_idx: Target timestep index
            generator: Random generator

        Returns:
            Tuple of (X_t, log_probs, generator)
        """
        batch_size, n_nodes = X_t_m1.shape[:2]
        device = X_t_m1.device

        # Handle extra dimension
        if X_t_m1.dim() == 3:
            X_t_m1 = X_t_m1.squeeze(-1)

        gamma_t = self.get_gamma_t(t_idx)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(device)
        else:
            gamma_t = torch.tensor(gamma_t, device=device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        # Sample flip mask: which nodes will flip
        flip_mask = torch.bernoulli(
            torch.full((batch_size, n_nodes), gamma_t.item(), device=device),
            generator=generator
        )

        # Sample new random positions for flipped nodes
        random_positions = torch.randint(
            0, self.n_categories,
            (batch_size, n_nodes),
            device=device,
            generator=generator
        )

        # Apply flips
        X_t = torch.where(flip_mask.bool(), random_positions, X_t_m1.long())

        # Compute log probability
        # p(X_t = x | X_{t-1}) = (1-gamma) if x == X_{t-1} else gamma/N
        same_mask = (X_t == X_t_m1.long()).float()
        log_p_same = torch.log(1 - gamma_t)
        log_p_flip = torch.log(gamma_t / self.n_categories)
        log_prob_per_node = same_mask * log_p_same + (1 - same_mask) * log_p_flip
        log_prob = log_prob_per_node.sum(dim=-1)

        return X_t, log_prob, generator

    def sample_prior(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Sample from the prior distribution (uniform over positions).

        Args:
            shape: (batch_size, n_nodes)
            device: Device to create tensor on
            generator: Random generator

        Returns:
            Sampled position indices
        """
        batch_size, n_nodes = shape[:2]
        return torch.randint(
            0, self.n_categories,
            (batch_size, n_nodes),
            device=device,
            generator=generator
        )

    def get_entropy(
        self,
        logits: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute entropy of the predicted distribution.

        Args:
            logits: Model output logits (batch, n_nodes, n_positions)
            node_graph_idx: Not used for batched mode
            n_graphs: Not used for batched mode

        Returns:
            Entropy per sample (batch,)
        """
        # Numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)

        # Entropy per node: -sum_i p_i log p_i
        entropy_per_node = -torch.sum(probs * torch.log(probs), dim=-1)  # (batch, n_nodes)
        entropy_per_node = torch.nan_to_num(entropy_per_node, nan=0.0)

        # Sum over nodes
        return entropy_per_node.sum(dim=-1)  # (batch,)

    def get_log_p_T_0(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t_idx: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute log probability log p(X_next | X_prev).

        Args:
            X_prev: Previous state (batch, n_nodes)
            X_next: Next state (batch, n_nodes)
            t_idx: Timestep index
            node_graph_idx: Not used
            n_graphs: Not used

        Returns:
            Log probability per sample (batch,)
        """
        # Handle extra dimension
        if X_prev.dim() == 3:
            X_prev = X_prev.squeeze(-1)
        if X_next.dim() == 3:
            X_next = X_next.squeeze(-1)

        gamma_t = self.get_gamma_t(t_idx)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_prev.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_prev.device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        # p(X_next | X_prev) = (1-gamma) if same, else gamma/N
        same_mask = (X_next.long() == X_prev.long()).float()
        log_p_same = torch.log(1 - gamma_t)
        log_p_flip = torch.log(gamma_t / self.n_categories)
        log_prob_per_node = same_mask * log_p_same + (1 - same_mask) * log_p_flip

        return log_prob_per_node.sum(dim=-1)

    def calc_noise_loss(
        self,
        logits: torch.Tensor,
        X_prev: torch.Tensor,
        t_idx: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None,
        T: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the noise-related loss term.

        Args:
            logits: Model output logits (batch, n_nodes, n_positions)
            X_prev: Previous state (batch, n_nodes)
            t_idx: Current timestep
            node_graph_idx: Not used
            n_graphs: Not used
            T: Temperature

        Returns:
            Tuple of (noise_loss, log_prob_per_node)
        """
        # Handle extra dimension
        if X_prev.dim() == 3:
            X_prev = X_prev.squeeze(-1)

        gamma_t = self.get_gamma_t(t_idx)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_prev.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_prev.device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        # Get predicted probabilities
        logits = torch.clamp(logits, min=-50, max=50)
        p_next = F.softmax(logits, dim=-1)  # (batch, n_nodes, n_positions)

        # Create one-hot of previous state
        X_prev_long = X_prev.long()
        one_hot_prev = F.one_hot(X_prev_long, num_classes=self.n_categories).float()

        # Expected log probability under forward noise
        # p(X_t | X_prev) = (1-gamma) * I(X_t == X_prev) + gamma/N
        # E_p[log q(X_t)] where q is the predicted distribution
        log_p_next = torch.log(p_next + 1e-8)

        # Expected log prob = sum_x p(x|X_prev) * log q(x)
        # = (1-gamma) * log q(X_prev) + (gamma/N) * sum_x log q(x)
        log_q_prev = (log_p_next * one_hot_prev).sum(dim=-1)  # (batch, n_nodes)
        mean_log_q = log_p_next.mean(dim=-1)  # (batch, n_nodes)

        noise_per_node = (1 - gamma_t) * log_q_prev + gamma_t * mean_log_q
        noise_loss = T * noise_per_node.sum(dim=-1)  # (batch,)

        return noise_loss, noise_per_node
