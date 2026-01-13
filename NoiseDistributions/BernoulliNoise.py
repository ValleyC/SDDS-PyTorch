"""
Bernoulli noise distribution for discrete diffusion.

This implements the forward and reverse diffusion process for binary variables
(e.g., edge presence/absence in graphs).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from .BaseNoise import BaseNoiseDistribution


class BernoulliNoise(BaseNoiseDistribution):
    """
    Bernoulli noise distribution for binary discrete diffusion.

    The forward process gradually flips bits with probability gamma_t:
        p(X_t | X_{t-1}) = (1-gamma_t) if X_t == X_{t-1} else gamma_t

    This is equivalent to:
        X_t = X_{t-1} with probability (1-gamma_t)
        X_t = 1-X_{t-1} with probability gamma_t
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bernoulli noise distribution.

        Args:
            config: Configuration dictionary containing:
                - n_diffusion_steps: Number of diffusion steps
                - diff_schedule: Type of noise schedule
                - beta_factor: Scaling factor for beta
        """
        super().__init__(config)

    def combine_losses(
        self,
        noise_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        T: float = 1.0
    ) -> torch.Tensor:
        """
        Combine different loss components.

        The total loss is: L_noise + L_energy - T * L_entropy

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
            Reward tensor: -(noise_distr_step - entropy_reward)
        """
        return -(noise_distr_step - entropy_reward)

    def get_log_p_T_0_per_node(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t_idx: int
    ) -> torch.Tensor:
        """
        Compute log probability log p(X_next | X_prev) per node.

        For Bernoulli noise:
            log p(X_t | X_{t-1}) = X_{t-1} * (X_t * log(1-gamma) + (1-X_t) * log(gamma))
                                + (1-X_{t-1}) * ((1-X_t) * log(1-gamma) + X_t * log(gamma))

        Args:
            X_prev: Previous state (batch, ..., features)
            X_next: Next state (batch, ..., features)
            t_idx: Timestep index

        Returns:
            Log probability per node (batch, ...)
        """
        gamma_t = self.get_gamma_t(t_idx)

        # Ensure gamma is on the same device
        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_prev.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_prev.device)

        # Clamp gamma to avoid log(0)
        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        X_next_down = 1 - X_next
        log_1_minus_gamma = torch.log(1 - gamma_t)
        log_gamma = torch.log(gamma_t)

        # Compute log probability
        # When X_prev = 1: p(same) = 1-gamma, p(flip) = gamma
        # When X_prev = 0: p(same) = 1-gamma, p(flip) = gamma
        noise_per_node = torch.sum(
            X_prev * (X_next * log_1_minus_gamma + X_next_down * log_gamma) +
            (1 - X_prev) * (X_next_down * log_1_minus_gamma + X_next * log_gamma),
            dim=-1
        )

        return noise_per_node

    def get_log_p_T_0(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t_idx: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute log probability log p(X_next | X_prev) aggregated per graph.

        Args:
            X_prev: Previous state
            X_next: Next state
            t_idx: Timestep index
            node_graph_idx: Index mapping nodes to graphs (for batched graphs)
            n_graphs: Number of graphs in batch

        Returns:
            Log probability per graph
        """
        log_p_per_node = self.get_log_p_T_0_per_node(X_prev, X_next, t_idx)

        if node_graph_idx is not None and n_graphs is not None:
            # Aggregate per graph using scatter_add
            log_p_per_graph = torch.zeros(n_graphs, device=log_p_per_node.device)
            log_p_per_graph.scatter_add_(0, node_graph_idx, log_p_per_node)
        else:
            # Assume batch dimension is first
            if log_p_per_node.dim() > 1:
                log_p_per_graph = log_p_per_node.sum(dim=tuple(range(1, log_p_per_node.dim())))
            else:
                log_p_per_graph = log_p_per_node

        return log_p_per_graph

    def sample_forward_diff_process(
        self,
        X_t_m1: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from the forward diffusion process q(X_t | X_{t-1}).

        This flips each bit with probability gamma_t.

        Args:
            X_t_m1: State at timestep t-1
            t_idx: Target timestep index
            generator: Random generator for reproducibility

        Returns:
            Tuple of:
                - X_t: Sampled state at timestep t
                - spin_log_probs: Log probabilities of sampled states
                - generator: Updated generator
        """
        gamma_t = self.get_gamma_t(t_idx)

        # Ensure gamma is on the same device
        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_t_m1.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_t_m1.device, dtype=X_t_m1.dtype)

        # Clamp gamma to avoid log(0)
        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        X_next_down = 1 - X_t_m1

        # Compute log probabilities for each class (0 and 1)
        log_1_minus_gamma = torch.log(1 - gamma_t)
        log_gamma = torch.log(gamma_t)

        # log p(X_t = 1 | X_{t-1})
        log_p_up = X_t_m1 * log_1_minus_gamma + X_next_down * log_gamma
        # log p(X_t = 0 | X_{t-1})
        log_p_down = X_next_down * log_1_minus_gamma + X_t_m1 * log_gamma

        # Stack to get logits for categorical sampling
        log_p_per_node = torch.stack([log_p_down, log_p_up], dim=-1)

        # Sample from categorical distribution
        probs = F.softmax(log_p_per_node, dim=-1)
        X_next = torch.bernoulli(probs[..., 1], generator=generator)

        # Compute log probability of sampled state
        one_hot_state = F.one_hot(X_next.long(), num_classes=2).float()
        spin_log_probs = torch.sum(log_p_per_node * one_hot_state, dim=-1)

        return X_next, spin_log_probs, generator

    def sample_forward_from_x0(
        self,
        x0: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample X_t directly from X_0 (marginal distribution).

        For Bernoulli noise, we can compute the marginal:
            q(X_t | X_0) where the cumulative flip probability is:
            alpha_t = product of (1 - 2*gamma_i) for i in 1..t

        Args:
            x0: Clean state at t=0
            t_idx: Target timestep
            generator: Random generator

        Returns:
            Tuple of (x_t, log_prob)
        """
        # Compute cumulative flip probability
        # For each step, prob of staying same: (1-gamma)
        # For t steps: need to track cumulative effect

        # Simplified: sample step by step (can be optimized)
        x_t = x0.clone()
        total_log_prob = torch.zeros(x0.shape[:-1], device=x0.device)

        for t in range(t_idx):
            x_t, log_prob, generator = self.sample_forward_diff_process(x_t, t, generator)
            total_log_prob = total_log_prob + log_prob

        return x_t, total_log_prob

    def calc_noise_loss(
        self,
        spin_logits_next: torch.Tensor,
        X_prev: torch.Tensor,
        t_idx: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None,
        T: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the noise-related loss term.

        This computes the KL divergence between the model's predicted
        transition and the true noise distribution.

        Args:
            spin_logits_next: Model output logits (batch, ..., 2)
            X_prev: Previous state
            t_idx: Current timestep
            node_graph_idx: Index mapping nodes to graphs
            n_graphs: Number of graphs
            T: Temperature

        Returns:
            Tuple of (noise_loss_per_graph, log_prob_per_node)
        """
        gamma_t = self.get_gamma_t(t_idx)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_prev.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_prev.device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)

        # Get predicted probabilities
        p_next_up = torch.softmax(spin_logits_next, dim=-1)[..., 1]
        p_next_down = 1 - p_next_up

        log_1_minus_gamma = torch.log(1 - gamma_t)
        log_gamma = torch.log(gamma_t)

        # Compute expected log probability under predicted distribution
        noise_per_node = torch.sum(
            X_prev * (p_next_up * log_1_minus_gamma + p_next_down * log_gamma) +
            (1 - X_prev) * (p_next_down * log_1_minus_gamma + p_next_up * log_gamma),
            dim=-1
        )

        # Aggregate per graph
        if node_graph_idx is not None and n_graphs is not None:
            noise_per_graph = torch.zeros(n_graphs, device=noise_per_node.device)
            noise_per_graph.scatter_add_(0, node_graph_idx, noise_per_node)
        else:
            if noise_per_node.dim() > 1:
                noise_per_graph = noise_per_node.sum(dim=tuple(range(1, noise_per_node.dim())))
            else:
                noise_per_graph = noise_per_node

        return T * noise_per_graph, noise_per_node

    def calc_noise_step(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse diffusion process.

        Given the model's prediction of X_0, sample X_{t-1}.

        Args:
            logits: Model output logits for X_0 prediction
            X_t: Current noisy state at timestep t
            t_idx: Current timestep
            generator: Random generator

        Returns:
            Tuple of (X_{t-1}, log_prob)
        """
        # Get predicted probabilities for X_0
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p_x0 = torch.softmax(logits, dim=-1)
        # Ensure probabilities are valid
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)
        # Renormalize
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

        if t_idx == 0:
            # At t=0, just take argmax or sample from predicted distribution
            X_prev = torch.bernoulli(p_x0[..., 1], generator=generator)
            log_prob = torch.sum(
                torch.log(torch.clamp(
                    X_prev * p_x0[..., 1] + (1 - X_prev) * p_x0[..., 0],
                    min=1e-8
                )),
                dim=-1
            )
            return X_prev, log_prob

        # For t > 0, compute posterior q(X_{t-1} | X_t, X_0)
        gamma_t = self.get_gamma_t(t_idx)
        gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else torch.tensor(0.0)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(X_t.device)
        else:
            gamma_t = torch.tensor(gamma_t, device=X_t.device)

        if isinstance(gamma_t_m1, torch.Tensor):
            gamma_t_m1 = gamma_t_m1.to(X_t.device)
        else:
            gamma_t_m1 = torch.tensor(gamma_t_m1, device=X_t.device)

        gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)
        gamma_t_m1 = torch.clamp(gamma_t_m1, min=1e-8, max=1-1e-8)

        # Compute posterior
        # p(X_{t-1} | X_t, X_0) proportional to p(X_t | X_{t-1}) * p(X_{t-1} | X_0)
        p_x0_1 = p_x0[..., 1]  # p(X_0 = 1)
        p_x0_0 = p_x0[..., 0]  # p(X_0 = 0)

        # p(X_{t-1} = 1 | X_t, X_0)
        # Simplified: use predicted X_0 to guide sampling
        # This is an approximation used in many discrete diffusion implementations

        # Sample from the model's prediction with some noise
        p_sample = p_x0_1 * (1 - gamma_t_m1) + p_x0_0 * gamma_t_m1

        # Ensure valid probability and handle NaN
        p_sample = torch.nan_to_num(p_sample, nan=0.5)
        p_sample = torch.clamp(p_sample, min=1e-6, max=1-1e-6)

        X_prev = torch.bernoulli(p_sample, generator=generator)

        # Compute log probability
        log_prob = torch.sum(
            torch.log(torch.clamp(
                X_prev * p_sample + (1 - X_prev) * (1 - p_sample),
                min=1e-8
            )),
            dim=-1
        )

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

        This computes log p(action | X_t, logits) using the same posterior
        distribution used in calc_noise_step, ensuring consistency for PPO.

        Args:
            logits: Model output logits for X_0 prediction
            X_t: Current noisy state at timestep t
            action: The action (X_{t-1}) to compute log prob for
            t_idx: Current timestep index

        Returns:
            Log probability per sample (B,) summed over all edges
        """
        # Get predicted probabilities for X_0
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p_x0 = torch.softmax(logits, dim=-1)
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

        if t_idx == 0:
            # At t=0, use p_x0 directly
            p_sample = p_x0[..., 1]
        else:
            # For t > 0, compute posterior p_sample
            gamma_t = self.get_gamma_t(t_idx)
            gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else torch.tensor(0.0)

            if isinstance(gamma_t, torch.Tensor):
                gamma_t = gamma_t.to(action.device)
            else:
                gamma_t = torch.tensor(gamma_t, device=action.device)

            if isinstance(gamma_t_m1, torch.Tensor):
                gamma_t_m1 = gamma_t_m1.to(action.device)
            else:
                gamma_t_m1 = torch.tensor(gamma_t_m1, device=action.device)

            gamma_t = torch.clamp(gamma_t, min=1e-8, max=1-1e-8)
            gamma_t_m1 = torch.clamp(gamma_t_m1, min=1e-8, max=1-1e-8)

            p_x0_1 = p_x0[..., 1]
            p_x0_0 = p_x0[..., 0]

            # Same posterior formula as calc_noise_step
            p_sample = p_x0_1 * (1 - gamma_t_m1) + p_x0_0 * gamma_t_m1

        # Ensure valid probability
        p_sample = torch.nan_to_num(p_sample, nan=0.5)
        p_sample = torch.clamp(p_sample, min=1e-6, max=1-1e-6)

        # Compute log probability of the given action
        # p(action=1) = p_sample, p(action=0) = 1 - p_sample
        log_prob_per_edge = torch.where(
            action == 1,
            torch.log(p_sample),
            torch.log(1 - p_sample)
        )

        # Sum over all edges for per-sample log prob
        return log_prob_per_edge.sum(dim=(-2, -1))

    def calc_noise_step_deterministic(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        t_idx: int,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Perform deterministic reverse step (for final decoding).

        Args:
            logits: Model output logits
            X_t: Current state
            t_idx: Current timestep
            threshold: Threshold for binarization

        Returns:
            X_{t-1}: Deterministic next state
        """
        p_x0 = torch.softmax(logits, dim=-1)[..., 1]
        return (p_x0 > threshold).float()

    def get_entropy(
        self,
        logits: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute entropy of the predicted distribution.

        Args:
            logits: Model output logits
            node_graph_idx: Index mapping nodes to graphs
            n_graphs: Number of graphs

        Returns:
            Entropy per graph
        """
        # Numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        entropy_per_node = -torch.sum(
            probs * torch.log(probs),
            dim=-1
        )
        entropy_per_node = torch.nan_to_num(entropy_per_node, nan=0.0)

        # Sum over feature dimension if present
        if entropy_per_node.dim() > 1:
            entropy_per_node = entropy_per_node.sum(dim=-1)

        if node_graph_idx is not None and n_graphs is not None:
            entropy_per_graph = torch.zeros(n_graphs, device=entropy_per_node.device)
            entropy_per_graph.scatter_add_(0, node_graph_idx, entropy_per_node)
        else:
            if entropy_per_node.dim() > 1:
                entropy_per_graph = entropy_per_node.sum(dim=tuple(range(1, entropy_per_node.dim())))
            else:
                entropy_per_graph = entropy_per_node

        return entropy_per_graph
