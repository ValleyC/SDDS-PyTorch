"""
Categorical noise distribution for discrete diffusion over N categories.

This implements the forward and reverse diffusion process for categorical variables
(e.g., position assignments in TSP permutation formulation).

For TSP, each node has a position variable X[i] in {0, 1, ..., N-1}.

Transition probabilities:
    p(X_t = j | X_{t-1} = i) = p_stay     if j == i
                             = p_change   if j != i

Where:
    beta_t = 2 * gamma_t
    p_stay = 1 - beta_t + beta_t/N = 1 - beta_t * (N-1)/N
    p_change = beta_t / N
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from .BaseNoise import BaseNoiseDistribution


class CategoricalNoise(BaseNoiseDistribution):
    """
    Categorical noise distribution for N-way discrete diffusion.

    Uses the DiffUCO formulation where:
    - beta_t = 2 * gamma_t
    - p(stay at same value) = 1 - beta_t + beta_t/N
    - p(change to specific different value) = beta_t/N
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize categorical noise distribution.

        Args:
            config: Configuration dictionary containing:
                - n_diffusion_steps: Number of diffusion steps
                - diff_schedule: Type of noise schedule
                - n_bernoulli_features: Number of categories (= n_nodes for TSP)
        """
        super().__init__(config)
        self.n_categories = config.get("n_bernoulli_features", config.get("n_nodes", 20))

    def _get_transition_probs(self, t_idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get transition probabilities for timestep t.

        Args:
            t_idx: Timestep index
            device: Device for tensors

        Returns:
            (p_stay, p_change) - probabilities for staying vs changing
        """
        gamma_t = self.get_gamma_t(t_idx)

        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(device)
        else:
            gamma_t = torch.tensor(gamma_t, device=device)

        # DiffUCO categorical formulation
        beta_t = 2 * gamma_t
        p_change = beta_t / self.n_categories
        p_stay = 1 - beta_t + p_change  # = 1 - beta_t * (N-1)/N

        # Numerical stability
        p_stay = torch.clamp(p_stay, min=1e-8, max=1-1e-8)
        p_change = torch.clamp(p_change, min=1e-8, max=1-1e-8)

        return p_stay, p_change

    def combine_losses(
        self,
        noise_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        T: float = 1.0
    ) -> torch.Tensor:
        """
        Combine different loss components.
        DiffUCO formulation: -T * L_entropy + T * L_noise + L_energy
        """
        return -T * entropy_loss + T * noise_loss + energy_loss

    def calculate_noise_distr_reward(
        self,
        noise_distr_step: torch.Tensor,
        entropy_reward: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the reward contribution from the noise distribution."""
        return -(noise_distr_step - entropy_reward)

    def get_log_p_T_0(
        self,
        X_prev: torch.Tensor,
        X_next: torch.Tensor,
        t_idx: int,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute log probability log p(X_next | X_prev) for categorical diffusion.

        Args:
            X_prev: Previous state (batch, n_nodes)
            X_next: Next state (batch, n_nodes)
            t_idx: Timestep index

        Returns:
            Log probability per sample (batch,)
        """
        if X_prev.dim() == 3:
            X_prev = X_prev.squeeze(-1)
        if X_next.dim() == 3:
            X_next = X_next.squeeze(-1)

        device = X_prev.device
        p_stay, p_change = self._get_transition_probs(t_idx, device)

        # p(X_next | X_prev) = p_stay if same, else p_change
        same_mask = (X_next.long() == X_prev.long()).float()
        log_p_per_node = same_mask * torch.log(p_stay) + (1 - same_mask) * torch.log(p_change)

        # Sum over all categories per node, then sum over nodes
        return log_p_per_node.sum(dim=-1)

    def sample_forward_diff_process(
        self,
        X_t_m1: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Generator]:
        """
        Sample from the forward diffusion process q(X_t | X_{t-1}).

        Transition probabilities:
            p(X_t = j | X_{t-1} = i) = p_stay     if j == i
                                     = p_change   if j != i
        Where:
            beta_t = 2 * gamma_t
            p_stay = 1 - beta_t + beta_t/N
            p_change = beta_t / N

        Implementation: With prob beta_t, sample uniformly from ALL N categories.
        This gives: P(stay) = (1-beta_t) + beta_t/N = p_stay ✓

        Args:
            X_t_m1: State at timestep t-1 (batch, n_nodes)
            t_idx: Target timestep index
            generator: Random generator

        Returns:
            Tuple of (X_t, log_probs, generator)
        """
        if X_t_m1.dim() == 3:
            X_t_m1 = X_t_m1.squeeze(-1)

        batch_size, n_nodes = X_t_m1.shape
        device = X_t_m1.device

        gamma_t = self.get_gamma_t(t_idx)
        if isinstance(gamma_t, torch.Tensor):
            gamma_t = gamma_t.to(device)
        else:
            gamma_t = torch.tensor(gamma_t, device=device)

        # DiffUCO formulation: beta_t = 2 * gamma_t
        beta_t = 2 * gamma_t
        beta_t = torch.clamp(beta_t, min=0.0, max=1.0)

        # With probability beta_t, "change" (sample uniformly from ALL N categories)
        # With probability (1-beta_t), "stay" at current value
        # This gives correct transition probabilities:
        #   P(stay) = (1-beta_t) + beta_t/N = p_stay
        #   P(change to j) = beta_t/N = p_change
        change_mask = torch.bernoulli(
            torch.full((batch_size, n_nodes), beta_t.item(), device=device),
            generator=generator
        ).bool()

        # Sample uniformly from all N categories
        random_categories = torch.randint(
            0, self.n_categories,
            (batch_size, n_nodes),
            device=device,
            generator=generator
        )

        # Apply: if change flag, use random category; else keep current
        X_t = torch.where(change_mask, random_categories, X_t_m1.long())

        # Compute log probability using exact transition probs
        p_stay, p_change = self._get_transition_probs(t_idx, device)
        same_mask = (X_t == X_t_m1.long()).float()
        log_p_per_node = same_mask * torch.log(p_stay) + (1 - same_mask) * torch.log(p_change)
        log_prob = log_p_per_node.sum(dim=-1)

        return X_t, log_prob, generator

    def sample_prior(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Sample from the prior distribution (uniform over categories).

        Args:
            shape: (batch_size, n_nodes)
            device: Device to create tensor on
            generator: Random generator

        Returns:
            Sampled category indices
        """
        batch_size, n_nodes = shape[:2]
        return torch.randint(
            0, self.n_categories,
            (batch_size, n_nodes),
            device=device,
            generator=generator
        )

    def calc_noise_step(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        t_idx: int,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse diffusion process (SDDS/PPO style).

        NOTE: This follows the SDDS/DiffUCO PPO approach where the model directly
        outputs a policy distribution over X_0. The reverse step samples from:
            p(X_{t-1}) ≈ (1 - gamma_{t-1}) * p_model(X_0) + gamma_{t-1} * uniform

        This is a "model-only" approach that treats the neural network output as
        the policy. X_t is NOT used in the posterior computation because in PPO
        we're learning a direct policy mapping from (coords, X_t, t) -> action.

        For a true DDPM posterior that depends on X_t, you would need:
            q(X_{t-1} | X_t, X_0) ∝ q(X_t | X_{t-1}) * q(X_{t-1} | X_0)

        Args:
            logits: Model output logits (batch, n_nodes, n_categories)
            X_t: Current state at timestep t (batch, n_nodes) - passed for interface
                 compatibility but not used in posterior (model-only approach)
            t_idx: Current timestep index (T-1 to 0, where T-1 is noisiest)
            generator: Random generator

        Returns:
            Tuple of (X_{t-1}, log_prob per sample)
        """
        batch_size = logits.shape[0]
        n_nodes = logits.shape[1]
        n_categories = logits.shape[2]
        device = logits.device

        if X_t.dim() == 3:
            X_t = X_t.squeeze(-1)

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Get predicted probabilities for X_0 (clean state)
        p_x0 = F.softmax(logits, dim=-1)  # (batch, n_nodes, n_categories)
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)

        if t_idx == 0:
            # At t=0, sample directly from the predicted distribution
            p_sample = p_x0
        else:
            # For t > 0, compute posterior blending with noise
            # Use gamma_{t-1} to get the noise level for X_{t-1}
            gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else 0.0

            if isinstance(gamma_t_m1, torch.Tensor):
                gamma_t_m1 = gamma_t_m1.to(device)
            else:
                gamma_t_m1 = torch.tensor(gamma_t_m1, device=device)

            gamma_t_m1 = torch.clamp(gamma_t_m1, min=0.0, max=1.0)

            # Posterior: blend predicted X_0 with uniform noise
            # p(X_{t-1} | X_t, X_0) ≈ (1 - gamma_{t-1}) * p(X_0) + gamma_{t-1} * uniform
            uniform = torch.ones_like(p_x0) / n_categories
            p_sample = (1 - gamma_t_m1) * p_x0 + gamma_t_m1 * uniform

        # Ensure valid probabilities
        p_sample = torch.clamp(p_sample, min=1e-8)
        p_sample = p_sample / p_sample.sum(dim=-1, keepdim=True)

        # Sample from posterior
        X_prev = torch.multinomial(
            p_sample.view(-1, n_categories),
            num_samples=1,
            generator=generator
        ).view(batch_size, n_nodes)

        # Compute log probability
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_nodes)
        node_idx = torch.arange(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        log_prob_per_node = torch.log(p_sample[batch_idx, node_idx, X_prev] + 1e-8)
        log_prob = log_prob_per_node.sum(dim=-1)

        return X_prev, log_prob

    def compute_action_log_prob(
        self,
        logits: torch.Tensor,
        X_t: torch.Tensor,
        action: torch.Tensor,
        t_idx: int
    ) -> torch.Tensor:
        """
        Compute log probability of a given action under the policy distribution.

        NOTE: Same as calc_noise_step, this uses the "model-only" SDDS/PPO approach.
        X_t is passed for interface compatibility but not used in the computation.
        The policy distribution is:
            p(action) = (1 - gamma_{t-1}) * softmax(logits) + gamma_{t-1} * uniform

        Args:
            logits: Model output logits (batch, n_nodes, n_categories)
            X_t: Current state at timestep t (batch, n_nodes) - not used (model-only)
            action: The action (X_{t-1}) to compute log prob for (batch, n_nodes)
            t_idx: Current timestep index

        Returns:
            Log probability per sample (batch,)
        """
        batch_size = logits.shape[0]
        n_nodes = logits.shape[1]
        n_categories = logits.shape[2]
        device = logits.device

        if action.dim() == 3:
            action = action.squeeze(-1)

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Get predicted probabilities for X_0
        p_x0 = F.softmax(logits, dim=-1)
        p_x0 = torch.clamp(p_x0, min=1e-8, max=1-1e-8)

        if t_idx == 0:
            p_sample = p_x0
        else:
            gamma_t_m1 = self.get_gamma_t(t_idx - 1) if t_idx > 0 else 0.0

            if isinstance(gamma_t_m1, torch.Tensor):
                gamma_t_m1 = gamma_t_m1.to(device)
            else:
                gamma_t_m1 = torch.tensor(gamma_t_m1, device=device)

            gamma_t_m1 = torch.clamp(gamma_t_m1, min=0.0, max=1.0)

            uniform = torch.ones_like(p_x0) / n_categories
            p_sample = (1 - gamma_t_m1) * p_x0 + gamma_t_m1 * uniform

        # Ensure valid probabilities
        p_sample = torch.clamp(p_sample, min=1e-8)
        p_sample = p_sample / p_sample.sum(dim=-1, keepdim=True)

        # Compute log probability of the given action
        action_long = action.long()
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_nodes)
        node_idx = torch.arange(n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        log_prob_per_node = torch.log(p_sample[batch_idx, node_idx, action_long] + 1e-8)

        return log_prob_per_node.sum(dim=-1)

    def get_entropy(
        self,
        logits: torch.Tensor,
        node_graph_idx: Optional[torch.Tensor] = None,
        n_graphs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute entropy of the predicted distribution.

        Args:
            logits: Model output logits (batch, n_nodes, n_categories)

        Returns:
            Entropy per sample (batch,)
        """
        logits = torch.clamp(logits, min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)

        # Entropy per node: -sum_i p_i log p_i
        entropy_per_node = -torch.sum(probs * torch.log(probs), dim=-1)
        entropy_per_node = torch.nan_to_num(entropy_per_node, nan=0.0)

        return entropy_per_node.sum(dim=-1)

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
            logits: Model output logits (batch, n_nodes, n_categories)
            X_prev: Previous state (batch, n_nodes)
            t_idx: Current timestep
            T: Temperature

        Returns:
            Tuple of (noise_loss, log_prob_per_node)
        """
        if X_prev.dim() == 3:
            X_prev = X_prev.squeeze(-1)

        device = X_prev.device
        p_stay, p_change = self._get_transition_probs(t_idx, device)

        # Get predicted probabilities
        logits = torch.clamp(logits, min=-50, max=50)
        p_next = F.softmax(logits, dim=-1)  # (batch, n_nodes, n_categories)

        # Create one-hot of previous state
        X_prev_long = X_prev.long()
        one_hot_prev = F.one_hot(X_prev_long, num_classes=self.n_categories).float()

        # Expected log probability under forward noise
        # E[log q(X_t)] = p_stay * log q(X_prev) + sum_{j != X_prev} p_change * log q(j)
        # = p_stay * log q(X_prev) + p_change * sum_{j != X_prev} log q(j)

        log_p_next = torch.log(p_next + 1e-8)

        # log q(X_prev) for each node
        log_q_prev = (log_p_next * one_hot_prev).sum(dim=-1)  # (batch, n_nodes)

        # Sum of log q(j) for all j
        sum_log_q = log_p_next.sum(dim=-1)  # (batch, n_nodes)

        # Sum of log q(j) for j != X_prev
        sum_log_q_other = sum_log_q - log_q_prev

        # Expected log prob = p_stay * log q(X_prev) + p_change * sum_{j != X_prev} log q(j)
        noise_per_node = p_stay * log_q_prev + p_change * sum_log_q_other
        noise_loss = T * noise_per_node.sum(dim=-1)

        return noise_loss, noise_per_node
