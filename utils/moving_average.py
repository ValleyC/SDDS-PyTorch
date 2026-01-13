"""
Moving average utilities for baseline computation in RL.
"""

import torch
from typing import Optional


class ExponentialMovingAverage:
    """
    Exponential moving average for baseline computation.
    """

    def __init__(self, alpha: float = 0.99, initial_value: Optional[float] = None):
        """
        Initialize EMA.

        Args:
            alpha: Smoothing factor (higher = more smoothing)
            initial_value: Initial value for the average
        """
        self.alpha = alpha
        self.value = initial_value
        self.initialized = initial_value is not None

    def update(self, new_value: torch.Tensor) -> torch.Tensor:
        """
        Update the moving average.

        Args:
            new_value: New value to incorporate

        Returns:
            Updated moving average
        """
        if isinstance(new_value, torch.Tensor):
            new_value = new_value.mean().item()

        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value

        return self.value

    def get_value(self) -> float:
        """Get current moving average value."""
        return self.value if self.initialized else 0.0


class RunningMeanStd:
    """
    Running mean and standard deviation for reward normalization.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        """
        Initialize running statistics.

        Args:
            shape: Shape of the values
            epsilon: Small constant for numerical stability
        """
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, batch: torch.Tensor):
        """
        Update running statistics with a batch of values.

        Args:
            batch: Batch of values
        """
        batch = batch.view(-1)
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = batch.numel()

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize values using running statistics.

        Args:
            x: Values to normalize

        Returns:
            Normalized values
        """
        return (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)
