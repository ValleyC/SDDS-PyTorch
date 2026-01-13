"""
Learning rate scheduling utilities.
"""

import math
from typing import Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


def cos_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
    warmup_steps: int = 0
) -> float:
    """
    Cosine annealing learning rate schedule with optional warmup.

    Args:
        step: Current step
        total_steps: Total number of steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps

    # Cosine annealing
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            total_steps: Total training steps
            warmup_steps: Warmup steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch (for resuming)
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rate for current step."""
        step = self.last_epoch

        return [
            cos_schedule(
                step,
                self.total_steps,
                base_lr,
                self.min_lr,
                self.warmup_steps
            )
            for base_lr in self.base_lrs
        ]
