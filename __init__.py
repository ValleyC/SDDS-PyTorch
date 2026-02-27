"""
Diffusion module for CVRP/TSP partitioning.

DiffUCO-faithful implementation of categorical diffusion with PPO training.
"""

from .noise_schedule import CategoricalNoiseSchedule
from .step_model import (
    DiffusionStepModel,
    ReluMLP,
    ProbMLP,
    ValueMLP,
    LinearMessagePassingLayer,
    EncodeProcessDecode,
    get_sinusoidal_positional_encoding,
    scatter_sum,
)
from .trajectory import (
    TrajectoryBuffer,
    collect_trajectory,
    collect_trajectories_batch,
)
from .ppo_trainer import (
    MovingAverage,
    compute_gae,
    normalize_advantages,
    PPOTrainer,
)
from .tsp_energy import (
    TSPEnergy,
    create_tsp_energy_fn,
    compute_tour_length_only,
    check_tour_validity,
)

__all__ = [
    'CategoricalNoiseSchedule',
    'DiffusionStepModel',
    'ReluMLP',
    'ProbMLP',
    'ValueMLP',
    'LinearMessagePassingLayer',
    'EncodeProcessDecode',
    'get_sinusoidal_positional_encoding',
    'scatter_sum',
    'TrajectoryBuffer',
    'collect_trajectory',
    'collect_trajectories_batch',
    'MovingAverage',
    'compute_gae',
    'normalize_advantages',
    'PPOTrainer',
    'TSPEnergy',
    'create_tsp_energy_fn',
    'compute_tour_length_only',
    'check_tour_validity',
]
