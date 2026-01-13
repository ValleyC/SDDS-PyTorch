"""
TSP Dataset for SDDS-PyTorch.

Provides TSP instances for training and evaluation.
For RL training, we generate random instances (no labels needed).
For evaluation, instances can be loaded from files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any
import numpy as np
import os


class TSPRandomDataset(Dataset):
    """
    Generate random TSP instances on-the-fly.

    Coordinates are uniformly sampled from [0, 1] x [0, 1].
    This is suitable for RL training where no ground truth is needed.
    """

    def __init__(
        self,
        n_nodes: int = 20,
        n_instances: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize random TSP dataset.

        Args:
            n_nodes: Number of nodes per instance
            n_instances: Number of instances to generate
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.n_instances = n_instances
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Pre-generate coordinates for consistency
        self.coords = torch.rand(n_instances, n_nodes, 2)

    def __len__(self) -> int:
        return self.n_instances

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"coords": self.coords[idx]}


class TSPFileDataset(Dataset):
    """
    Load TSP instances from file.

    Expected file format (DIFUSCO/EDISCO compatible):
    Each line contains space-separated floats:
    x1 y1 x2 y2 ... xn yn [tour_indices if available]
    """

    def __init__(
        self,
        data_file: str,
        n_nodes: Optional[int] = None
    ):
        """
        Initialize TSP dataset from file.

        Args:
            data_file: Path to data file
            n_nodes: Number of nodes (inferred from file if None)
        """
        self.data_file = data_file
        self.coords = []
        self.tours = []

        self._load_data(data_file, n_nodes)

    def _load_data(self, data_file: str, n_nodes: Optional[int] = None):
        """Load data from file."""
        with open(data_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                values = [float(x) for x in parts]

                # Infer n_nodes from first line if not specified
                if n_nodes is None:
                    # Assume format: x1 y1 x2 y2 ... xn yn output idx1 idx2 ...
                    # Find where 'output' appears or use half the length
                    if 'output' in parts:
                        n_nodes = parts.index('output') // 2
                    else:
                        n_nodes = len(values) // 2

                # Extract coordinates
                coord_values = values[:n_nodes * 2]
                coords = torch.tensor(coord_values).reshape(n_nodes, 2)
                self.coords.append(coords)

                # Extract tour if available
                if len(values) > n_nodes * 2:
                    tour_values = [int(x) for x in values[n_nodes * 2:]]
                    self.tours.append(torch.tensor(tour_values))

        self.coords = torch.stack(self.coords)
        if self.tours:
            self.tours = torch.stack(self.tours)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {"coords": self.coords[idx]}
        if self.tours is not None and len(self.tours) > 0:
            result["tour"] = self.tours[idx]
        return result


class TSPOnlineDataset(Dataset):
    """
    Generate TSP instances on-the-fly during iteration.

    Each call to __getitem__ generates a fresh random instance.
    Useful for infinite training data.
    """

    def __init__(
        self,
        n_nodes: int = 20,
        epoch_size: int = 10000
    ):
        """
        Initialize online TSP dataset.

        Args:
            n_nodes: Number of nodes per instance
            epoch_size: Virtual size of dataset (for iteration)
        """
        self.n_nodes = n_nodes
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        coords = torch.rand(self.n_nodes, 2)
        return {"coords": coords}


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for TSP batches.

    Args:
        batch: List of dictionaries

    Returns:
        Batched dictionary
    """
    result = {}
    keys = batch[0].keys()

    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])

    return result


def get_tsp_dataloader(
    n_nodes: int = 20,
    batch_size: int = 32,
    n_instances: int = 10000,
    data_file: Optional[str] = None,
    online: bool = False,
    seed: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a TSP dataloader.

    Args:
        n_nodes: Number of nodes
        batch_size: Batch size
        n_instances: Number of instances (for random dataset)
        data_file: Path to data file (if loading from file)
        online: If True, generate instances on-the-fly
        seed: Random seed
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data

    Returns:
        DataLoader for TSP instances
    """
    if data_file is not None:
        dataset = TSPFileDataset(data_file, n_nodes)
    elif online:
        dataset = TSPOnlineDataset(n_nodes, n_instances)
    else:
        dataset = TSPRandomDataset(n_nodes, n_instances, seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
