from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make common training operations reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside each DataLoader worker.

    PyTorch gives each worker a deterministic base seed when a generator is
    supplied to the DataLoader. We map that seed to NumPy/Python so random data
    augmentation is reproducible across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for DataLoader shuffling and workers."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
