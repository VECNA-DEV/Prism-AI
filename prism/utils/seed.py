"""Reproducibility utilities for Prism AI.

Sets seeds across all random number generators to ensure
deterministic behavior during debugging and validation.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all sources for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def set_seed_for_training(seed: int = 42, rank: int = 0) -> None:
    """Set seed with rank offset for distributed training.

    Each rank gets a unique but reproducible seed so that data
    sampling differs across ranks while remaining deterministic.

    Args:
        seed: Base seed value.
        rank: Process rank in distributed training.
    """
    set_seed(seed + rank)
