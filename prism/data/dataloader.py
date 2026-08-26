"""Distributed DataLoader for Prism AI training.

Provides a configured DataLoader that handles distributed training
concerns (each GPU sees a different subset of data) and optimizes
for throughput with prefetching and pinned memory.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from prism.data.dataset import PreTokenizedDataset
from prism.utils.distributed import get_world_size, get_rank


def create_train_dataloader(
    data_dir: str,
    max_seq_len: int = 4096,
    micro_batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create a training DataLoader with distributed sampling.

    In distributed training, each GPU gets a non-overlapping subset
    of the data via DistributedSampler. The sampler handles shuffling
    across epochs.

    Args:
        data_dir: Directory with pre-tokenized .bin shards.
        max_seq_len: Sequence length (must match preprocessing).
        micro_batch_size: Batch size per GPU per forward pass.
        num_workers: Number of data loading worker processes.
        pin_memory: Pin memory for faster CPU-to-GPU transfer.
        seed: Random seed for reproducible shuffling.

    Returns:
        Configured DataLoader instance.
    """
    dataset = PreTokenizedDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        split="train",
    )

    # Distributed sampler ensures each GPU sees different data
    world_size = get_world_size()
    rank = get_rank()

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if no distributed sampler
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batches for consistent batch sizes
        persistent_workers=num_workers > 0,
    )

    return dataloader


def create_val_dataloader(
    data_dir: str,
    max_seq_len: int = 4096,
    micro_batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a validation DataLoader.

    Validation data is NOT shuffled and uses a sequential sampler.
    In distributed mode, each GPU evaluates a subset and results
    are aggregated.

    Args:
        data_dir: Directory with pre-tokenized .bin shards.
        max_seq_len: Sequence length.
        micro_batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Configured DataLoader instance.
    """
    dataset = PreTokenizedDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        split="val",
    )

    world_size = get_world_size()
    rank = get_rank()

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return dataloader
