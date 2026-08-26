"""Distributed training utilities for Prism AI.

Provides helper functions for DeepSpeed distributed training setup,
rank/world-size queries, and device management.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed() -> None:
    """Initialize the distributed process group.

    This should be called once at the start of training.
    DeepSpeed's `deepspeed.init_distributed()` typically handles
    this, but we provide a manual fallback.
    """
    if not dist.is_initialized():
        # DeepSpeed sets these environment variables
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get the rank of the current process (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node).

    DeepSpeed sets LOCAL_RANK in the environment.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return get_rank() == 0


def get_device() -> torch.device:
    """Get the device for the current process.

    Returns the CUDA device corresponding to the local rank,
    or CPU if CUDA is not available.
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def barrier() -> None:
    """Synchronize all processes.

    This is a no-op if not in distributed mode.
    """
    if dist.is_initialized():
        dist.barrier()


def print_rank0(message: str) -> None:
    """Print a message only on rank 0."""
    if is_main_process():
        print(message, flush=True)


def all_reduce_scalar(value: float, op: str = "mean") -> float:
    """All-reduce a scalar value across processes.

    Args:
        value: Scalar value to reduce.
        op: Reduction operation — "mean" or "sum".

    Returns:
        Reduced scalar value.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return value

    tensor = torch.tensor(value, device=get_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if op == "mean":
        tensor /= get_world_size()

    return tensor.item()
