"""Profiling utilities for memory and performance analysis.

Used during local debugging to verify memory estimates, measure
throughput, and identify bottlenecks before shipping to cloud.
"""

import time
from contextlib import contextmanager
from typing import Optional, Dict

import torch


def get_gpu_memory_stats() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage in GB.

    Returns:
        Dictionary with memory stats, or None if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return None

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def print_gpu_memory(prefix: str = "") -> None:
    """Print current GPU memory usage."""
    stats = get_gpu_memory_stats()
    if stats is None:
        print(f"{prefix}CUDA not available")
        return

    print(
        f"{prefix}"
        f"GPU Mem: {stats['allocated_gb']:.2f}GB allocated, "
        f"{stats['reserved_gb']:.2f}GB reserved, "
        f"peak {stats['max_allocated_gb']:.2f}GB"
    )


def reset_gpu_memory_stats() -> None:
    """Reset peak GPU memory tracking counters."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def estimate_model_memory(num_params: int, dtype: torch.dtype = torch.float16) -> Dict[str, float]:
    """Estimate memory requirements for a model.

    Args:
        num_params: Total number of parameters.
        dtype: Parameter data type.

    Returns:
        Dictionary with memory estimates in GB.
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 2)

    model_gb = num_params * bytes_per_param / (1024 ** 3)

    # Adam optimizer states: 2 copies (momentum + variance) in fp32
    optimizer_gb = num_params * 4 * 2 / (1024 ** 3)

    # Gradients: same size as model
    gradient_gb = model_gb

    return {
        "model_weights_gb": model_gb,
        "optimizer_states_gb": optimizer_gb,
        "gradients_gb": gradient_gb,
        "total_gb": model_gb + optimizer_gb + gradient_gb,
    }


@contextmanager
def timer(name: str = "Operation"):
    """Context manager to time a code block.

    Usage:
        with timer("Forward pass"):
            output = model(input)

    Args:
        name: Label for the timing output.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")


def compute_throughput(
    num_tokens: int,
    elapsed_seconds: float,
    num_gpus: int = 1,
) -> Dict[str, float]:
    """Compute training throughput metrics.

    Args:
        num_tokens: Total tokens processed.
        elapsed_seconds: Wall-clock time in seconds.
        num_gpus: Number of GPUs used.

    Returns:
        Dictionary with throughput metrics.
    """
    tokens_per_second = num_tokens / elapsed_seconds
    tokens_per_gpu_per_second = tokens_per_second / num_gpus

    return {
        "tokens_per_second": tokens_per_second,
        "tokens_per_gpu_per_second": tokens_per_gpu_per_second,
        "samples_per_second": tokens_per_second,  # 1 token = 1 sample in LM
        "estimated_hours_per_billion_tokens": 1e9 / tokens_per_second / 3600,
    }
