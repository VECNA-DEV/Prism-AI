"""Training metrics computation for Prism AI.

Tracks and computes key training metrics:
  - Loss (cross-entropy, smoothed)
  - Perplexity (exp of loss)
  - Throughput (tokens/second, samples/second)
  - Gradient norm
  - Learning rate
"""

import time
from collections import deque
from typing import Dict

import torch

from prism.utils.distributed import all_reduce_scalar, get_world_size


class MetricsTracker:
    """Tracks and smooths training metrics over a rolling window.

    Args:
        window_size: Number of steps for the rolling average.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Rolling windows for smoothing
        self._loss_window: deque = deque(maxlen=window_size)
        self._step_times: deque = deque(maxlen=window_size)
        self._tokens_per_step: deque = deque(maxlen=window_size)

        # Cumulative tracking
        self.total_tokens: int = 0
        self.total_steps: int = 0
        self.start_time: float = time.time()
        self._last_step_time: float = time.time()

    def update(
        self,
        loss: float,
        batch_tokens: int,
        grad_norm: float = 0.0,
    ) -> Dict[str, float]:
        """Record metrics for one training step.

        Args:
            loss: Training loss for this step.
            batch_tokens: Number of tokens in this batch (across all GPUs).
            grad_norm: Gradient norm (if available).

        Returns:
            Dict with computed metrics.
        """
        # Time tracking
        now = time.time()
        step_time = now - self._last_step_time
        self._last_step_time = now

        # Update windows
        self._loss_window.append(loss)
        self._step_times.append(step_time)
        self._tokens_per_step.append(batch_tokens)

        # Update cumulative
        self.total_tokens += batch_tokens
        self.total_steps += 1

        # Compute metrics
        avg_loss = sum(self._loss_window) / len(self._loss_window)
        avg_step_time = sum(self._step_times) / len(self._step_times)

        # Tokens per second (global, across all GPUs)
        tokens_per_second = batch_tokens / step_time if step_time > 0 else 0

        # Smoothed tokens per second
        avg_tps = (
            sum(self._tokens_per_step) / sum(self._step_times)
            if self._step_times else 0
        )

        # Perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        # Clamp to avoid inf
        perplexity = min(perplexity, 1e8)

        # Elapsed time
        elapsed = now - self.start_time

        return {
            "loss": loss,
            "loss_smoothed": avg_loss,
            "perplexity": perplexity,
            "grad_norm": grad_norm,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_smoothed": avg_tps,
            "step_time": step_time,
            "step_time_smoothed": avg_step_time,
            "total_tokens": self.total_tokens,
            "total_steps": self.total_steps,
            "elapsed_hours": elapsed / 3600,
        }

    def estimate_remaining_time(self, total_tokens_target: int) -> float:
        """Estimate remaining training time in hours.

        Args:
            total_tokens_target: Target number of tokens to train on.

        Returns:
            Estimated remaining hours.
        """
        if not self._step_times or self.total_tokens == 0:
            return float("inf")

        avg_tps = sum(self._tokens_per_step) / sum(self._step_times)
        remaining_tokens = total_tokens_target - self.total_tokens

        if avg_tps <= 0 or remaining_tokens <= 0:
            return 0.0

        return remaining_tokens / avg_tps / 3600

    def get_summary(self) -> Dict[str, float]:
        """Get a summary of training progress.

        Returns:
            Dict with overall training statistics.
        """
        elapsed = time.time() - self.start_time
        return {
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_tokens_billions": self.total_tokens / 1e9,
            "elapsed_hours": elapsed / 3600,
            "avg_tokens_per_second": self.total_tokens / elapsed if elapsed > 0 else 0,
        }
