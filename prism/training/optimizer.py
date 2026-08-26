"""Optimizer and Learning Rate Scheduler for Prism AI.

Uses AdamW with decoupled weight decay and a cosine annealing
schedule with linear warmup — the standard combination for
large-scale transformer training.

Key hyperparameters for the 10B model:
  - Peak LR: 1.5e-4
  - Min LR: 1.5e-5 (10% of peak)
  - Warmup steps: 2000
  - Weight decay: 0.1
  - Beta1: 0.9, Beta2: 0.95
"""

import math
from typing import Iterable, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model_params: Iterable[Tuple[str, torch.nn.Parameter]],
    learning_rate: float = 1.5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Create an AdamW optimizer with proper parameter groups.

    Weight decay is NOT applied to:
      - Bias parameters
      - LayerNorm / RMSNorm weights
      - Embedding weights

    This follows the standard convention from GPT-2 / LLaMA training.

    Args:
        model_params: Model parameters (from model.named_parameters()).
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam momentum parameters.
        eps: Adam epsilon for numerical stability.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model_params:
        if not param.requires_grad:
            continue

        # Don't decay: biases, norms, embeddings
        if param.dim() == 1:
            # This catches: RMSNorm weights, any biases
            no_decay_params.append(param)
        elif "tok_embeddings" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer parameter groups:")
    print(f"  Decay:    {num_decay:>15,} params (wd={weight_decay})")
    print(f"  No-decay: {num_no_decay:>15,} params (wd=0.0)")

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


def create_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine annealing LR schedule with linear warmup.

    Schedule:
      1. Linear warmup from 0 to peak_lr over `warmup_steps`
      2. Cosine decay from peak_lr to min_lr over remaining steps

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR (default: 0.1).

    Returns:
        LambdaLR scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Phase 2: Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Interpolate between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
