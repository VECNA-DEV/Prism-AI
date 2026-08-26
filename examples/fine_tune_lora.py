"""Example: Parameter-Efficient Fine-Tuning (LoRA) for Prism AI (10B).

Demonstrates attaching Low-Rank Adaptation (LoRA) adapters to
attention projection matrices ($W_q, W_v$) to fine-tune the 10B
model on domain-specific code datasets with minimal VRAM.

Usage:
    python examples/fine_tune_lora.py --r 16 --alpha 32
"""

import math
import torch
import torch.nn as nn
from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for PyTorch Linear layers."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.scaling = lora_alpha / r

        # Freeze base linear weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Low-rank matrices A and B
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize A with He uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model: PrismForCausalLM, r: int = 16, alpha: float = 32.0):
    """Replace attention Q and V linear projections with LoRA layers."""
    lora_params = 0
    total_params = 0

    for block in model.layers:
        block.attention.q_proj = LoRALinear(block.attention.q_proj, r=r, lora_alpha=alpha)
        block.attention.v_proj = LoRALinear(block.attention.v_proj, r=r, lora_alpha=alpha)

    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            lora_params += param.numel()
        else:
            param.requires_grad = False

    print(f"LoRA Injection Summary:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {lora_params:,} ({lora_params / total_params * 100:.3f}%)")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Prism AI (10B) — LoRA Adapter Initialization")
    print("=" * 60)
    config = PrismConfig()
    # Micro config for test demonstration
    config.num_layers = 4
    model = PrismForCausalLM(config)
    model = apply_lora_to_model(model, r=16, alpha=32.0)
    print("LoRA adapter architecture ready for supervised fine-tuning.")
