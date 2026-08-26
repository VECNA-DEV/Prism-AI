"""SwiGLU Feed-Forward Network for Prism AI.

SwiGLU is a gated variant of the standard FFN that uses the SiLU (Swish)
activation as the gating function. It consistently outperforms standard
GELU/ReLU FFNs by 1-2% on downstream benchmarks.

Architecture:
    FFN(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

Where:
    - gate_proj: hidden_size → intermediate_size
    - up_proj:   hidden_size → intermediate_size
    - down_proj: intermediate_size → hidden_size
    - ⊙ denotes elementwise multiplication

Note: SwiGLU uses 3 weight matrices instead of 2 (compared to standard FFN),
so intermediate_size is set to compensate (~8/3 × hidden_size) to maintain
roughly the same parameter count.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
           https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU-gated Feed-Forward Network.

    Args:
        hidden_size: Model hidden dimension (e.g., 4096).
        intermediate_size: FFN intermediate dimension (e.g., 14336).
        hidden_dropout: Dropout rate on FFN output (default: 0.0).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()

        # Gate projection: determines how much of each intermediate dim to use
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Up projection: computes the values to be gated
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Down projection: maps back to model dimension
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.hidden_dropout = hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU FFN.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).
        """
        # SwiGLU: SiLU(gate) ⊙ up, then project down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up

        output = self.down_proj(intermediate)

        if self.training and self.hidden_dropout > 0.0:
            output = F.dropout(output, p=self.hidden_dropout, training=True)

        return output
