"""RMSNorm — Root Mean Square Layer Normalization.

RMSNorm is computationally simpler and faster than standard LayerNorm
because it skips the mean-centering step. It normalizes by the root
mean square of activations, then scales by a learnable weight.

Used in: LLaMA, Mistral, Gemma, and other modern transformer architectures.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
           https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor by its RMS value, then applies a
    learned elementwise scaling. The computation is performed in
    float32 for numerical stability, regardless of input dtype.

    Args:
        hidden_size: Size of the last dimension of input tensors.
        eps: Small constant for numerical stability (default: 1e-6).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization in float32.

        Args:
            x: Input tensor of any shape with last dim = hidden_size.

        Returns:
            Normalized tensor (still in float32).
        """
        # x.pow(2).mean(-1) computes the mean of squared values per position
        # rsqrt is 1/sqrt, giving us 1/RMS
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized and scaled tensor of same shape and dtype as input.
        """
        # Upcast to float32 for precision, normalize, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
