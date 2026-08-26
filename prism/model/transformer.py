"""Transformer Block for Prism AI.

Each block consists of:
  1. Pre-norm → Grouped-Query Attention → Residual connection
  2. Pre-norm → SwiGLU FFN → Residual connection

Pre-normalization (normalizing BEFORE attention/FFN) is used instead of
post-normalization because it provides more stable training, especially
at scale. This is the standard for all modern LLMs (GPT-3+, LLaMA, etc.).

Gradient checkpointing is supported at the block level to trade compute
for memory — essential for fitting the 10B model in GPU memory.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from prism.model.normalization import RMSNorm
from prism.model.attention import GroupedQueryAttention
from prism.model.feedforward import SwiGLUFFN


class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of query heads.
        num_kv_heads: Number of key-value heads (GQA).
        head_dim: Dimension per attention head.
        intermediate_size: FFN intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        rms_norm_eps: Epsilon for RMSNorm.
        attention_dropout: Attention dropout rate.
        hidden_dropout: FFN dropout rate.
        layer_idx: Index of this layer (for debugging/logging).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Grouped-Query Attention with RoPE
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
        )

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # SwiGLU Feed-Forward Network
        self.feed_forward = SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through one transformer block.

        Pre-norm architecture:
            x = x + Attention(RMSNorm(x))
            x = x + FFN(RMSNorm(x))

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices for RoPE.
            past_key_value: Optional cached (K, V) from previous step.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of (output_hidden_states, optional_kv_cache).
        """
        # ── Attention sub-layer ─────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_cache = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # ── FFN sub-layer ───────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv_cache
