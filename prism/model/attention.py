"""Grouped-Query Attention with Rotary Position Embeddings and FlashAttention-2.

This module implements the attention mechanism used in Prism AI:
  1. Grouped-Query Attention (GQA) — fewer KV heads than Q heads, reducing
     KV-cache memory by 4× (with our 32/8 config) while maintaining quality.
  2. Rotary Position Embeddings (RoPE) — encodes position information directly
     into Q and K vectors via rotation. Supports extrapolation to longer
     sequences and requires no learned parameters.
  3. FlashAttention-2 — IO-aware exact attention with O(N) memory and 2-4×
     speedup. Falls back to standard SDPA when unavailable.

References:
  - GQA:  Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
          Models from Multi-Head Checkpoints" (2023)
  - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
          Embedding" (2021)
  - FlashAttention-2: Dao, "FlashAttention-2: Faster Attention with Better
          Parallelism and Work Partitioning" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# FlashAttention-2 is optional — we fall back to standard attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ── Rotary Position Embeddings ──────────────────────────────────────────


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for Rotary Position Embeddings.

    The frequency bands are computed as:
        freq_i = 1 / (theta^(2i / head_dim))  for i in [0, head_dim/2)

    Then for each position p, we compute cos(p * freq) and sin(p * freq).
    These are duplicated to cover the full head_dim.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency (default: 10000.0).
        device: Target device for tensors.

    Returns:
        Tuple of (cos_table, sin_table), each of shape (max_seq_len, head_dim).
    """
    # Frequency bands: shape (head_dim / 2,)
    freq_exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (freq_exponents / head_dim))

    # Position indices: shape (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # Outer product: shape (max_seq_len, head_dim / 2)
    angles = torch.outer(positions, freqs)

    # Duplicate to cover full head_dim: shape (max_seq_len, head_dim)
    # This matches the "half-rotation" convention used by LLaMA/Mistral
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)

    return cos_table, sin_table


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to the first, negated.

    For input [..., x1, x2] where x1 and x2 are the first and second halves:
        output = [..., -x2, x1]

    This implements the rotation matrix component of RoPE.

    Args:
        x: Tensor of shape (..., head_dim).

    Returns:
        Rotated tensor of same shape.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_embeddings(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embeddings to a tensor.

    RoPE encodes position by rotating pairs of dimensions in the
    query/key vectors. The rotation angle is proportional to the
    position and inversely proportional to the dimension index.

    The formula is:
        RoPE(x, pos) = x * cos(pos * freq) + rotate_half(x) * sin(pos * freq)

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine table, broadcastable to x's shape.
        sin: Sine table, broadcastable to x's shape.

    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    return (x * cos) + (rotate_half(x) * sin)


# ── Grouped-Query Attention ─────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE and FlashAttention-2.

    In GQA, multiple query heads share a single key-value head group.
    With 32 query heads and 8 KV heads (our config), each KV head is
    shared by 4 query heads. This reduces KV-cache memory by 4× during
    inference with negligible quality loss.

    During training, we use FlashAttention-2 when available for the
    2-4× speedup and O(N) memory. FlashAttention natively supports GQA
    — we pass the un-expanded KV tensors directly.

    During inference (with KV cache), we fall back to standard scaled
    dot-product attention since FlashAttention doesn't support incremental
    decoding.

    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of query heads (e.g., 32).
        num_kv_heads: Number of key-value heads (e.g., 8).
        head_dim: Dimension per head (hidden_size // num_attention_heads).
        max_seq_len: Maximum sequence length for RoPE precomputation.
        rope_theta: RoPE base frequency.
        attention_dropout: Dropout rate on attention weights (training only).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_kv_heads
        self.attention_dropout = attention_dropout
        self.scaling = 1.0 / math.sqrt(head_dim)

        # ── Linear Projections (no bias, following LLaMA convention) ──
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # ── Precompute RoPE frequency tables ──
        cos_table, sin_table = precompute_rope_frequencies(
            head_dim, max_seq_len, rope_theta
        )
        self.register_buffer("rope_cos", cos_table, persistent=False)
        self.register_buffer("rope_sin", sin_table, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match number of query heads for standard attention.

        Each KV head is repeated `num_kv_groups` times.

        Args:
            x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim).

        Returns:
            Tensor of shape (batch, num_attention_heads, seq_len, head_dim).
        """
        if self.num_kv_groups == 1:
            return x

        batch, num_kv_heads, seq_len, head_dim = x.shape
        # Insert a repeat dimension and expand
        x = x[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim
        )
        return x.reshape(batch, self.num_attention_heads, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through Grouped-Query Attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            attention_mask: Optional mask of shape (batch, 1, seq_len, kv_seq_len).
                           Contains 0 for valid positions and -inf for masked.
            position_ids: Position indices of shape (batch, seq_len).
                         Auto-computed if None.
            past_key_value: Optional cached (K, V) tuple from previous step.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of:
              - Output tensor of shape (batch, seq_len, hidden_size)
              - Optional (K, V) cache tuple for next step
        """
        bsz, seq_len, _ = hidden_states.shape

        # ── Project to Q, K, V ──────────────────────────────────────
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── Compute position IDs if not provided ────────────────────
        if position_ids is None:
            past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0).expand(bsz, -1)

        # ── Apply RoPE to Q and K ───────────────────────────────────
        # Index the precomputed tables: (batch, seq_len, head_dim)
        cos = self.rope_cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = self.rope_sin[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)

        q = apply_rotary_embeddings(q, cos, sin)
        k = apply_rotary_embeddings(k, cos, sin)

        # ── KV Cache concatenation ──────────────────────────────────
        if past_key_value is not None:
            # Append current K, V to cached K, V
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # ── Attention computation ───────────────────────────────────
        use_flash_attn = (
            FLASH_ATTN_AVAILABLE
            and not use_cache
            and attention_mask is None
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash_attn:
            # FlashAttention-2 path: used during training and prefill
            # FlashAttention natively supports GQA — no need to expand KV
            # Expected shapes: (batch, seq_len, num_heads, head_dim)
            q_fa = q.transpose(1, 2)   # (bsz, seq_len, num_q_heads, head_dim)
            k_fa = k.transpose(1, 2)   # (bsz, kv_seq_len, num_kv_heads, head_dim)
            v_fa = v.transpose(1, 2)   # (bsz, kv_seq_len, num_kv_heads, head_dim)

            attn_output = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )
            # (bsz, seq_len, num_q_heads, head_dim) -> (bsz, seq_len, hidden_size)
            attn_output = attn_output.reshape(bsz, seq_len, -1)
        else:
            # Standard attention path: used during inference or when FA unavailable
            # Expand KV heads for standard matmul
            k_expanded = self._repeat_kv(k)
            v_expanded = self._repeat_kv(v)

            # Scaled dot-product: (bsz, num_heads, seq_len, kv_seq_len)
            attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scaling

            # Apply causal mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                # Build causal mask: upper triangle = -inf
                causal_mask = torch.triu(
                    torch.full(
                        (seq_len, kv_seq_len),
                        float("-inf"),
                        device=q.device,
                        dtype=q.dtype,
                    ),
                    diagonal=kv_seq_len - seq_len + 1,
                )
                attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            # Softmax in float32 for numerical stability
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

            # Attention dropout (training only)
            if self.training and self.attention_dropout > 0.0:
                attn_weights = F.dropout(
                    attn_weights, p=self.attention_dropout, training=True
                )

            # Weighted sum of values
            attn_output = torch.matmul(attn_weights, v_expanded)

            # (bsz, num_heads, seq_len, head_dim) -> (bsz, seq_len, hidden_size)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)

        # ── Output projection ───────────────────────────────────────
        output = self.o_proj(attn_output)

        return output, new_cache
