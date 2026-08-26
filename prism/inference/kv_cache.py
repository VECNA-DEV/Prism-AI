"""KV Cache for efficient autoregressive generation.

During generation, each new token only needs to attend to all
previous tokens. Without caching, we'd recompute K and V for ALL
tokens at every step — O(n²) total work. With KV cache, we compute
K and V only for the NEW token and concatenate with cached values,
reducing total work to O(n).

Memory cost: 2 × num_layers × batch × seq_len × num_kv_heads × head_dim × bytes_per_element
For the 10B model at seq_len=4096 in fp16:
  2 × 46 × 1 × 4096 × 8 × 128 × 2 = ~7.7 GB per batch element
"""

from typing import List, Optional, Tuple

import torch


class KVCache:
    """Key-Value cache for transformer autoregressive generation.

    Stores K and V tensors for each layer, growing as new tokens
    are generated. Supports both single-sequence and batched generation.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size to pre-allocate for.
        max_seq_len: Maximum sequence length.
        num_kv_heads: Number of KV heads per layer.
        head_dim: Dimension per attention head.
        dtype: Data type for cached tensors.
        device: Device for cached tensors.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Current sequence length in the cache
        self.seq_len = 0

        # Pre-allocate cache tensors: (batch, num_kv_heads, max_seq_len, head_dim)
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

        for _ in range(num_layers):
            self.k_cache.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )
            self.v_cache.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache for a specific layer and return full K, V.

        Args:
            layer_idx: Index of the transformer layer.
            k: New key tensor of shape (batch, num_kv_heads, new_seq_len, head_dim).
            v: New value tensor of same shape.

        Returns:
            Tuple of full (K, V) tensors including cached values.
        """
        new_seq_len = k.shape[2]

        # Write new K, V into the pre-allocated buffer
        self.k_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = k
        self.v_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = v

        # Return the full cached K, V up to current position
        total_len = self.seq_len + new_seq_len
        k_full = self.k_cache[layer_idx][:, :, :total_len, :]
        v_full = self.v_cache[layer_idx][:, :, :total_len, :]

        return k_full, v_full

    def advance(self, num_tokens: int = 1) -> None:
        """Advance the sequence position after processing tokens.

        Args:
            num_tokens: Number of new tokens that were processed.
        """
        self.seq_len += num_tokens

    def get_seq_len(self) -> int:
        """Get the current cached sequence length."""
        return self.seq_len

    def reset(self) -> None:
        """Clear the cache for a new generation sequence."""
        self.seq_len = 0
        for i in range(self.num_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()

    def get_past_key_values(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get the cached K, V tensors for all layers.

        Returns:
            List of (K, V) tuples, one per layer, or None if cache is empty.
        """
        if self.seq_len == 0:
            return None

        return [
            (
                self.k_cache[i][:, :, :self.seq_len, :],
                self.v_cache[i][:, :, :self.seq_len, :],
            )
            for i in range(self.num_layers)
        ]

    def memory_usage_gb(self) -> float:
        """Calculate current memory usage of the cache in GB."""
        bytes_per_element = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(self.dtype, 2)

        total_elements = (
            2  # K and V
            * self.num_layers
            * self.max_batch_size
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim
        )

        return total_elements * bytes_per_element / (1024 ** 3)
