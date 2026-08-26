"""Unit tests for GroupedQueryAttention, RoPE, and KV caching."""

import torch
from prism.model.attention import GroupedQueryAttention, precompute_rope_frequencies


def test_rope_frequencies():
    cos, sin = precompute_rope_frequencies(128, 4096)
    assert cos.shape == (4096, 128)
    assert sin.shape == (4096, 128)
    assert torch.allclose(cos[0], torch.ones(128))
    assert torch.allclose(sin[0], torch.zeros(128))


def test_gqa_forward_and_cache(tiny_config):
    attn = GroupedQueryAttention(
        tiny_config.hidden_size,
        tiny_config.num_attention_heads,
        tiny_config.num_kv_heads,
        tiny_config.head_dim,
        tiny_config.max_seq_len,
        tiny_config.rope_theta,
    )
    x = torch.randn(1, 16, tiny_config.hidden_size)
    out, cache = attn(x, use_cache=True)
    assert out.shape == (1, 16, tiny_config.hidden_size)
    assert cache is not None
    k, v = cache
    assert k.shape == (1, tiny_config.num_kv_heads, 16, tiny_config.head_dim)

    # Next single token
    next_x = torch.randn(1, 1, tiny_config.hidden_size)
    out_next, cache_next = attn(next_x, past_key_value=cache, use_cache=True)
    assert out_next.shape == (1, 1, tiny_config.hidden_size)
    k2, v2 = cache_next
    assert k2.shape[2] == 17  # 16 + 1
