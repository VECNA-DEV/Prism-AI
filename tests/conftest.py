"""PyTest fixtures for Prism AI test suite."""

import pytest
import torch
from prism.model.config import PrismConfig


@pytest.fixture
def tiny_config():
    """Returns a lightweight PrismConfig for unit testing."""
    return PrismConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_seq_len=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        gradient_checkpointing=False,
    )
