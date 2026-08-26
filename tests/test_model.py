"""Unit tests for PrismForCausalLM and TransformerBlock."""

import torch
from prism.model.model import PrismForCausalLM
from prism.model.normalization import RMSNorm
from prism.model.feedforward import SwiGLUFFN


def test_rmsnorm_forward_backward():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_swiglu_forward_backward():
    ffn = SwiGLUFFN(64, 128)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = ffn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_causal_lm_forward_loss(tiny_config):
    model = PrismForCausalLM(tiny_config)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 32))
    outputs = model(input_ids=tokens, labels=tokens)
    assert outputs["logits"].shape == (2, 32, tiny_config.vocab_size)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0
    outputs["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
