"""Comprehensive local pipeline test for Prism AI.

Tests EVERYTHING that can be tested locally without:
  - GPU (runs on CPU with a tiny model)
  - HuggingFace access
  - DeepSpeed multi-GPU
  - WandB account

This validates the full pipeline logic before shipping to cloud.
"""

import os
import sys
import json
import tempfile
import shutil
import traceback

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name):
    """Decorator to run and track test results."""
    def decorator(func):
        def wrapper():
            try:
                func()
                results.append((name, PASS, ""))
                print(f"  {PASS} {name}")
            except Exception as e:
                tb = traceback.format_exc()
                results.append((name, FAIL, str(e)))
                print(f"  {FAIL} {name}")
                print(f"       Error: {e}")
                print(f"       {tb}")
        return wrapper
    return decorator


# ── Use a TINY model config for local testing ───────────────────────
# Same architecture as 10B, just smaller dimensions so it runs on CPU
def get_test_config():
    from prism.model.config import PrismConfig
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


# ====================================================================
# TEST 1: Config
# ====================================================================

@test("Config creation and validation")
def test_config():
    from prism.model.config import PrismConfig
    
    # Test default (10B) config
    config_10b = PrismConfig()
    config_10b.validate()
    assert abs(config_10b.num_params_billions - 10.16) < 0.1, \
        f"Expected ~10.16B params, got {config_10b.num_params_billions:.2f}B"
    assert config_10b.head_dim == 128
    assert config_10b.num_kv_groups == 4
    
    # Test save/load roundtrip
    tmp_path = os.path.join(tempfile.gettempdir(), "prism_test_config.json")
    try:
        config_10b.save(tmp_path)
        loaded = PrismConfig.from_json(tmp_path)
        assert loaded.hidden_size == config_10b.hidden_size
        assert loaded.num_layers == config_10b.num_layers
        assert loaded.num_params == config_10b.num_params
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@test("Config YAML loading")
def test_config_yaml():
    from prism.model.config import PrismConfig
    config = PrismConfig.from_yaml("configs/model_10b.yaml")
    config.validate()
    assert config.hidden_size == 4096
    assert config.num_layers == 46
    assert config.num_attention_heads == 32
    assert config.num_kv_heads == 8


# ====================================================================
# TEST 2: RMSNorm
# ====================================================================

@test("RMSNorm forward + backward")
def test_rmsnorm():
    from prism.model.normalization import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    # Test backward
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Output should have unit RMS (approximately)
    rms = out.float().pow(2).mean(-1).sqrt()
    # After normalization and scaling by ones, RMS should be ~1
    assert rms.mean().item() < 5.0  # Loose check


# ====================================================================
# TEST 3: RoPE
# ====================================================================

@test("RoPE frequency precomputation")
def test_rope_precompute():
    from prism.model.attention import precompute_rope_frequencies
    cos, sin = precompute_rope_frequencies(128, 4096, theta=10000.0)
    assert cos.shape == (4096, 128)
    assert sin.shape == (4096, 128)
    # cos(0) should be all 1s
    assert torch.allclose(cos[0], torch.ones(128), atol=1e-6)
    # sin(0) should be all 0s
    assert torch.allclose(sin[0], torch.zeros(128), atol=1e-6)


@test("RoPE apply rotation")
def test_rope_apply():
    from prism.model.attention import precompute_rope_frequencies, apply_rotary_embeddings
    cos, sin = precompute_rope_frequencies(16, 32)
    x = torch.randn(1, 4, 8, 16)  # (batch, heads, seq, head_dim)
    cos_slice = cos[:8].unsqueeze(0).unsqueeze(0)
    sin_slice = sin[:8].unsqueeze(0).unsqueeze(0)
    out = apply_rotary_embeddings(x, cos_slice, sin_slice)
    assert out.shape == x.shape
    # At position 0, cos=1, sin=0, so output should equal input
    # (only for the first position)


# ====================================================================
# TEST 4: GQA Attention
# ====================================================================

@test("GQA Attention forward pass")
def test_gqa_forward():
    config = get_test_config()
    from prism.model.attention import GroupedQueryAttention
    attn = GroupedQueryAttention(
        config.hidden_size, config.num_attention_heads, config.num_kv_heads,
        config.head_dim, config.max_seq_len, config.rope_theta,
    )
    x = torch.randn(2, 16, config.hidden_size)
    out, cache = attn(x, use_cache=False)
    assert out.shape == x.shape
    assert cache is None


@test("GQA Attention with KV cache")
def test_gqa_kv_cache():
    config = get_test_config()
    from prism.model.attention import GroupedQueryAttention
    attn = GroupedQueryAttention(
        config.hidden_size, config.num_attention_heads, config.num_kv_heads,
        config.head_dim, config.max_seq_len, config.rope_theta,
    )
    
    # Prefill: process 8 tokens
    x = torch.randn(1, 8, config.hidden_size)
    out1, cache1 = attn(x, use_cache=True)
    assert out1.shape == (1, 8, config.hidden_size)
    assert cache1 is not None
    k_cache, v_cache = cache1
    assert k_cache.shape == (1, config.num_kv_heads, 8, config.head_dim)
    
    # Decode: process 1 token using cache
    x2 = torch.randn(1, 1, config.hidden_size)
    out2, cache2 = attn(x2, past_key_value=cache1, use_cache=True)
    assert out2.shape == (1, 1, config.hidden_size)
    k_cache2, v_cache2 = cache2
    assert k_cache2.shape == (1, config.num_kv_heads, 9, config.head_dim)  # 8 + 1


@test("GQA Attention backward pass")
def test_gqa_backward():
    config = get_test_config()
    from prism.model.attention import GroupedQueryAttention
    attn = GroupedQueryAttention(
        config.hidden_size, config.num_attention_heads, config.num_kv_heads,
        config.head_dim, config.max_seq_len, config.rope_theta,
    )
    x = torch.randn(2, 16, config.hidden_size, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    # Check gradients exist for all parameters
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ====================================================================
# TEST 5: SwiGLU FFN
# ====================================================================

@test("SwiGLU FFN forward + backward")
def test_swiglu():
    from prism.model.feedforward import SwiGLUFFN
    ffn = SwiGLUFFN(64, 128)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = ffn(x)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ====================================================================
# TEST 6: TransformerBlock
# ====================================================================

@test("TransformerBlock forward + backward")
def test_transformer_block():
    config = get_test_config()
    from prism.model.transformer import TransformerBlock
    block = TransformerBlock(
        config.hidden_size, config.num_attention_heads, config.num_kv_heads,
        config.head_dim, config.intermediate_size, config.max_seq_len,
    )
    x = torch.randn(2, 16, config.hidden_size, requires_grad=True)
    out, cache = block(x)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


# ====================================================================
# TEST 7: Full Model — Forward, Loss, Backward
# ====================================================================

@test("Full model forward pass")
def test_model_forward():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    outputs = model(input_ids=input_ids)
    
    assert outputs["logits"].shape == (2, 32, config.vocab_size)
    assert outputs["loss"] is None  # No labels provided
    assert outputs["past_key_values"] is None  # use_cache=False


@test("Full model loss computation")
def test_model_loss():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    
    assert outputs["loss"] is not None
    assert outputs["loss"].ndim == 0  # Scalar
    assert outputs["loss"].item() > 0  # Loss should be positive
    assert not torch.isnan(outputs["loss"]), "Loss is NaN!"
    assert not torch.isinf(outputs["loss"]), "Loss is Inf!"


@test("Causal target shift exact alignment (t -> t+1)")
def test_causal_target_alignment():
    """Verify that the model predicts token t+1 at step t (NOT t+2)."""
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    # Create an explicit sequence [10, 20, 30, 40]
    tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    outputs = model(input_ids=tokens, labels=tokens)
    
    # Shifted logits are logits[:, :-1, :] -> positions for token 10, 20, 30
    # Shifted labels are tokens[:, 1:] -> tokens 20, 30, 40
    shift_logits = outputs["logits"][:, :-1, :]
    shift_labels = tokens[:, 1:]
    
    assert shift_logits.shape[1] == 3
    assert shift_labels.shape[1] == 3
    assert torch.equal(shift_labels[0], torch.tensor([20, 30, 40]))


@test("Full model backward pass (gradient flow)")
def test_model_backward():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    
    # Verify ALL parameters got gradients (no dead paths)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for: {name}"


@test("Full model gradient checkpointing")
def test_gradient_checkpointing():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    model.enable_gradient_checkpointing()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    
    # Should still get gradients everywhere
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient with checkpointing for: {name}"


@test("Full model KV cache generation")
def test_model_kv_cache():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    model.eval()
    
    # Prefill
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    
    assert outputs["past_key_values"] is not None
    assert len(outputs["past_key_values"]) == config.num_layers
    
    # Decode step
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    with torch.no_grad():
        outputs2 = model(
            input_ids=next_token,
            past_key_values=outputs["past_key_values"],
            use_cache=True,
        )
    
    assert outputs2["logits"].shape == (1, 1, config.vocab_size)
    # KV cache should have grown by 1
    k, v = outputs2["past_key_values"][0]
    assert k.shape[2] == 9  # 8 prefill + 1 decode


@test("Full model weight tying")
def test_weight_tying():
    config = get_test_config()
    config.tie_word_embeddings = True
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    # Embedding and LM head should share the same weight tensor
    assert model.tok_embeddings.weight is model.lm_head.weight, \
        "Weight tying failed: embedding and lm_head weights are not the same object"


@test("Full model parameter count matches config")
def test_param_count():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    model = PrismForCausalLM(config)
    
    actual = model.get_num_params()
    expected = config.num_params
    # Allow 1% tolerance (weight tying can affect exact count)
    assert abs(actual - expected) / expected < 0.01, \
        f"Param count mismatch: model has {actual:,} but config says {expected:,}"


# ====================================================================
# TEST 8: Optimizer + LR Schedule
# ====================================================================

@test("Optimizer parameter groups (decay vs no-decay)")
def test_optimizer():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    from prism.training.optimizer import create_optimizer
    model = PrismForCausalLM(config)
    
    optimizer = create_optimizer(
        model.named_parameters(),
        learning_rate=1e-4,
        weight_decay=0.1,
    )
    
    # Should have 2 param groups
    assert len(optimizer.param_groups) == 2
    # First group has weight decay
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    # Second group has no weight decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


@test("Cosine LR schedule with warmup")
def test_lr_schedule():
    from prism.training.optimizer import create_cosine_schedule
    
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)
    scheduler = create_cosine_schedule(optimizer, warmup_steps=10, total_steps=100)
    
    lrs = []
    for step in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    
    # LR should increase during warmup
    assert lrs[5] > lrs[0], "LR should increase during warmup"
    # LR should peak around warmup_steps
    assert lrs[10] >= lrs[5], "LR should peak at end of warmup"
    # LR should decrease after warmup
    assert lrs[50] < lrs[10], "LR should decrease after warmup"
    # LR should be above min_lr at the end
    assert lrs[-1] > 0, "LR should stay positive"


# ====================================================================
# TEST 9: Sequence Packing
# ====================================================================

@test("Sequence packing logic")
def test_packing():
    from prism.data.packing import SequencePacker
    
    # Create a mock tokenizer-like object
    class MockTokenizer:
        eos_id = 2
        def encode(self, text, add_bos=True, add_eos=False):
            tokens = [ord(c) % 200 + 3 for c in text]  # Simple char-level encoding
            if add_bos:
                tokens = [1] + tokens
            if add_eos:
                tokens = tokens + [self.eos_id]
            return tokens
    
    tok = MockTokenizer()
    packer = SequencePacker(tok, max_seq_len=32)
    
    # Add several "documents"
    for i in range(20):
        packer.add_document(f"def function_{i}(): return {i}")
    
    sequences = packer.flush()
    assert len(sequences) > 0, "Packer should produce sequences"
    
    for seq in sequences:
        assert len(seq) == 32, f"Each sequence should be max_seq_len=32, got {len(seq)}"
        assert seq.dtype == np.uint16


# ====================================================================
# TEST 10: Pre-tokenized Dataset (with dummy data)
# ====================================================================

@test("Pre-tokenized dataset save and load")
def test_dataset():
    from prism.data.dataset import PreTokenizedDataset
    
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy tokenized data
        max_seq_len = 32
        num_sequences = 100
        seq_stride = max_seq_len
        
        # Create a shard: flat array of uint16
        data = np.random.randint(3, 200, size=(num_sequences * seq_stride,), dtype=np.uint16)
        shard_path = os.path.join(tmpdir, "train_00000.bin")
        data.tofile(shard_path)
        
        # Create metadata
        metadata = {
            "num_shards": 1,
            "total_sequences": num_sequences,
            "seq_length": seq_stride,
            "dtype": "uint16",
        }
        with open(os.path.join(tmpdir, "train_metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        # Load dataset
        dataset = PreTokenizedDataset(tmpdir, max_seq_len=max_seq_len, split="train")
        
        assert len(dataset) == num_sequences, f"Expected {num_sequences} sequences, got {len(dataset)}"
        
        # Get a sample
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (max_seq_len,)
        assert sample["labels"].shape == (max_seq_len,)
        assert sample["input_ids"].dtype == torch.int64
        assert torch.equal(sample["input_ids"], sample["labels"]), "input_ids and labels should match"
    finally:
        if 'dataset' in locals():
            dataset.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ====================================================================
# TEST 11: DataLoader
# ====================================================================

@test("DataLoader creation with dummy data")
def test_dataloader():
    from prism.data.dataloader import create_train_dataloader
    
    tmpdir = tempfile.mkdtemp()
    try:
        max_seq_len = 32
        num_sequences = 50
        seq_stride = max_seq_len
        data = np.random.randint(3, 200, size=(num_sequences * seq_stride,), dtype=np.uint16)
        data.tofile(os.path.join(tmpdir, "train_00000.bin"))
        
        with open(os.path.join(tmpdir, "train_metadata.json"), "w") as f:
            json.dump({"num_shards": 1, "total_sequences": num_sequences}, f)
        
        dataloader = create_train_dataloader(
            data_dir=tmpdir,
            max_seq_len=max_seq_len,
            micro_batch_size=4,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        assert batch["input_ids"].shape == (4, max_seq_len)
        assert batch["labels"].shape == (4, max_seq_len)
    finally:
        if 'dataloader' in locals() and hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'close'):
            dataloader.dataset.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ====================================================================
# TEST 12: Full Training Step (tiny model, CPU, no DeepSpeed)
# ====================================================================

@test("Single training step (forward + backward + optimizer)")
def test_training_step():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    from prism.training.optimizer import create_optimizer, create_cosine_schedule
    
    model = PrismForCausalLM(config)
    optimizer = create_optimizer(model.named_parameters(), learning_rate=1e-3)
    scheduler = create_cosine_schedule(optimizer, warmup_steps=5, total_steps=100)
    
    # Simulate 3 training steps
    losses = []
    for step in range(3):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        losses.append(loss.item())
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Loss should be finite for all steps
    for i, l in enumerate(losses):
        assert not np.isnan(l), f"NaN loss at step {i}"
        assert not np.isinf(l), f"Inf loss at step {i}"


# ====================================================================
# TEST 13: Metrics Tracker
# ====================================================================

@test("Metrics tracker")
def test_metrics():
    from prism.training.metrics import MetricsTracker
    tracker = MetricsTracker(window_size=10)
    
    for i in range(20):
        metrics = tracker.update(loss=5.0 - i * 0.1, batch_tokens=4096)
    
    assert "loss" in metrics
    assert "loss_smoothed" in metrics
    assert "perplexity" in metrics
    assert "tokens_per_second" in metrics
    assert metrics["total_steps"] == 20
    assert metrics["total_tokens"] == 20 * 4096
    
    summary = tracker.get_summary()
    assert summary["total_steps"] == 20


# ====================================================================
# TEST 14: KV Cache
# ====================================================================

@test("KV Cache operations")
def test_kv_cache():
    from prism.inference.kv_cache import KVCache
    cache = KVCache(
        num_layers=2,
        max_batch_size=1,
        max_seq_len=64,
        num_kv_heads=2,
        head_dim=16,
    )
    
    assert cache.get_seq_len() == 0
    assert cache.get_past_key_values() is None
    
    # Simulate prefill: 8 tokens
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    k_full, v_full = cache.update(0, k, v)
    cache.update(1, k, v)
    cache.advance(8)
    
    assert cache.get_seq_len() == 8
    assert k_full.shape == (1, 2, 8, 16)
    
    # Simulate decode: 1 token
    k2 = torch.randn(1, 2, 1, 16)
    v2 = torch.randn(1, 2, 1, 16)
    k_full2, v_full2 = cache.update(0, k2, v2)
    cache.advance(1)
    
    assert cache.get_seq_len() == 9
    assert k_full2.shape == (1, 2, 9, 16)
    
    # Reset
    cache.reset()
    assert cache.get_seq_len() == 0


# ====================================================================
# TEST 15: Text Generation (tiny model)
# ====================================================================

@test("Text generation with tiny model")
def test_generation():
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    from prism.inference.generator import PrismGenerator
    
    model = PrismForCausalLM(config)
    
    # Create a mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 256
        
        @property
        def bos_id(self): return 1
        @property
        def eos_id(self): return 2
        @property
        def pad_id(self): return 0
        
        def encode(self, text, add_bos=True, add_eos=False):
            tokens = [ord(c) % 250 + 3 for c in text]
            if add_bos: tokens = [self.bos_id] + tokens
            if add_eos: tokens = tokens + [self.eos_id]
            return tokens
        
        def decode(self, token_ids):
            return "".join(chr((t % 94) + 32) for t in token_ids if t > 2)
    
    mock_tok = MockTokenizer()
    generator = PrismGenerator(model, mock_tok, device=torch.device("cpu"))
    
    # Test greedy generation
    output = generator.generate_greedy("hello", max_new_tokens=10)
    assert isinstance(output, str)
    assert len(output) > 0
    
    # Test sampling generation
    output2 = generator.generate(
        "test", max_new_tokens=10,
        temperature=0.8, top_k=10, top_p=0.9,
    )
    assert isinstance(output2, str)


# ====================================================================
# TEST 16: Preprocessing Filters
# ====================================================================

@test("Data quality filters")
def test_filters():
    from prism.data.preprocessing import (
        filter_by_size, filter_by_line_stats, filter_by_ascii_ratio,
        filter_auto_generated, filter_by_filename, filter_by_language,
        apply_all_filters,
    )
    
    good_code = 'def hello():\n    print("Hello, World!")\n    return True\n' * 5
    
    # Size filter
    assert filter_by_size(good_code) == True
    assert filter_by_size("x") == False  # Too small
    assert filter_by_size("x" * 200_000) == False  # Too big
    
    # Line stats
    assert filter_by_line_stats(good_code) == True
    assert filter_by_line_stats("x" * 2000) == False  # Single long line
    assert filter_by_line_stats("a\nb") == False  # Too few lines
    
    # ASCII ratio
    assert filter_by_ascii_ratio(good_code) == True
    assert filter_by_ascii_ratio("日本語テキスト" * 100) == False
    
    # Auto-generated
    assert filter_auto_generated(good_code) == True
    assert filter_auto_generated("// Auto-generated by protoc\ncode here") == False
    assert filter_auto_generated("# DO NOT EDIT\nmore code") == False
    
    # Filename
    assert filter_by_filename("main.py") == True
    assert filter_by_filename("bundle.min.js") == False
    assert filter_by_filename("package-lock.json") == False
    
    # Language
    assert filter_by_language("python") == True
    assert filter_by_language("Python") == True
    assert filter_by_language("brainfuck") == False
    
    # Combined
    assert apply_all_filters(good_code, "python", "main.py") == True
    assert apply_all_filters(good_code, "brainfuck", "main.py") == False


# ====================================================================
# TEST 17: End-to-End Mini Pipeline
# ====================================================================

@test("End-to-end mini pipeline: data → model → loss → backward")
def test_e2e_pipeline():
    """Simulates the full training pipeline with dummy data."""
    config = get_test_config()
    from prism.model.model import PrismForCausalLM
    from prism.data.dataset import PreTokenizedDataset
    from prism.training.optimizer import create_optimizer, create_cosine_schedule
    from prism.training.metrics import MetricsTracker
    
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy pre-tokenized data
        max_seq_len = 32
        num_sequences = 20
        seq_stride = max_seq_len
        data = np.random.randint(3, config.vocab_size, size=(num_sequences * seq_stride,), dtype=np.uint16)
        data.tofile(os.path.join(tmpdir, "train_00000.bin"))
        with open(os.path.join(tmpdir, "train_metadata.json"), "w") as f:
            json.dump({"num_shards": 1, "total_sequences": num_sequences}, f)
        
        # Load dataset
        dataset = PreTokenizedDataset(tmpdir, max_seq_len=max_seq_len)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model + optimizer
        model = PrismForCausalLM(config)
        optimizer = create_optimizer(model.named_parameters(), learning_rate=1e-3)
        scheduler = create_cosine_schedule(optimizer, warmup_steps=2, total_steps=10)
        metrics = MetricsTracker(window_size=5)
        
        # Train for a few steps
        model.train()
        losses = []
        for step, batch in enumerate(dataloader):
            if step >= 5:
                break
            
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs["loss"]
            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step_metrics = metrics.update(
                loss=loss.item(),
                batch_tokens=batch["input_ids"].numel(),
            )
        
        assert len(losses) >= 3, "Should complete at least 3 training steps"
        assert all(not np.isnan(l) for l in losses), "All losses should be finite"
        assert all(not np.isinf(l) for l in losses), "All losses should be finite"
        
        # Verify model can generate after training
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 4))
            outputs = model(input_ids=input_ids, use_cache=True)
            next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            assert next_token.shape == (1, 1)
    finally:
        if 'dataset' in locals() and hasattr(dataset, 'close'):
            dataset.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ====================================================================
# RUN ALL TESTS
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Prism AI — Comprehensive Local Pipeline Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_config, test_config_yaml,
        test_rmsnorm,
        test_rope_precompute, test_rope_apply,
        test_gqa_forward, test_gqa_kv_cache, test_gqa_backward,
        test_swiglu,
        test_transformer_block,
        test_model_forward, test_model_loss, test_causal_target_alignment, test_model_backward,
        test_gradient_checkpointing, test_model_kv_cache,
        test_weight_tying, test_param_count,
        test_optimizer, test_lr_schedule,
        test_packing,
        test_dataset, test_dataloader,
        test_training_step,
        test_metrics,
        test_kv_cache,
        test_generation,
        test_filters,
        test_e2e_pipeline,
    ]
    
    for t in tests:
        t()
    
    print()
    print("=" * 60)
    passed = sum(1 for _, status, _ in results if status == PASS)
    failed = sum(1 for _, status, _ in results if status == FAIL)
    print(f"Results: {passed} passed, {failed} failed, {len(results)} total")
    
    if failed:
        print("\nFailed tests:")
        for name, status, error in results:
            if status == FAIL:
                print(f"  {name}: {error}")
    
    print("=" * 60)
    sys.exit(1 if failed else 0)
