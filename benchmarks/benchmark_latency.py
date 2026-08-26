"""Autoregressive Inference Latency Benchmark for Prism AI.

Measures:
  - TTFT (Time to First Token / Prompt prefill latency)
  - Inter-token generation latency (ms/token)
  - KV-cache memory scaling

Usage:
    python benchmarks/benchmark_latency.py --prompt_len 512 --gen_len 128
"""

import argparse
import time
import torch

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.inference.generator import PrismGenerator


class BenchmarkTokenizer:
    """Lightweight dummy tokenizer for benchmarking."""
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text, add_bos=True, add_eos=False):
        return [1] + [hash(w) % (self.vocab_size - 4) + 3 for w in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def run_latency_benchmark():
    parser = argparse.ArgumentParser(description="Prism AI Inference Latency Benchmark")
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = PrismConfig()
    if device.type == "cpu":
        config.num_layers = 2
        config.vocab_size = 1000

    print("=" * 65)
    print("Prism AI (10B) — Inference Latency & KV Cache Profiler")
    print("=" * 65)
    print(f"Device:            {device}")
    print(f"Prompt Length:     {args.prompt_len} tokens")
    print(f"Generated Tokens:  {args.gen_len} tokens")
    print("=" * 65)

    tokenizer = BenchmarkTokenizer(config.vocab_size)
    model = PrismForCausalLM(config).to(device=device)
    model.eval()

    generator = PrismGenerator(model, tokenizer, device=device)
    dummy_prompt = " ".join(["token"] * args.prompt_len)

    # Warmup
    print("\nWarming up...")
    _ = generator.generate_greedy(dummy_prompt, max_new_tokens=5)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Measuring Time to First Token (TTFT)...")
    t0 = time.perf_counter()
    tokens = torch.randint(0, config.vocab_size, (1, args.prompt_len), device=device)
    with torch.no_grad():
        out = model(input_ids=tokens, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    print("Measuring Autoregressive Decode Loop...")
    t0 = time.perf_counter()
    _ = generator.generate_greedy(dummy_prompt, max_new_tokens=args.gen_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_gen_time = time.perf_counter() - t0
    ms_per_token = (total_gen_time / args.gen_len) * 1000

    print("\n" + "=" * 65)
    print("Latency Results")
    print("=" * 65)
    print(f"Time to First Token (TTFT): {ttft_ms:.2f} ms")
    print(f"Decode Latency:             {ms_per_token:.2f} ms / token")
    print(f"Generation Throughput:      {1000 / ms_per_token:.1f} tokens / second")
    print("=" * 65)


if __name__ == "__main__":
    run_latency_benchmark()
