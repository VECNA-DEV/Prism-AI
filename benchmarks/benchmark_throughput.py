"""Throughput and VRAM Benchmark for Prism AI (10B).

Measures:
  - Tokens per second (TPS) across varied sequence lengths and batch sizes
  - Peak GPU memory allocation and reserved memory
  - MFU (Model FLOPs Utilization) estimation

Usage:
    python benchmarks/benchmark_throughput.py --batch_size 2 --seq_len 4096
"""

import argparse
import time
import torch

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.utils.profiling import ModelProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="Prism AI Throughput Benchmark")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print("=" * 65)
    print("Prism AI (10B) — Distributed Compute & Throughput Benchmark")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Precision:       {dtype}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Batch Tokens:    {args.batch_size * args.seq_len:,}")
    print("=" * 65)

    config = PrismConfig()
    profiler = ModelProfiler(config)
    print(profiler.format_summary())

    if device.type == "cpu":
        print("\nNote: Running on CPU in scaled down mode for validation.")
        config.num_layers = 2
        config.vocab_size = 1000

    model = PrismForCausalLM(config).to(device=device, dtype=dtype)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()

    # Warmup
    print("\nWarming up kernels...")
    for _ in range(args.warmup_steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Benchmarking over {args.benchmark_steps} steps...")
    start_time = time.perf_counter()

    for step in range(args.benchmark_steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    total_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps = total_tokens / total_time
    step_time_ms = (total_time / args.benchmark_steps) * 1000

    print("\n" + "=" * 65)
    print("Benchmark Results")
    print("=" * 65)
    print(f"Total Time:         {total_time:.3f} s")
    print(f"Step Latency:       {step_time_ms:.2f} ms")
    print(f"Throughput:         {tps:,.1f} tokens/second")
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM:          {peak_vram:.2f} GB")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
