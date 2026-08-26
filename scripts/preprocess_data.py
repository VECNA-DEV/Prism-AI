"""Preprocess and tokenize The Stack v2 for Prism AI training.

This script:
  1. Streams code from The Stack v2 via HuggingFace
  2. Applies quality filters (size, language, auto-generated, etc.)
  3. Tokenizes with the trained SentencePiece tokenizer
  4. Packs sequences to max_seq_len for 100% token utilization
  5. Saves as binary shards for memory-mapped training

Usage:
    python scripts/preprocess_data.py \\
        --tokenizer_path tokenizer/prism_tokenizer.model \\
        --output_dir data/tokenized \\
        --num_tokens 200000000000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.data.dataset import StreamingCodeDataset
from prism.data.preprocessing import apply_all_filters, TARGET_LANGUAGES
from prism.data.tokenizer import PrismTokenizer
from prism.data.packing import SequencePacker, save_packed_shards


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for Prism AI training")

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to trained tokenizer .model file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenized",
        help="Output directory for tokenized shards",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcode/the-stack-v2-dedup",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=200_000_000_000,
        help="Target total tokens to process (default: 200B)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000,
        help="Number of sequences per shard",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.001,
        help="Fraction of data for validation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Prism AI Data Preprocessing")
    print("=" * 60)

    # ── Load tokenizer ──────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.tokenizer_path}")
    tokenizer = PrismTokenizer(args.tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Setup packers ───────────────────────────────────────────────
    train_packer = SequencePacker(tokenizer, max_seq_len=args.max_seq_len)
    val_packer = SequencePacker(tokenizer, max_seq_len=args.max_seq_len)

    # ── Stream and process ──────────────────────────────────────────
    print(f"\nStreaming from {args.dataset}...")
    print(f"  Target: {args.num_tokens / 1e9:.0f}B tokens")
    print(f"  Sequence length: {args.max_seq_len}")

    dataset = StreamingCodeDataset(
        dataset_name=args.dataset,
        languages=list(TARGET_LANGUAGES),
        streaming=True,
    )

    total_tokens = 0
    total_samples = 0
    filtered_samples = 0
    start_time = time.time()
    train_shard_idx = 0
    val_shard_idx = 0

    for sample in dataset.iter_samples():
        content = sample["content"]
        lang = sample["lang"]
        total_samples += 1

        # Quality filter
        if not apply_all_filters(content, lang):
            filtered_samples += 1
            continue

        # Tokenize
        tokens = tokenizer.encode(content, add_bos=False, add_eos=True)
        total_tokens += len(tokens)

        # Route to train or val packer
        import random
        if random.random() < args.val_ratio:
            val_packer.add_tokens(tokens)
        else:
            train_packer.add_tokens(tokens)

        # Progress reporting
        if total_samples % 50000 == 0:
            elapsed = time.time() - start_time
            rate = total_tokens / elapsed
            eta_hours = (args.num_tokens - total_tokens) / rate / 3600 if rate > 0 else 0

            print(
                f"  Processed {total_samples:>10,} files | "
                f"Filtered {filtered_samples:>10,} | "
                f"Tokens {total_tokens / 1e9:.2f}B / {args.num_tokens / 1e9:.0f}B | "
                f"Train seqs {train_packer.num_ready:,} | "
                f"Rate {rate / 1e6:.1f}M tok/s | "
                f"ETA {eta_hours:.1f}h"
            )

        # Flush shards periodically to manage memory
        if train_packer.num_ready >= args.shard_size:
            sequences = train_packer.flush()
            _, train_shard_idx = save_packed_shards(
                sequences, args.output_dir, 
                shard_size=args.shard_size, prefix="train",
                start_shard_idx=train_shard_idx,
            )

        if val_packer.num_ready >= args.shard_size:
            val_seqs = val_packer.flush()
            _, val_shard_idx = save_packed_shards(
                val_seqs, args.output_dir,
                shard_size=args.shard_size, prefix="val",
                start_shard_idx=val_shard_idx,
            )

        # Check if we've reached target
        if total_tokens >= args.num_tokens:
            print(f"\n  Reached target of {args.num_tokens / 1e9:.0f}B tokens.")
            break

    # ── Save remaining sequences ────────────────────────────────────
    print("\nSaving remaining sequences...")

    train_sequences = train_packer.flush()
    if train_sequences:
        save_packed_shards(
            train_sequences, args.output_dir, 
            shard_size=args.shard_size, prefix="train",
            start_shard_idx=train_shard_idx,
        )

    val_sequences = val_packer.flush()
    if val_sequences:
        save_packed_shards(
            val_sequences, args.output_dir, 
            shard_size=args.shard_size, prefix="val",
            start_shard_idx=val_shard_idx,
        )

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete!")
    print(f"  Total files processed: {total_samples:,}")
    print(f"  Files filtered out:    {filtered_samples:,}")
    print(f"  Total tokens:          {total_tokens / 1e9:.2f}B")
    print(f"  Training sequences:    {len(train_sequences):,}")
    print(f"  Validation sequences:  {len(val_sequences):,}")
    print(f"  Output directory:      {args.output_dir}")
    print(f"  Elapsed time:          {elapsed / 3600:.1f}h")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
