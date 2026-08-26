"""Train SentencePiece BPE tokenizer on code from The Stack v2.

This script:
  1. Streams code samples from HuggingFace (The Stack v2)
  2. Applies quality filters
  3. Saves filtered text to a temporary file
  4. Trains a SentencePiece BPE tokenizer with 32k vocabulary

Usage:
    python scripts/train_tokenizer.py \\
        --output_prefix tokenizer/prism_tokenizer \\
        --vocab_size 32000 \\
        --num_samples 2000000
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.data.dataset import StreamingCodeDataset
from prism.data.preprocessing import apply_all_filters, TARGET_LANGUAGES
from prism.data.tokenizer import train_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Prism AI tokenizer")

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="tokenizer/prism_tokenizer",
        help="Output prefix for tokenizer files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2_000_000,
        help="Number of code samples to use for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcode/the-stack-v2-dedup",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--max_file_size",
        type=int,
        default=50000,
        help="Maximum characters per file to include",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Training tokenizer with {args.num_samples} samples")
    print(f"  Dataset: {args.dataset}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Output: {args.output_prefix}.*")

    # ── Step 1: Stream and filter code samples ──────────────────────
    print("\nStep 1: Streaming and filtering code samples...")

    dataset = StreamingCodeDataset(
        dataset_name=args.dataset,
        languages=list(TARGET_LANGUAGES),
        streaming=True,
    )

    # Write filtered samples to a temp file
    temp_file = args.output_prefix + "_training_text.txt"

    samples_written = 0
    bytes_written = 0

    with open(temp_file, "w", encoding="utf-8") as f:
        for sample in dataset.iter_samples(max_samples=args.num_samples * 3):
            content = sample["content"]
            lang = sample["lang"]

            # Apply quality filters
            if not apply_all_filters(content, lang):
                continue

            # Truncate very long files
            if len(content) > args.max_file_size:
                content = content[:args.max_file_size]

            # Write code directly (SentencePiece trains naturally on raw lines/indentation)
            if not content.endswith("\n"):
                content += "\n"
            f.write(content)

            samples_written += 1
            bytes_written += len(content)

            if samples_written % 10000 == 0:
                print(f"  {samples_written:,} samples written ({bytes_written / 1e9:.2f} GB)")

            if samples_written >= args.num_samples:
                break

    print(f"\n  Total: {samples_written:,} samples, {bytes_written / 1e9:.2f} GB")

    # ── Step 2: Train SentencePiece ─────────────────────────────────
    print("\nStep 2: Training SentencePiece BPE tokenizer...")

    model_path = train_tokenizer(
        input_file=temp_file,
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
    )

    # ── Step 3: Verify ──────────────────────────────────────────────
    print("\nStep 3: Verifying tokenizer...")

    from prism.data.tokenizer import PrismTokenizer
    tokenizer = PrismTokenizer(model_path)

    test_texts = [
        'def hello_world():\n    print("Hello, World!")',
        'function fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n-1) + fibonacci(n-2);\n}',
        'SELECT COUNT(*) FROM users WHERE active = true;',
        'fn main() {\n    println!("Hello from Rust!");\n}',
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(tokens)
        pieces = tokenizer.tokenize(text)
        print(f"\n  Original:  {text[:80]}...")
        print(f"  Tokens:    {len(tokens)}")
        print(f"  Pieces:    {pieces[:10]}...")
        print(f"  Roundtrip: {'✓' if decoded.strip() == text.strip() else '✗'}")

    print(f"\nTokenizer saved to: {model_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Clean up temp file (optional — keeping it for reproducibility)
    # os.remove(temp_file)


if __name__ == "__main__":
    main()
