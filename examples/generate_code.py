"""Example: Autoregressive Python Code Generation with Prism AI.

Usage:
    python examples/generate_code.py --checkpoint checkpoints/step_0100000
"""

import argparse
import torch
from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.tokenizer import PrismTokenizer
from prism.inference.generator import PrismGenerator


def main():
    parser = argparse.ArgumentParser(description="Prism AI Code Generation Example")
    parser.add_argument("--config", default="configs/model_10b.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/step_0100000")
    parser.add_argument("--tokenizer", default="tokenizer/prism_tokenizer.model")
    parser.add_argument("--prompt", default="def merge_sort(arr: list) -> list:\n    \"\"\"Sort an array using merge sort algorithm.\"\"\"\n")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    config = PrismConfig.from_yaml(args.config)
    tokenizer = PrismTokenizer(args.tokenizer)
    model = PrismForCausalLM(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = PrismGenerator(model, tokenizer, device=device)

    print("\n" + "=" * 60)
    print("Prompt:")
    print("=" * 60)
    print(args.prompt)
    print("=" * 60)
    print("Generating Completion...\n")

    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    print("=" * 60)
    print("Model Output:")
    print("=" * 60)
    print(output)


if __name__ == "__main__":
    main()
