"""Evaluate Prism AI on code generation benchmarks.

Supports:
  - HumanEval (OpenAI): 164 Python programming problems
  - MBPP (Google): 974 basic Python problems

Usage:
    python scripts/evaluate.py \\
        --model_config configs/model_10b.yaml \\
        --checkpoint checkpoints/step_0100000 \\
        --tokenizer tokenizer/prism_tokenizer.model \\
        --benchmark humaneval
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.tokenizer import PrismTokenizer
from prism.inference.generator import PrismGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Prism AI")

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp"],
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per problem (pass@k)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl")

    return parser.parse_args()


def load_humaneval() -> List[Dict]:
    """Load HumanEval benchmark problems."""
    try:
        from human_eval.data import read_problems
        problems = read_problems()
        return [
            {
                "task_id": task_id,
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
                "test": problem["test"],
            }
            for task_id, problem in problems.items()
        ]
    except ImportError:
        print("human_eval not installed. Install with: pip install human-eval")
        print("Falling back to a simple evaluation...")
        return []


def evaluate_humaneval(generator: PrismGenerator, args):
    """Run HumanEval evaluation."""
    problems = load_humaneval()

    if not problems:
        print("No problems loaded. Skipping evaluation.")
        return

    print(f"\nRunning HumanEval ({len(problems)} problems, {args.num_samples} samples each)")
    print(f"Settings: temp={args.temperature}, top_p={args.top_p}")
    print("-" * 60)

    results = []

    for i, problem in enumerate(problems):
        prompt = problem["prompt"]

        for sample_idx in range(args.num_samples):
            completion = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=0,  # Disable top-k for HumanEval
            )

            # Extract only the completion (remove prompt)
            completion_only = completion[len(prompt):]

            results.append({
                "task_id": problem["task_id"],
                "completion": completion_only,
                "sample_idx": sample_idx,
            })

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(problems)} problems")

    # Save results
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nResults saved to {args.output_file}")
    print(f"Total completions: {len(results)}")
    print(f"\nTo compute pass@k, run:")
    print(f"  evaluate_functional_correctness {args.output_file}")


def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    config = PrismConfig.from_yaml(args.model_config)
    tokenizer = PrismTokenizer(args.tokenizer)
    model = PrismForCausalLM(config)

    # Load weights
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        import glob
        weight_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
        if weight_files:
            state_dict = torch.load(weight_files[0], map_location="cpu")
        else:
            mp_rank_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
            state_dict = torch.load(mp_rank_file, map_location="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = PrismGenerator(model, tokenizer, device)

    print(f"Model loaded: {config.num_params_billions:.2f}B parameters on {device}")

    # Run evaluation
    if args.benchmark == "humaneval":
        evaluate_humaneval(generator, args)
    else:
        print(f"Benchmark '{args.benchmark}' not yet implemented.")


if __name__ == "__main__":
    main()
