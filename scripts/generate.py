"""Interactive text generation CLI for Prism AI.

Usage:
    python scripts/generate.py \\
        --model_config configs/model_10b.yaml \\
        --checkpoint checkpoints/step_0100000 \\
        --tokenizer tokenizer/prism_tokenizer.model
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.tokenizer import PrismTokenizer
from prism.inference.generator import PrismGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate code with Prism AI")

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()


def load_model(args):
    """Load model from checkpoint."""
    print("Loading model...")

    config = PrismConfig.from_yaml(args.model_config)
    tokenizer = PrismTokenizer(args.tokenizer)

    model = PrismForCausalLM(config)

    # Load checkpoint weights
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        # DeepSpeed checkpoint directory — look for model state
        import glob
        weight_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
        if weight_files:
            state_dict = torch.load(weight_files[0], map_location="cpu")
        else:
            # Try DeepSpeed consolidated checkpoint
            mp_rank_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
            if os.path.exists(mp_rank_file):
                state_dict = torch.load(mp_rank_file, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]
            else:
                raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "module" in state_dict:
        state_dict = state_dict["module"]

    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = PrismGenerator(model, tokenizer, device)

    print(f"Model loaded on {device}")
    print(f"Parameters: {config.num_params_billions:.2f}B")

    return generator


def main():
    args = parse_args()
    generator = load_model(args)

    print("\n" + "=" * 60)
    print("Prism AI — Interactive Code Generation")
    print("=" * 60)
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()

            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Handle multi-line input (end with empty line)
            if prompt.endswith("\\"):
                lines = [prompt[:-1]]
                while True:
                    line = input("... ")
                    if not line:
                        break
                    if line.endswith("\\"):
                        lines.append(line[:-1])
                    else:
                        lines.append(line)
                        break
                prompt = "\n".join(lines)

            print("\n--- Generated ---")
            output = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(output)
            print("--- End ---\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
