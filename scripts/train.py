"""Main training entry point for Prism AI.

Launch with DeepSpeed:
    deepspeed scripts/train.py \\
        --model_config configs/model_10b.yaml \\
        --train_config configs/training_10b.yaml \\
        --deepspeed configs/deepspeed/ds_zero3.json

For single-GPU debugging (not for real training):
    python scripts/train.py \\
        --model_config configs/model_10b.yaml \\
        --train_config configs/training_10b.yaml \\
        --deepspeed configs/deepspeed/ds_zero3.json
"""

import argparse
import json
import sys
import os

import yaml
import deepspeed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.model.config import PrismConfig
from prism.training.trainer import PrismTrainer
from prism.utils.distributed import print_rank0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Prism AI")

    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_10b.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/training_10b.yaml",
        help="Path to training configuration YAML",
    )

    # DeepSpeed adds its own arguments (--deepspeed, --local_rank, etc.)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # ── Load Configurations ─────────────────────────────────────────
    print_rank0("Loading configurations...")

    # Model config
    model_config = PrismConfig.from_yaml(args.model_config)
    model_config.validate()
    print_rank0(f"Model: {model_config}")
    print_rank0(f"Parameters: {model_config.num_params_billions:.2f}B")

    # Training config
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

    print_rank0(f"Training steps: {train_config.get('max_steps', 'N/A')}")
    print_rank0(f"Learning rate: {train_config.get('learning_rate', 'N/A')}")

    # DeepSpeed config path
    ds_config = args.deepspeed_config

    # ── Initialize and Run Trainer ──────────────────────────────────
    trainer = PrismTrainer(
        model_config=model_config,
        train_config=train_config,
        deepspeed_config=ds_config,
    )

    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
