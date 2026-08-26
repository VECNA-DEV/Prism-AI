"""Main Training Loop for Prism AI with DeepSpeed.

This is the core training driver that orchestrates:
  - DeepSpeed model/optimizer initialization
  - Distributed data loading
  - Training loop with gradient accumulation
  - Checkpoint save/resume
  - Logging and metrics tracking

DeepSpeed handles:
  - ZeRO-3 parameter sharding across GPUs
  - Mixed-precision (bf16) training
  - Gradient accumulation and all-reduce
  - Optimizer state management
  - Loss scaling (if using fp16)

Usage:
    deepspeed scripts/train.py --deepspeed configs/deepspeed/ds_zero3.json
"""

import os
import math
import time
from typing import Optional, Dict, Any

import torch
import deepspeed

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.dataloader import create_train_dataloader, create_val_dataloader
from prism.training.optimizer import create_optimizer, create_cosine_schedule
from prism.training.checkpointing import CheckpointManager
from prism.training.train_logger import TrainLogger
from prism.training.metrics import MetricsTracker
from prism.utils.distributed import (
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    print_rank0,
    barrier,
    all_reduce_scalar,
)
from prism.utils.seed import set_seed_for_training


class PrismTrainer:
    """Main trainer for Prism AI with DeepSpeed.

    Handles the complete training lifecycle from initialization
    through training, evaluation, and checkpointing.

    Args:
        model_config: PrismConfig for model architecture.
        train_config: Dictionary of training hyperparameters.
        deepspeed_config: Path to DeepSpeed JSON config.
    """

    def __init__(
        self,
        model_config: PrismConfig,
        train_config: Dict[str, Any],
        deepspeed_config: str,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.deepspeed_config = deepspeed_config

        # Training hyperparameters
        self.max_steps = train_config.get("max_steps", 100_000)
        self.warmup_steps = train_config.get("warmup_steps", 2000)
        self.learning_rate = train_config.get("learning_rate", 1.5e-4)
        self.weight_decay = train_config.get("weight_decay", 0.1)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.micro_batch_size = train_config.get("micro_batch_size", 2)
        self.eval_interval = train_config.get("eval_interval", 1000)
        self.log_interval = train_config.get("log_interval", 10)
        self.checkpoint_interval = train_config.get("checkpoint_interval", 500)
        self.max_checkpoints = train_config.get("max_checkpoints", 5)
        self.seed = train_config.get("seed", 42)

        # Data paths
        self.data_dir = train_config.get("data_dir", "data/tokenized")
        self.val_data_dir = train_config.get("val_data_dir", "data/tokenized")
        self.checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")

        # Initialize components
        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.checkpoint_manager = None
        self.logger = None
        self.metrics = None

        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def setup(self) -> None:
        """Initialize all training components.

        This sets up (in order):
          1. Random seeds
          2. Model
          3. DeepSpeed engine (wraps model + optimizer)
          4. Data loaders
          5. Checkpoint manager (+ auto-resume)
          6. Logger and metrics
        """
        print_rank0("=" * 60)
        print_rank0("Prism AI Training Setup")
        print_rank0("=" * 60)

        # ── 1. Seeds ────────────────────────────────────────────────
        set_seed_for_training(self.seed, get_rank())

        # ── 2. Model ────────────────────────────────────────────────
        print_rank0(f"\nInitializing model...")
        model = PrismForCausalLM(self.model_config)
        model.enable_gradient_checkpointing()

        num_params = model.get_num_params()
        print_rank0(f"  Model parameters: {num_params / 1e9:.2f}B")
        print_rank0(f"  Config: {self.model_config}")

        # ── 3. DeepSpeed ────────────────────────────────────────────
        print_rank0(f"\nInitializing DeepSpeed...")

        # Create optimizer (DeepSpeed will manage it)
        optimizer = create_optimizer(
            model.named_parameters(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create LR scheduler
        lr_scheduler = create_cosine_schedule(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
        )

        # Initialize DeepSpeed engine
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.deepspeed_config,
        )

        print_rank0(f"  DeepSpeed ZeRO stage: {self.model_engine.zero_optimization_stage()}")
        print_rank0(f"  World size: {get_world_size()}")
        print_rank0(f"  Local rank: {get_local_rank()}")

        # ── 4. Data Loaders ─────────────────────────────────────────
        print_rank0(f"\nLoading training data from {self.data_dir}...")

        self.train_dataloader = create_train_dataloader(
            data_dir=self.data_dir,
            max_seq_len=self.model_config.max_seq_len,
            micro_batch_size=self.micro_batch_size,
            seed=self.seed,
        )

        if os.path.exists(os.path.join(self.val_data_dir, "val_00000.bin")):
            self.val_dataloader = create_val_dataloader(
                data_dir=self.val_data_dir,
                max_seq_len=self.model_config.max_seq_len,
            )
            print_rank0(f"  Validation data loaded.")
        else:
            print_rank0(f"  No validation data found. Skipping eval.")

        # ── 5. Checkpoint Manager ───────────────────────────────────
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=self.max_checkpoints,
            save_interval=self.checkpoint_interval,
        )

        # Auto-resume from latest checkpoint
        client_state = self.checkpoint_manager.load_latest(self.model_engine)
        if client_state:
            self.global_step = client_state.get("global_step", 0)
            self.tokens_seen = client_state.get("tokens_seen", 0)
            self.epoch = client_state.get("epoch", 0)
            self.best_val_loss = client_state.get("best_loss", float("inf"))

        # ── 6. Logger & Metrics ─────────────────────────────────────
        self.logger = TrainLogger(
            project_name="prism-ai",
            run_name=self.train_config.get("run_name", f"prism-{num_params // 1e9:.0f}b"),
            config={**self.model_config.to_dict(), **self.train_config},
            log_interval=self.log_interval,
            use_wandb=self.train_config.get("use_wandb", True),
        )

        self.metrics = MetricsTracker(window_size=100)
        if self.global_step > 0:
            self.metrics.total_steps = self.global_step
            self.metrics.total_tokens = self.tokens_seen

        print_rank0(f"\n{'=' * 60}")
        print_rank0(f"Setup complete. Starting training from step {self.global_step}")
        print_rank0(f"{'=' * 60}\n")

    def train(self) -> None:
        """Run the main training loop.

        Iterates over the training data, performing:
          1. Forward pass (loss computation)
          2. Backward pass (gradient computation)
          3. Optimizer step (parameter update)
          4. Logging, checkpointing, evaluation
        """
        self.model_engine.train()

        while self.global_step < self.max_steps:
            self.epoch += 1

            # Update distributed sampler epoch for shuffling
            if hasattr(self.train_dataloader, "sampler") and hasattr(
                self.train_dataloader.sampler, "set_epoch"
            ):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            for batch in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.model_engine.device)
                labels = batch["labels"].to(self.model_engine.device)

                # ── Forward pass ────────────────────────────────────
                outputs = self.model_engine(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

                # ── Backward pass ───────────────────────────────────
                self.model_engine.backward(loss)

                # ── Optimizer step ──────────────────────────────────
                self.model_engine.step()

                # ── Token tracking ──────────────────────────────────
                batch_tokens = input_ids.numel() * get_world_size()
                self.tokens_seen += batch_tokens

                # DeepSpeed only updates weights at accumulation boundaries
                is_boundary = (
                    self.model_engine.is_gradient_accumulation_boundary()
                    if hasattr(self.model_engine, "is_gradient_accumulation_boundary")
                    else True
                )

                if is_boundary:
                    self.global_step += 1

                    # Aggregate loss across GPUs
                    loss_value = all_reduce_scalar(loss.item(), op="mean")

                    # Compute metrics
                    step_metrics = self.metrics.update(
                        loss=loss_value,
                        batch_tokens=batch_tokens,
                    )

                    # Get current LR
                    current_lr = (
                        self.lr_scheduler.get_last_lr()[0]
                        if self.lr_scheduler is not None
                        else self.learning_rate
                    )

                    # ── Logging ─────────────────────────────────────────
                    self.logger.log_step(
                        global_step=self.global_step,
                        metrics=step_metrics,
                        learning_rate=current_lr,
                    )

                    # System metrics (GPU memory) every 100 steps
                    if self.global_step % 100 == 0:
                        self.logger.log_system(self.global_step)

                    # ── Checkpointing ───────────────────────────────────
                    if self.checkpoint_manager.should_save(self.global_step):
                        self.checkpoint_manager.save(
                            model_engine=self.model_engine,
                            global_step=self.global_step,
                            tokens_seen=self.tokens_seen,
                            epoch=self.epoch,
                            best_loss=self.best_val_loss,
                        )

                    # ── Evaluation ──────────────────────────────────────
                    if (
                        self.val_dataloader is not None
                        and self.global_step % self.eval_interval == 0
                    ):
                        val_metrics = self.evaluate()
                        self.logger.log_eval(self.global_step, val_metrics)

                        # Track best validation loss
                        val_loss = val_metrics.get("loss", float("inf"))
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            print_rank0(f"  New best validation loss: {val_loss:.4f}")

                        self.model_engine.train()

        # ── Training complete ───────────────────────────────────────
        print_rank0("\n" + "=" * 60)
        print_rank0("Training complete!")
        summary = self.metrics.get_summary()
        print_rank0(f"  Total steps: {summary['total_steps']:,}")
        print_rank0(f"  Total tokens: {summary['total_tokens_billions']:.2f}B")
        print_rank0(f"  Elapsed: {summary['elapsed_hours']:.1f}h")
        print_rank0("=" * 60)

        # Final checkpoint
        self.checkpoint_manager.save(
            model_engine=self.model_engine,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            epoch=self.epoch,
            best_loss=self.best_val_loss,
        )

        self.logger.finish()

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> Dict[str, float]:
        """Run evaluation on the validation set.

        Args:
            max_batches: Maximum number of batches to evaluate.

        Returns:
            Dict with evaluation metrics.
        """
        self.model_engine.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in self.val_dataloader:
            if num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.model_engine.device)
            labels = batch["labels"].to(self.model_engine.device)

            outputs = self.model_engine(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

        # Aggregate across GPUs
        avg_loss = total_loss / max(total_tokens, 1)
        avg_loss = all_reduce_scalar(avg_loss, op="mean")

        perplexity = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches,
        }
