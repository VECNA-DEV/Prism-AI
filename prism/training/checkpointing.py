"""Checkpoint management for Prism AI training.

Handles saving and loading training state for:
  - DeepSpeed model + optimizer + scheduler checkpoints
  - Training metadata (step count, tokens seen, best loss)
  - Auto-resume from the latest checkpoint on restart

Designed for robustness: training can be interrupted and resumed
at any checkpoint boundary without data loss.
"""

import os
import json
import glob
from typing import Optional, Dict, Any

from prism.utils.distributed import is_main_process, print_rank0


class CheckpointManager:
    """Manages training checkpoint save/load lifecycle.

    Checkpoints are saved as directories (DeepSpeed convention):
        checkpoint_dir/
          step_0500/
            config.json          ← training metadata
            global_step0500/     ← DeepSpeed model/optimizer state
          step_1000/
            ...
          latest                 ← pointer to latest checkpoint

    Args:
        checkpoint_dir: Root directory for all checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep (oldest deleted).
        save_interval: Save a checkpoint every N steps.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_interval: int = 500,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval

        os.makedirs(checkpoint_dir, exist_ok=True)

    def should_save(self, global_step: int) -> bool:
        """Check if a checkpoint should be saved at this step.

        Args:
            global_step: Current training step.

        Returns:
            True if this step is a checkpoint boundary.
        """
        return global_step > 0 and global_step % self.save_interval == 0

    def save(
        self,
        model_engine,
        global_step: int,
        tokens_seen: int,
        epoch: int,
        best_loss: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a training checkpoint.

        Uses DeepSpeed's native checkpoint saving, which handles
        model sharding, optimizer states, and scheduler state.

        Args:
            model_engine: DeepSpeed model engine.
            global_step: Current training step.
            tokens_seen: Total tokens processed so far.
            epoch: Current epoch.
            best_loss: Best validation loss so far.
            extra_state: Any additional state to save.

        Returns:
            Path to the saved checkpoint directory.
        """
        tag = f"step_{global_step:07d}"
        checkpoint_path = os.path.join(self.checkpoint_dir, tag)

        # Save training metadata (only on rank 0)
        client_state = {
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "epoch": epoch,
            "best_loss": best_loss,
        }
        if extra_state:
            client_state.update(extra_state)

        # DeepSpeed saves model, optimizer, scheduler, and client_state
        model_engine.save_checkpoint(
            self.checkpoint_dir,
            tag=tag,
            client_state=client_state,
        )

        # Update "latest" pointer
        if is_main_process():
            latest_path = os.path.join(self.checkpoint_dir, "latest")
            with open(latest_path, "w") as f:
                f.write(tag)

        print_rank0(f"Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest(self, model_engine) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint and resume training.

        Args:
            model_engine: DeepSpeed model engine.

        Returns:
            Client state dict if checkpoint found, None otherwise.
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest")

        if not os.path.exists(latest_path):
            print_rank0("No checkpoint found. Starting from scratch.")
            return None

        with open(latest_path, "r") as f:
            tag = f.read().strip()

        checkpoint_path = os.path.join(self.checkpoint_dir, tag)
        if not os.path.exists(checkpoint_path):
            print_rank0(f"Checkpoint {tag} referenced but not found. Starting from scratch.")
            return None

        print_rank0(f"Loading checkpoint: {checkpoint_path}")

        # DeepSpeed loads model, optimizer, scheduler state
        _, client_state = model_engine.load_checkpoint(
            self.checkpoint_dir,
            tag=tag,
        )

        if client_state:
            print_rank0(
                f"  Resumed from step {client_state.get('global_step', '?')}, "
                f"tokens seen: {client_state.get('tokens_seen', '?'):,}"
            )

        return client_state

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        if not is_main_process():
            return

        checkpoint_dirs = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "step_*")),
            key=os.path.getmtime,
        )

        while len(checkpoint_dirs) > self.max_checkpoints:
            oldest = checkpoint_dirs.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)
            print_rank0(f"  Removed old checkpoint: {os.path.basename(oldest)}")

    def list_checkpoints(self):
        """List all available checkpoints.

        Returns:
            List of (tag, path) tuples sorted by step.
        """
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "step_*"))
        )
        return [
            (os.path.basename(d), d)
            for d in checkpoint_dirs
        ]
