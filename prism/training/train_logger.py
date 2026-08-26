"""Training logger for Prism AI — WandB and console logging.

Provides a unified logging interface that writes to both WandB
(for experiment tracking dashboards) and console (for real-time
monitoring in the terminal).

Named train_logger.py (not logging.py) to avoid conflicts with
Python's standard library logging module.
"""

import os
import time
from typing import Dict, Optional, Any

from prism.utils.distributed import is_main_process


class TrainLogger:
    """Unified training logger for WandB and console output.

    Only the main process (rank 0) logs to WandB and prints to
    console. Other ranks are silent to avoid duplicate output.

    Args:
        project_name: WandB project name.
        run_name: WandB run name (e.g., "prism-10b-v1").
        config: Training configuration dict to log.
        log_interval: Print to console every N steps.
        use_wandb: Whether to use WandB (requires wandb login).
    """

    def __init__(
        self,
        project_name: str = "prism-ai",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_interval: int = 10,
        use_wandb: bool = True,
    ):
        self.log_interval = log_interval
        self.use_wandb = use_wandb and is_main_process()
        self._wandb_run = None
        self._start_time = time.time()
        self._step_start_time = time.time()

        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    resume="allow",
                )
                print(f"WandB initialized: {wandb.run.url}")
            except Exception as e:
                print(f"WandB initialization failed: {e}")
                print("Continuing without WandB logging.")
                self.use_wandb = False

    def log_step(
        self,
        global_step: int,
        metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """Log metrics for a training step.

        Args:
            global_step: Current training step.
            metrics: Dict of metric name -> value.
            learning_rate: Current learning rate.
        """
        if not is_main_process():
            return

        # Add LR to metrics
        metrics["learning_rate"] = learning_rate

        # Compute elapsed time
        step_time = time.time() - self._step_start_time
        self._step_start_time = time.time()
        metrics["step_time_sec"] = step_time

        # WandB logging
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=global_step)

        # Console logging
        if global_step % self.log_interval == 0:
            elapsed = time.time() - self._start_time
            loss = metrics.get("loss", 0.0)
            ppl = metrics.get("perplexity", 0.0)
            tps = metrics.get("tokens_per_second", 0.0)

            print(
                f"Step {global_step:>7d} | "
                f"Loss {loss:.4f} | "
                f"PPL {ppl:.2f} | "
                f"LR {learning_rate:.2e} | "
                f"TPS {tps:,.0f} | "
                f"Time {step_time:.2f}s | "
                f"Elapsed {elapsed / 3600:.1f}h",
                flush=True,
            )

    def log_eval(
        self,
        global_step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log evaluation metrics.

        Args:
            global_step: Current training step.
            metrics: Dict of eval metric name -> value.
        """
        if not is_main_process():
            return

        # Prefix eval metrics
        eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}

        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log(eval_metrics, step=global_step)

        # Console
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"[Eval @ Step {global_step}] {' | '.join(parts)}", flush=True)

    def log_system(
        self,
        global_step: int,
    ) -> None:
        """Log system metrics (GPU memory, utilization).

        Args:
            global_step: Current training step.
        """
        if not is_main_process():
            return

        try:
            import torch
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                metrics = {
                    "system/gpu_mem_allocated_gb": mem_allocated,
                    "system/gpu_mem_reserved_gb": mem_reserved,
                }
                if self.use_wandb and self._wandb_run is not None:
                    import wandb
                    wandb.log(metrics, step=global_step)
        except Exception:
            pass

    def finish(self) -> None:
        """Finish the logging session."""
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.finish()
            print("WandB run finished.")

    def __del__(self):
        self.finish()
