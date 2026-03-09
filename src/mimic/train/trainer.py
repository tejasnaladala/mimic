from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from mimic.config.models import TrainConfig
from mimic.train.dataloader import create_dataloader
from mimic.train.policies.base import MimicPolicy

logger = logging.getLogger(__name__)


class MimicTrainer:
    """Unified training loop for Mimic policies."""

    def __init__(
        self,
        policy: MimicPolicy,
        config: TrainConfig,
        dataset_path: str | Path,
        output_dir: str | Path = "outputs",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.policy = policy.to(self.device)
        self.optimizer = policy.get_optimizer(lr=config.lr)
        self.dataloader = create_dataloader(
            dataset_path,
            batch_size=config.batch_size,
            chunk_size=policy.action_chunk_size,
        )

        self._step = 0
        self._losses: list[float] = []

    def train(self, steps: int | None = None):
        """Run training loop."""
        steps = steps or self.config.steps
        self.policy.train()

        data_iter = iter(self.dataloader)
        start_time = time.time()

        for step in range(steps):
            # Get batch (restart dataloader if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    logger.error("Dataset is empty")
                    return

            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward + backward
            self.optimizer.zero_grad()
            output = self.policy.forward(batch)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            self._step += 1
            self._losses.append(loss.item())

            # Logging
            if self._step % 100 == 0:
                avg_loss = np.mean(self._losses[-100:])
                elapsed = time.time() - start_time
                steps_per_sec = self._step / elapsed
                logger.info(
                    f"Step {self._step}/{steps} | Loss: {avg_loss:.4f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

            # Save checkpoint
            if self.config.save_every > 0 and self._step % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_{self._step}.pt")

        # Final save
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, name: str):
        path = self.output_dir / name
        self.policy.save(str(path))
        logger.info(f"Saved checkpoint: {path}")

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def recent_loss(self) -> float:
        if not self._losses:
            return float("inf")
        return np.mean(self._losses[-100:])
