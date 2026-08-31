from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _safe_config_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list | tuple):
        return all(_safe_config_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _safe_config_value(item)
            for key, item in value.items()
        )
    return False


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load Mimic's tensor-only checkpoint format without enabling pickle globals."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint rejected: only Mimic tensor-only checkpoints are accepted"
        ) from exc

    if not isinstance(checkpoint, dict) or set(checkpoint) != {"state_dict", "config"}:
        raise RuntimeError("Checkpoint rejected: expected state_dict and config only")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise RuntimeError("Checkpoint rejected: state_dict must contain tensors only")

    config = checkpoint["config"]
    if not isinstance(config, dict) or not _safe_config_value(config):
        raise RuntimeError("Checkpoint rejected: config contains unsupported values")
    return checkpoint


class MimicPolicy(ABC, nn.Module):
    """Base class for all Mimic policies."""

    def __init__(self, obs_dim: int, action_dim: int, action_chunk_size: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass. Returns dict with 'loss' key."""
        ...

    @abstractmethod
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference: predict action(s) from observation."""
        ...

    def get_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": self._get_config()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> MimicPolicy:
        checkpoint = load_checkpoint(path)
        config = checkpoint.get("config", {})
        config.update(kwargs)
        policy = cls(**config)
        policy.load_state_dict(checkpoint["state_dict"])
        return policy

    def _get_config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_chunk_size": self.action_chunk_size,
        }
