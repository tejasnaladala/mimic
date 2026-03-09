from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
