from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class InferenceServer:
    """Lightweight inference server for deployed policies."""

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "auto",  # "onnx", "torch", "auto"
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self._session = None
        self._policy = None
        self._action_buffer: np.ndarray | None = None
        self._buffer_idx = 0

        self._load_model()

    def _load_model(self):
        suffix = self.model_path.suffix.lower()

        if self.backend == "auto":
            if suffix == ".onnx":
                self.backend = "onnx"
            else:
                self.backend = "torch"

        if self.backend == "onnx":
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.model_path))
            logger.info(f"Loaded ONNX model: {self.model_path}")

        elif self.backend == "torch":
            import torch

            from mimic.train.policies.act import ACTPolicy
            from mimic.train.policies.diffusion import DiffusionPolicy

            ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            if "n_diffusion_steps" in config:
                self._policy = DiffusionPolicy.load(str(self.model_path))
            else:
                self._policy = ACTPolicy.load(str(self.model_path))
            self._policy.eval()
            logger.info(f"Loaded PyTorch model: {self.model_path}")

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict a single action from state observation.

        Uses temporal ensembling: predicts a full action chunk,
        then returns actions one at a time from the buffer.
        """
        # If we have buffered actions, return the next one
        if self._action_buffer is not None and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            return action

        # Need to predict a new action chunk
        if self.backend == "onnx":
            state_input = state.astype(np.float32).reshape(1, -1)
            result = self._session.run(None, {"state": state_input})
            actions = result[0].squeeze(0)  # [T, action_dim]

        elif self.backend == "torch":
            import torch

            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                actions_tensor = self._policy.predict({"state": state_tensor})
            actions = actions_tensor.squeeze(0).cpu().numpy()  # [T, action_dim]

        self._action_buffer = actions
        self._buffer_idx = 1
        return actions[0]

    def reset(self):
        """Clear the action buffer (call on episode reset)."""
        self._action_buffer = None
        self._buffer_idx = 0

    @property
    def is_loaded(self) -> bool:
        return self._session is not None or self._policy is not None
