from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    obs_dim: int | None = None,
) -> Path:
    """Export a Mimic policy checkpoint to ONNX format.

    The exported model takes a state observation and outputs an action chunk.
    """
    from mimic.train.policies.act import ACTPolicy
    from mimic.train.policies.diffusion import DiffusionPolicy

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint to determine type
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    if "n_diffusion_steps" in config:
        policy = DiffusionPolicy.load(str(checkpoint_path))
    else:
        policy = ACTPolicy.load(str(checkpoint_path))

    policy.eval()
    actual_obs_dim = obs_dim or policy.obs_dim

    # Create wrapper that takes flat state input
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            obs = {"state": state}
            return self.policy.predict(obs)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Dummy input
    dummy_state = torch.randn(1, actual_obs_dim)

    # Export using legacy exporter (dynamo=False avoids onnxscript dependency)
    torch.onnx.export(
        wrapper,
        dummy_state,
        str(output_path),
        input_names=["state"],
        output_names=["actions"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    logger.info(f"Exported ONNX model to {output_path}")
    return output_path


def verify_onnx(
    onnx_path: str | Path,
    obs_dim: int,
) -> bool:
    """Verify an ONNX model loads and runs correctly."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        dummy = np.random.randn(1, obs_dim).astype(np.float32)
        result = sess.run(None, {"state": dummy})
        logger.info(f"ONNX verification passed. Output shape: {result[0].shape}")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False
