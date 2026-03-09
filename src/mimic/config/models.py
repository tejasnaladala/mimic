from __future__ import annotations

from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Configuration for a single camera in the environment."""

    name: str
    width: int = 320
    height: int = 240
    fovy: float = 45.0


class EnvConfig(BaseModel):
    """Configuration for a Mimic environment."""

    name: str
    robot: str = "panda"
    cameras: list[CameraConfig] = Field(
        default_factory=lambda: [
            CameraConfig(name="front", width=320, height=240),
            CameraConfig(name="wrist", width=320, height=240),
        ]
    )
    control_hz: int = 20
    physics_hz: int = 1000
    episode_length: int = 300
    action_space: str = "joint"  # "joint" | "cartesian"


class TrainConfig(BaseModel):
    """Configuration for training."""

    policy: str = "act"
    batch_size: int = 32
    lr: float = 1e-4
    steps: int = 100000
    eval_every: int = 5000
    save_every: int = 10000
    seed: int = 42
    device: str = "auto"
    wandb: bool = False


class TeleopConfig(BaseModel):
    """Configuration for teleoperation."""

    host: str = "0.0.0.0"
    port: int = 8765
    control_mode: str = "cartesian"  # "joint" | "cartesian"
    camera: str = "front"
    fps: int = 30
