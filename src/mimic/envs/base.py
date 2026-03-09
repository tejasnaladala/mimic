from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mimic.config.models import EnvConfig


class MimicEnv:
    """Base environment for Mimic manipulation tasks."""

    def __init__(self, config: EnvConfig, xml_path: str | Path):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._step_count = 0
        self._steps_per_control = config.physics_hz // config.control_hz

        for cam in config.cameras:
            self._renderers[cam.name] = mujoco.Renderer(
                self.model, height=cam.height, width=cam.width
            )

    def reset(self) -> dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        """Step the environment with an action."""
        self._apply_action(action)
        for _ in range(self._steps_per_control):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._step_count >= self.config.episode_length or self._is_success()
        info = {"is_success": self._is_success()}
        return obs, reward, done, info

    def render(self, camera: str = "front") -> np.ndarray:
        """Render a camera view as RGB numpy array."""
        renderer = self._renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def render_all_cameras(self) -> dict[str, np.ndarray]:
        """Render all configured cameras."""
        return {cam.name: self.render(cam.name) for cam in self.config.cameras}

    def _apply_action(self, action: np.ndarray):
        """Apply action to MuJoCo actuators."""
        np.copyto(self.data.ctrl[: len(action)], action)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get current observation dict."""
        obs = {
            "state": np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]),
            "joint_pos": self.data.qpos.copy(),
            "joint_vel": self.data.qvel.copy(),
        }
        for cam in self.config.cameras:
            obs[f"image.{cam.name}"] = self.render(cam.name)
        return obs

    def _reset_task(self):
        """Override in subclass to randomize task."""

    def _compute_reward(self) -> float:
        """Override in subclass."""
        return 0.0

    def _is_success(self) -> bool:
        """Override in subclass."""
        return False

    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def state_dim(self) -> int:
        return self.model.nq + self.model.nv

    def close(self):
        """Clean up renderers."""
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __del__(self):
        self.close()
