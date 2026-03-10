from __future__ import annotations

import numpy as np

from mimic.envs.base import MimicEnv


class JointController:
    """Direct joint-space control with exponential smoothing.

    Maintains a _target array (what the user commands) separate from
    _current (the smoothed position sent to the env). The tick() method
    interpolates _current toward _target each frame for jitter-free motion.
    """

    def __init__(self, env: MimicEnv, speed: float = 0.12, alpha: float = 0.2):
        self.env = env
        self.speed = speed
        self.alpha = alpha
        self._target = np.zeros(env.action_dim)
        self._current = np.zeros(env.action_dim)
        self._reset_action()

    def _reset_action(self):
        """Reset target and current to current joint positions."""
        nq_arm = min(7, self.env.model.nq)  # 7 for Panda arm
        self._target[:nq_arm] = self.env.data.qpos[:nq_arm].copy()
        self._current[:nq_arm] = self.env.data.qpos[:nq_arm].copy()
        # Gripper stays at current
        if self.env.action_dim > 7:
            self._target[7:] = self.env.data.ctrl[7:].copy()
            self._current[7:] = self.env.data.ctrl[7:].copy()

    def tick(self) -> np.ndarray:
        """Smooth _current toward _target by alpha. Returns smoothed action."""
        self._current += (self._target - self._current) * self.alpha
        return self._current.copy()

    def process_command(self, command: dict) -> np.ndarray | None:
        """Process a command and update _target.

        Returns _target copy, or None if not handled.
        """
        cmd_type = command.get("type")

        if cmd_type == "joint_delta":
            joint = command.get("joint", 0)
            delta = command.get("delta", 0.0)
            if 0 <= joint < self.env.action_dim:
                self._target[joint] = np.clip(
                    self._target[joint] + delta * self.speed,
                    -3.14,
                    3.14,
                )
            return self._target.copy()

        elif cmd_type == "joint_absolute":
            joints = command.get("joints", [])
            for i, val in enumerate(joints):
                if i < self.env.action_dim:
                    self._target[i] = float(val)
            return self._target.copy()

        elif cmd_type == "gripper":
            value = command.get("value", 0.0)  # 0=closed, 1=open
            if self.env.action_dim > 7:
                self._target[7] = value * 0.04
                if self.env.action_dim > 8:
                    self._target[8] = value * 0.04
            return self._target.copy()

        elif cmd_type == "reset":
            self._reset_action()
            return None  # signal to reset env

        return None

    def get_action(self) -> np.ndarray:
        return self._current.copy()
