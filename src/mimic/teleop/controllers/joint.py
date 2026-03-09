from __future__ import annotations

import numpy as np

from mimic.envs.base import MimicEnv


class JointController:
    """Direct joint-space control. Maps joint deltas to actions."""

    def __init__(self, env: MimicEnv, speed: float = 0.05):
        self.env = env
        self.speed = speed
        self._action = np.zeros(env.action_dim)
        self._reset_action()

    def _reset_action(self):
        """Reset action to current joint positions."""
        nq_arm = min(7, self.env.model.nq)  # 7 for Panda arm
        self._action[:nq_arm] = self.env.data.qpos[:nq_arm].copy()
        # Gripper stays at current
        if self.env.action_dim > 7:
            self._action[7:] = self.env.data.ctrl[7:].copy()

    def process_command(self, command: dict) -> np.ndarray | None:
        """Process a command dict and return action array, or None if not handled."""
        cmd_type = command.get("type")

        if cmd_type == "joint_delta":
            joint = command.get("joint", 0)
            delta = command.get("delta", 0.0)
            if 0 <= joint < self.env.action_dim:
                self._action[joint] = np.clip(
                    self._action[joint] + delta * self.speed,
                    -3.14,
                    3.14,
                )
            return self._action.copy()

        elif cmd_type == "joint_absolute":
            joints = command.get("joints", [])
            for i, val in enumerate(joints):
                if i < self.env.action_dim:
                    self._action[i] = float(val)
            return self._action.copy()

        elif cmd_type == "gripper":
            value = command.get("value", 0.0)  # 0=closed, 1=open
            if self.env.action_dim > 7:
                self._action[7] = value * 0.04
                if self.env.action_dim > 8:
                    self._action[8] = value * 0.04
            return self._action.copy()

        elif cmd_type == "reset":
            self._reset_action()
            return None  # signal to reset env

        return None

    def get_action(self) -> np.ndarray:
        return self._action.copy()
