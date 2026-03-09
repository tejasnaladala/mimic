from __future__ import annotations

import mujoco
import numpy as np

from mimic.envs.base import MimicEnv


class CartesianController:
    """End-effector Cartesian control using MuJoCo Jacobian IK."""

    def __init__(
        self,
        env: MimicEnv,
        speed: float = 0.02,
        site_name: str = "grip_site",
    ):
        self.env = env
        self.speed = speed
        self.site_name = site_name
        self._action = np.zeros(env.action_dim)
        self._nullspace_target = np.zeros(7)  # rest pose for arm
        self._reset_action()

        # Resolve site or fall back to body for Jacobian computation
        self._use_site = False
        self._site_id = -1
        self._body_id = -1
        self._resolve_ee_target()

    def _resolve_ee_target(self):
        """Resolve the end-effector target: prefer site, fall back to body."""
        model = self.env.model
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if site_id >= 0:
            self._use_site = True
            self._site_id = site_id
        else:
            # Fall back to panda_hand body
            self._use_site = False
            self._body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand"
            )
            if self._body_id < 0:
                # Last resort: use the last body
                self._body_id = model.nbody - 1

    def _reset_action(self):
        nq_arm = min(7, self.env.model.nq)
        self._action[:nq_arm] = self.env.data.qpos[:nq_arm].copy()

    def _compute_ik(self, dx: np.ndarray) -> np.ndarray:
        """Compute joint deltas from Cartesian delta using Jacobian."""
        model = self.env.model
        data = self.env.data
        nv = model.nv

        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        if self._use_site:
            mujoco.mj_jacSite(model, data, jacp, jacr, self._site_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, self._body_id)

        # Use only arm joints (first 7 velocity DOFs)
        n_arm = min(7, nv)
        J = jacp[:, :n_arm]

        # Damped least squares IK
        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, dx[:3])

        return dq

    def process_command(self, command: dict) -> np.ndarray | None:
        cmd_type = command.get("type")

        if cmd_type == "cartesian_delta":
            dx = np.array([
                command.get("dx", 0.0),
                command.get("dy", 0.0),
                command.get("dz", 0.0),
            ]) * self.speed

            dq = self._compute_ik(dx)
            n_arm = min(7, len(self._action))
            self._action[:n_arm] += dq
            # Clip to joint limits
            self._action[:n_arm] = np.clip(self._action[:n_arm], -2.9, 2.9)
            return self._action.copy()

        elif cmd_type == "gripper":
            value = command.get("value", 0.0)
            if self.env.action_dim > 7:
                self._action[7] = value * 0.04
                if self.env.action_dim > 8:
                    self._action[8] = value * 0.04
            return self._action.copy()

        elif cmd_type == "reset":
            self._reset_action()
            return None

        return None

    def get_ee_pos(self) -> np.ndarray:
        """Get current end-effector position."""
        if self._use_site:
            return self.env.data.site_xpos[self._site_id].copy()
        if self._body_id >= 0:
            return self.env.data.xpos[self._body_id].copy()
        return np.zeros(3)

    def get_action(self) -> np.ndarray:
        return self._action.copy()
