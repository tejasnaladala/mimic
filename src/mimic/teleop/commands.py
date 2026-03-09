from __future__ import annotations

from mimic.envs.base import MimicEnv
from mimic.teleop.controllers.cartesian import CartesianController
from mimic.teleop.controllers.joint import JointController


class CommandRouter:
    """Routes teleop commands to the appropriate controller."""

    def __init__(self, env: MimicEnv, mode: str = "joint"):
        self.env = env
        self.mode = mode
        self.joint_ctrl = JointController(env)
        self.cartesian_ctrl = CartesianController(env)
        self._recording = False
        self._episode_callback = None

    @property
    def controller(self):
        if self.mode == "cartesian":
            return self.cartesian_ctrl
        return self.joint_ctrl

    def process(self, command: dict) -> dict:
        """Process a command and return response dict."""
        cmd_type = command.get("type")

        # Mode switching
        if cmd_type == "set_mode":
            self.mode = command.get("mode", "joint")
            return {"status": "ok", "mode": self.mode}

        # Recording control
        if cmd_type == "start_recording":
            self._recording = True
            return {"status": "ok", "recording": True}
        if cmd_type == "stop_recording":
            self._recording = False
            return {"status": "ok", "recording": False}

        # Environment commands
        if cmd_type == "reset":
            self.env.reset()
            self.joint_ctrl._reset_action()
            self.cartesian_ctrl._reset_action()
            return {"status": "ok", "action": "reset"}

        # Control commands
        action = self.controller.process_command(command)
        if action is not None:
            obs, reward, done, info = self.env.step(action)
            return {
                "status": "ok",
                "reward": float(reward),
                "done": done,
                "is_success": info.get("is_success", False),
                "recording": self._recording,
            }

        return {"status": "unknown_command", "type": cmd_type}
