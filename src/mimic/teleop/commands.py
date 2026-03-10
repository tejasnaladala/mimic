from __future__ import annotations

from mimic.data.recorder import EpisodeRecorder
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
        self.recorder: EpisodeRecorder | None = None

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
        # Flow: start_recording → stop_recording (pauses) → save_episode or discard_episode
        if cmd_type == "start_recording":
            self._recording = True
            if self.recorder:
                self.recorder.start_recording()
            return {"status": "ok", "recording": True}
        if cmd_type == "stop_recording":
            self._recording = False
            # Just pause — don't save yet, wait for save/discard
            return {"status": "ok", "recording": False}
        if cmd_type == "save_episode":
            self._recording = False
            ep_idx = -1
            if self.recorder:
                ep_idx = self.recorder.stop_recording()
            return {
                "status": "ok",
                "episode_index": ep_idx,
                "episode_count": self.recorder.episode_count if self.recorder else 0,
            }
        if cmd_type == "discard_episode":
            self._recording = False
            if self.recorder:
                self.recorder.discard_recording()
            return {
                "status": "ok",
                "episode_count": self.recorder.episode_count if self.recorder else 0,
            }

        # Environment commands
        if cmd_type == "reset":
            self.env.reset()
            self.joint_ctrl._reset_action()
            self.cartesian_ctrl._reset_action()
            return {"status": "ok", "action": "reset"}

        # Joint release: target stays, smooth deceleration handled by tick()
        if cmd_type == "joint_release":
            return {"status": "ok"}

        # Control commands — only update targets, render loop handles stepping
        action = self.controller.process_command(command)
        if action is not None:
            return {"status": "ok"}

        return {"status": "unknown_command", "type": cmd_type}
