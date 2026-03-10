from __future__ import annotations

import asyncio
import logging
import math
import threading
import webbrowser

import mujoco
import numpy as np
import pyarrow.parquet as pq
import uvicorn

from mimic.config.models import TeleopConfig
from mimic.envs.base import MimicEnv
from mimic.teleop.server import TeleopServer

logger = logging.getLogger(__name__)


class ReplayLoop:
    """Replays a recorded episode through the WebRTC viewer.

    Loads actions from a saved Parquet episode and steps the environment,
    streaming rendered frames to the browser. Camera orbit still works.
    """

    def __init__(
        self,
        env: MimicEnv,
        data_dir: str,
        episode_idx: int = 0,
        config: TeleopConfig | None = None,
    ):
        self.env = env
        self.config = config or TeleopConfig()
        self.server = TeleopServer(env)
        self._running = False

        # Load episode actions from Parquet
        from pathlib import Path
        parquet_path = Path(data_dir) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode {episode_idx} not found at {parquet_path}")

        table = pq.read_table(parquet_path)
        self._actions = [np.array(a) for a in table.column("action").to_pylist()]
        self._episode_idx = episode_idx
        self._num_frames = len(self._actions)

        # Free-orbit camera
        self._free_cam = mujoco.MjvCamera()
        self._init_free_camera()

        # Wire up camera commands only
        self.server.on_command(self._handle_command)

    def _init_free_camera(self):
        cam = self._free_cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        cam_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera
        )
        if cam_id >= 0:
            cam_pos = self.env.model.cam_pos[cam_id].copy()
            cam.lookat[:] = [0.5, 0.0, 0.45]
            dx = cam_pos[0] - cam.lookat[0]
            dy = cam_pos[1] - cam.lookat[1]
            dz = cam_pos[2] - cam.lookat[2]
            cam.distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            cam.azimuth = float(math.degrees(math.atan2(dy, dx)))
            cam.elevation = float(
                math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))
            )
        else:
            cam.lookat[:] = [0.5, 0.0, 0.45]
            cam.distance = 1.8
            cam.azimuth = 30.0
            cam.elevation = -25.0

    def _handle_command(self, command: dict):
        cmd_type = command.get("type")

        if cmd_type == "camera_orbit":
            self._free_cam.azimuth += command.get("daz", 0.0)
            self._free_cam.elevation = np.clip(
                self._free_cam.elevation + command.get("del", 0.0), -89.0, 89.0
            )
            return

        if cmd_type == "camera_zoom":
            self._free_cam.distance = max(
                0.3, self._free_cam.distance + command.get("dd", 0.0)
            )
            return

        if cmd_type == "camera_pan":
            dpx = command.get("dx", 0.0)
            dpy = command.get("dy", 0.0)
            az = math.radians(self._free_cam.azimuth)
            self._free_cam.lookat[0] -= dpx * math.cos(az) - dpy * math.sin(az)
            self._free_cam.lookat[1] -= dpx * math.sin(az) + dpy * math.cos(az)
            self._free_cam.lookat[2] += command.get("dz", 0.0)
            return

        if cmd_type == "camera_reset":
            self._init_free_camera()
            return

    async def _replay_loop(self):
        """Play back recorded actions at the environment's control rate."""
        interval = 1.0 / self.env.config.control_hz
        self.env.reset()
        frame_idx = 0

        while self._running and frame_idx < self._num_frames:
            try:
                action = self._actions[frame_idx]
                obs, reward, done, info = self.env.step(action)

                frame = self.env.render_free(self._free_cam, self.config.camera)
                await self.server.push_frame(frame)

                self.server.send_state({
                    "reward": float(reward),
                    "done": done,
                    "is_success": info.get("is_success", False),
                    "recording": False,
                    "joint_pos": list(self.env.data.qpos[:7]),
                    "episode_count": 0,
                    "replay": True,
                    "replay_frame": frame_idx,
                    "replay_total": self._num_frames,
                })

                frame_idx += 1
            except Exception as e:
                logger.error("Replay error: %s", e)
            await asyncio.sleep(interval)

        # Replay finished — keep streaming the final pose
        logger.info("Replay complete (%d frames)", self._num_frames)
        while self._running:
            try:
                frame = self.env.render_free(self._free_cam, self.config.camera)
                await self.server.push_frame(frame)
                self.server.send_state({
                    "reward": 0.0,
                    "done": True,
                    "is_success": False,
                    "recording": False,
                    "joint_pos": list(self.env.data.qpos[:7]),
                    "episode_count": 0,
                    "replay": False,
                })
            except Exception as e:
                logger.error("Post-replay error: %s", e)
            await asyncio.sleep(1.0 / 30)

    async def _run_async(self):
        self._running = True
        config = uvicorn.Config(
            self.server.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        replay_task = asyncio.create_task(self._replay_loop())

        try:
            await server.serve()
        finally:
            self._running = False
            replay_task.cancel()
            await self.server.cleanup()

    def run(self, open_browser: bool = True):
        """Run the replay server (blocking)."""
        if open_browser:
            url = f"http://localhost:{self.config.port}"
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        asyncio.run(self._run_async())
