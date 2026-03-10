from __future__ import annotations

import asyncio
import logging
import math
import threading
import webbrowser
from collections.abc import Callable

import mujoco
import numpy as np
import uvicorn

from mimic.config.models import TeleopConfig
from mimic.data.dataset import MimicDataset
from mimic.data.recorder import EpisodeRecorder
from mimic.envs.base import MimicEnv
from mimic.teleop.commands import CommandRouter
from mimic.teleop.server import TeleopServer

logger = logging.getLogger(__name__)


class TeleopLoop:
    """Main teleoperation event loop.

    Combines the WebRTC server, command router, and render loop
    into a single runnable unit that serves the browser UI and
    streams MuJoCo frames in real time.
    """

    def __init__(self, env: MimicEnv, config: TeleopConfig | None = None, data_dir: str = "demo_data"):
        self.env = env
        self.config = config or TeleopConfig()
        self.server = TeleopServer(env)
        self.router = CommandRouter(env, mode=self.config.control_mode)
        self._running = False
        self._frame_callback: Callable | None = None

        # Set up episode recording (reuse existing dataset or create new)
        from pathlib import Path
        data_path = Path(data_dir)
        if (data_path / "meta" / "info.json").exists():
            self._dataset = MimicDataset(data_path)
        else:
            camera_names = [cam.name for cam in env.config.cameras]
            self._dataset = MimicDataset.create(
                path=data_dir,
                env_name=env.config.name,
                action_dim=env.action_dim,
                state_dim=env.state_dim,
                camera_names=camera_names,
                fps=env.config.control_hz,
            )
        self._recorder = EpisodeRecorder(self._dataset, env)
        self.router.recorder = self._recorder

        # Click-to-navigate: IK target tracking
        self._goto_target: np.ndarray | None = None
        self._goto_speed = 0.02  # meters per tick toward target
        self._goto_threshold = 0.015  # close enough to clear target
        self._resolve_ee()

        # Free-orbit camera (initialized from the named camera in the scene)
        self._free_cam = mujoco.MjvCamera()
        self._init_free_camera()

        # Wire up command handling
        self.server.on_command(self._handle_command)

    def _resolve_ee(self):
        """Find the end-effector site or body for IK."""
        model = self.env.model
        # Try "gripper" site first (Panda menagerie)
        self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self._ee_body_id = -1
        if self._ee_site_id < 0:
            # Fallback to hand body
            self._ee_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "hand"
            )
            if self._ee_body_id < 0:
                self._ee_body_id = model.nbody - 1

    def _get_ee_pos(self) -> np.ndarray:
        """Get current end-effector world position."""
        if self._ee_site_id >= 0:
            return self.env.data.site_xpos[self._ee_site_id].copy()
        return self.env.data.xpos[self._ee_body_id].copy()

    def _raycast_click(self, nx: float, ny: float) -> np.ndarray | None:
        """Convert normalized screen click (0-1) to 3D world position via ray casting."""
        cam = self._free_cam
        # Camera position from spherical coordinates
        az = math.radians(cam.azimuth)
        el = math.radians(cam.elevation)
        cos_el = math.cos(el)
        cam_pos = np.array([
            cam.lookat[0] + cam.distance * cos_el * math.cos(az),
            cam.lookat[1] + cam.distance * cos_el * math.sin(az),
            cam.lookat[2] + cam.distance * math.sin(el),
        ])

        # Camera coordinate frame
        forward = np.array(cam.lookat) - cam_pos
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / max(np.linalg.norm(right), 1e-8)
        up = np.cross(right, forward)

        # Get camera FOV from the renderer config
        fovy = 45.0  # default
        for cam_cfg in self.env.config.cameras:
            if cam_cfg.name == self.config.camera:
                fovy = cam_cfg.fovy
                break
        aspect = cam_cfg.width / cam_cfg.height if hasattr(cam_cfg, 'width') else 640 / 480

        # Convert normalized coords to ray direction
        half_fov = math.radians(fovy / 2)
        # nx, ny are [0,1] from top-left; convert to centered [-1,1]
        cx = (nx - 0.5) * 2.0
        cy = (0.5 - ny) * 2.0  # flip Y (screen Y is down)

        ray_dir = forward + cx * math.tan(half_fov) * aspect * right + cy * math.tan(half_fov) * up
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # MuJoCo ray cast: find intersection with scene geometry
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self.env.model, self.env.data,
            cam_pos, ray_dir,
            None,  # geomgroup (all groups)
            1,     # flg_static (include static geoms)
            -1,    # bodyexclude (-1 = none)
            geomid,
        )

        if dist < 0:
            # No intersection — fall back to table plane (z=0.45)
            if abs(ray_dir[2]) > 1e-6:
                t = (0.45 - cam_pos[2]) / ray_dir[2]
                if t > 0:
                    return cam_pos + t * ray_dir
            return None

        return cam_pos + dist * ray_dir

    def _ik_step_toward(self, target: np.ndarray) -> np.ndarray:
        """Compute one IK step toward the target position. Returns joint action."""
        model = self.env.model
        data = self.env.data
        nv = model.nv
        n_arm = min(7, nv)

        ee_pos = self._get_ee_pos()
        dx = target - ee_pos
        dist = np.linalg.norm(dx)

        # Clamp step size
        if dist > self._goto_speed:
            dx = dx * (self._goto_speed / dist)

        # Jacobian
        jacp = np.zeros((3, nv))
        if self._ee_site_id >= 0:
            mujoco.mj_jacSite(model, data, jacp, None, self._ee_site_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, None, self._ee_body_id)

        J = jacp[:, :n_arm]
        lam = 0.01
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), dx)

        # Apply to current joint targets
        action = self.router.controller.get_action()
        action[:n_arm] += dq
        action[:n_arm] = np.clip(action[:n_arm], -2.9, 2.9)
        return action

    def _init_free_camera(self):
        """Initialize the free camera from the scene's named camera."""
        cam = self._free_cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Read the named camera's position from the model
        cam_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera
        )
        if cam_id >= 0:
            # Get camera position and forward direction from model
            cam_pos = self.env.model.cam_pos[cam_id].copy()
            # Set lookat to table center (approximate)
            cam.lookat[:] = [0.5, 0.0, 0.45]
            # Compute distance, azimuth, elevation from position
            dx = cam_pos[0] - cam.lookat[0]
            dy = cam_pos[1] - cam.lookat[1]
            dz = cam_pos[2] - cam.lookat[2]
            cam.distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            cam.azimuth = float(math.degrees(math.atan2(dy, dx)))
            cam.elevation = float(
                math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))
            )
        else:
            # Sensible defaults
            cam.lookat[:] = [0.5, 0.0, 0.45]
            cam.distance = 1.8
            cam.azimuth = 30.0
            cam.elevation = -25.0

    def _handle_command(self, command: dict):
        """Handle incoming teleop commands."""
        cmd_type = command.get("type")

        # Camera orbit/pan/zoom commands
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
            # Pan in the camera's local XY plane
            dpx = command.get("dx", 0.0)
            dpy = command.get("dy", 0.0)
            az = math.radians(self._free_cam.azimuth)
            # Right vector
            self._free_cam.lookat[0] -= dpx * math.cos(az) - dpy * math.sin(az)
            self._free_cam.lookat[1] -= dpx * math.sin(az) + dpy * math.cos(az)
            self._free_cam.lookat[2] += command.get("dz", 0.0)
            return

        if cmd_type == "camera_reset":
            self._init_free_camera()
            return

        # Click-to-navigate: compute 3D target from screen click
        if cmd_type == "goto_click":
            nx = command.get("nx", 0.5)
            ny = command.get("ny", 0.5)
            world_pos = self._raycast_click(nx, ny)
            if world_pos is not None:
                self._goto_target = world_pos
                logger.info("Goto target: [%.3f, %.3f, %.3f]", *world_pos)
                self.server.send_state({
                    "status": "ok",
                    "goto_target": world_pos.tolist(),
                })
            return

        if cmd_type == "goto_cancel":
            self._goto_target = None
            return

        # All other commands go to the router
        response = self.router.process(command)
        self.server.send_state(response)

    async def _render_loop(self):
        """Continuously render and push frames to WebRTC.

        This is the ONLY place that steps physics — command handlers
        only update targets. Each frame: smooth, step, render, send state.
        """
        interval = 1.0 / self.config.fps
        while self._running:
            try:
                # If we have a goto target, override with IK step
                if self._goto_target is not None:
                    ee_pos = self._get_ee_pos()
                    if np.linalg.norm(self._goto_target - ee_pos) < self._goto_threshold:
                        self._goto_target = None
                        smoothed = self.router.controller.tick()
                    else:
                        action = self._ik_step_toward(self._goto_target)
                        # Sync the controller's internal state
                        ctrl = self.router.controller
                        if hasattr(ctrl, '_current'):
                            # JointController: update both target and current
                            ctrl._target[:] = action
                            ctrl._current[:] = action
                        else:
                            # CartesianController
                            ctrl._action[:] = action
                        smoothed = action
                else:
                    smoothed = self.router.controller.tick()
                obs, reward, done, info = self.env.step(smoothed)

                # Record frame if recording is active
                if self._recorder.is_recording:
                    self._recorder.record_frame(obs, smoothed, reward, done)

                # Render from the free-orbit camera
                frame = self.env.render_free(self._free_cam, self.config.camera)
                await self.server.push_frame(frame)

                # Send live state to frontend every frame for HUD updates
                state = {
                    "reward": float(reward),
                    "done": done,
                    "is_success": info.get("is_success", False),
                    "recording": self._recorder.is_recording,
                    "joint_pos": list(self.env.data.qpos[:7]),
                    "episode_count": self._recorder.episode_count,
                    "goto_active": self._goto_target is not None,
                }
                if self._goto_target is not None:
                    state["ee_pos"] = self._get_ee_pos().tolist()
                self.server.send_state(state)

                if self._frame_callback:
                    self._frame_callback(frame)
            except Exception as e:
                logger.error("Render error: %s", e)
            await asyncio.sleep(interval)

    async def _run_async(self):
        """Run the teleop server and render loop."""
        self._running = True
        config = uvicorn.Config(
            self.server.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        # Start render loop as a task
        render_task = asyncio.create_task(self._render_loop())

        try:
            await server.serve()
        finally:
            self._running = False
            render_task.cancel()
            await self.server.cleanup()

    def run(self, open_browser: bool = True):
        """Run the teleoperation server (blocking)."""
        if open_browser:
            # Open browser after a short delay
            url = f"http://localhost:{self.config.port}"
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        asyncio.run(self._run_async())

    def on_frame(self, callback: Callable):
        """Register a callback for each rendered frame (for recording)."""
        self._frame_callback = callback
