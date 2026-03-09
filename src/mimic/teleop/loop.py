from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser
from collections.abc import Callable

import uvicorn

from mimic.config.models import TeleopConfig
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

    def __init__(self, env: MimicEnv, config: TeleopConfig | None = None):
        self.env = env
        self.config = config or TeleopConfig()
        self.server = TeleopServer(env)
        self.router = CommandRouter(env, mode=self.config.control_mode)
        self._running = False
        self._frame_callback: Callable | None = None

        # Wire up command handling
        self.server.on_command(self._handle_command)

    def _handle_command(self, command: dict):
        """Handle incoming teleop commands."""
        response = self.router.process(command)
        self.server.send_state(response)

    async def _render_loop(self):
        """Continuously render and push frames to WebRTC."""
        interval = 1.0 / self.config.fps
        while self._running:
            try:
                frame = self.env.render(self.config.camera)
                await self.server.push_frame(frame)
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
