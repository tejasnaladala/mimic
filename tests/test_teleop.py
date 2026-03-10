import asyncio

from fastapi.testclient import TestClient

import mimic.envs.tasks  # noqa: F401
from mimic.envs.registry import make


class TestTeleopServer:
    def test_server_creates(self):
        from mimic.teleop.server import TeleopServer

        env = make("pick-place")
        server = TeleopServer(env)
        assert server.app is not None
        assert server.env is env
        env.close()

    def test_health_endpoint(self):
        from mimic.teleop.server import TeleopServer

        env = make("pick-place")
        server = TeleopServer(env)
        client = TestClient(server.app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["env"] == "pick-place"
        env.close()

    def test_env_info_endpoint(self):
        from mimic.teleop.server import TeleopServer

        env = make("pick-place")
        server = TeleopServer(env)
        client = TestClient(server.app)
        resp = client.get("/api/env-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pick-place"
        assert "action_dim" in data
        assert "cameras" in data
        assert data["control_hz"] == 20
        env.close()

    def test_fallback_html_served(self):
        from mimic.teleop.server import TeleopServer

        env = make("pick-place")
        server = TeleopServer(env)
        client = TestClient(server.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MIMIC" in resp.text
        env.close()

    def test_on_command_callback(self):
        from mimic.teleop.server import TeleopServer

        env = make("pick-place")
        server = TeleopServer(env)
        received = []
        server.on_command(lambda cmd: received.append(cmd))
        assert server._command_callback is not None
        env.close()

    def test_video_track_creates(self):
        from mimic.teleop.video import MuJoCoVideoTrack

        queue = asyncio.Queue(maxsize=2)
        track = MuJoCoVideoTrack(queue)
        assert track.kind == "video"


class TestTeleopLoop:
    def test_creates(self):
        from mimic.teleop.loop import TeleopLoop

        env = make("pick-place")
        env.reset()
        loop = TeleopLoop(env)
        assert loop.env is env
        assert loop.server is not None
        assert loop.router is not None
        env.close()

    def test_creates_with_config(self):
        from mimic.config.models import TeleopConfig
        from mimic.teleop.loop import TeleopLoop

        env = make("pick-place")
        env.reset()
        config = TeleopConfig(port=9999, control_mode="cartesian")
        loop = TeleopLoop(env, config)
        assert loop.config.port == 9999
        assert loop.config.control_mode == "cartesian"
        env.close()

    def test_command_handling_reset(self):
        from mimic.teleop.loop import TeleopLoop

        env = make("pick-place")
        env.reset()
        loop = TeleopLoop(env)
        # Simulate a reset command -- should not raise
        loop._handle_command({"type": "reset"})
        env.close()

    def test_command_handling_joint_delta(self):
        from mimic.teleop.loop import TeleopLoop

        env = make("pick-place")
        env.reset()
        loop = TeleopLoop(env)
        # Simulate a joint delta command
        loop._handle_command({"type": "joint_delta", "joint": 0, "delta": 0.1})
        env.close()

    def test_frame_callback(self):
        from mimic.teleop.loop import TeleopLoop

        env = make("pick-place")
        env.reset()
        loop = TeleopLoop(env)
        frames = []
        loop.on_frame(lambda f: frames.append(f))
        assert loop._frame_callback is not None
        env.close()
