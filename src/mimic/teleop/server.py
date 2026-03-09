from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mimic.envs.base import MimicEnv
from mimic.teleop.video import MuJoCoVideoTrack

logger = logging.getLogger(__name__)


class TeleopServer:
    """WebRTC teleoperation server for MuJoCo environments."""

    def __init__(self, env: MimicEnv):
        self.env = env
        self.app = FastAPI(title="Mimic Teleoperation")
        self._pcs: set[RTCPeerConnection] = set()
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._command_callback: Callable | None = None
        self._data_channels: list = []
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/api/health")
        async def health():
            return {"status": "ok", "env": self.env.config.name}

        @self.app.get("/api/env-info")
        async def env_info():
            return {
                "name": self.env.config.name,
                "action_dim": self.env.action_dim,
                "state_dim": self.env.state_dim,
                "cameras": [c.model_dump() for c in self.env.config.cameras],
                "control_hz": self.env.config.control_hz,
            }

        @self.app.post("/api/offer")
        async def offer(request: Request):
            params = await request.json()
            sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            pc = RTCPeerConnection()
            self._pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                if pc.connectionState == "failed":
                    await pc.close()
                    self._pcs.discard(pc)

            # Add video track
            video = MuJoCoVideoTrack(self._frame_queue)
            pc.addTrack(video)

            # Handle data channel for commands
            @pc.on("datachannel")
            def on_datachannel(channel):
                self._data_channels.append(channel)

                @channel.on("message")
                def on_message(msg):
                    try:
                        data = json.loads(msg)
                        if self._command_callback:
                            self._command_callback(data)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from data channel: %s", msg)

                @channel.on("close")
                def on_close():
                    if channel in self._data_channels:
                        self._data_channels.remove(channel)

            await pc.setRemoteDescription(sdp)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Wait for ICE gathering
            while pc.iceGatheringState != "complete":
                await asyncio.sleep(0.05)

            return JSONResponse(
                {
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                }
            )

        # Serve frontend static files if they exist
        frontend_dir = Path(__file__).parent / "frontend" / "dist"
        if frontend_dir.exists():
            self.app.mount("/", StaticFiles(directory=str(frontend_dir), html=True))
        else:
            # Serve a minimal fallback HTML page
            @self.app.get("/")
            async def index():
                return HTMLResponse(self._fallback_html())

    async def push_frame(self, frame: np.ndarray):
        """Push a rendered frame to the WebRTC video track."""
        try:
            self._frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop frame if queue is full (maintain real-time)
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass

    def send_state(self, state: dict):
        """Send environment state to all connected clients via data channel."""
        msg = json.dumps(state)
        for channel in self._data_channels:
            try:
                channel.send(msg)
            except Exception:
                pass

    def on_command(self, callback: Callable):
        """Register a callback for incoming control commands."""
        self._command_callback = callback

    async def cleanup(self):
        """Close all peer connections."""
        for pc in self._pcs:
            await pc.close()
        self._pcs.clear()

    def _fallback_html(self) -> str:
        """Minimal HTML page for when React frontend isn't built."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Mimic Teleoperation</title>
    <style>
        body { font-family: system-ui; background: #1a1a2e; color: #eee;
               margin: 0; padding: 20px; }
        h1 { color: #00d4ff; }
        video { max-width: 100%; border: 2px solid #333; border-radius: 8px;
                background: #000; }
        .container { max-width: 800px; margin: 0 auto; }
        .controls { display: flex; gap: 10px; margin: 10px 0; }
        button { padding: 8px 16px; border-radius: 6px; border: none;
                 background: #00d4ff; color: #000; font-weight: bold;
                 cursor: pointer; }
        button:hover { background: #00b4d8; }
        #status { padding: 4px 8px; border-radius: 4px; font-size: 14px; }
        .connected { background: #2d6a4f; }
        .disconnected { background: #6a2d2d; }
        .info { background: #16213e; padding: 12px; border-radius: 8px;
                margin: 10px 0; }
        .keys { display: grid;
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                gap: 4px; margin: 10px 0; }
        .key { background: #16213e; padding: 4px 8px; border-radius: 4px;
               font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Mimic Teleoperation</h1>
        <div class="controls">
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <button onclick="sendCommand({type:'reset'})">Reset Env</button>
            <span id="status" class="disconnected">Disconnected</span>
        </div>
        <video id="video" autoplay playsinline></video>
        <div class="info">
            <strong>Keyboard Controls:</strong>
            <div class="keys">
                <span class="key">W/S: +/- Joint 0</span>
                <span class="key">A/D: +/- Joint 1</span>
                <span class="key">Q/E: +/- Joint 2</span>
                <span class="key">R/F: +/- Joint 3</span>
                <span class="key">T/G: +/- Joint 4</span>
                <span class="key">Y/H: +/- Joint 5</span>
                <span class="key">U/J: +/- Joint 6</span>
                <span class="key">O/L: Gripper</span>
                <span class="key">Space: Reset</span>
            </div>
        </div>
        <div class="info" id="env-info"></div>
    </div>
    <script>
        let pc = null;
        let dc = null;
        const video = document.getElementById("video");
        const status = document.getElementById("status");

        async function connect() {
            pc = new RTCPeerConnection();
            dc = pc.createDataChannel("commands");

            dc.onopen = () => {
                status.textContent = "Connected";
                status.className = "connected";
            };
            dc.onclose = () => {
                status.textContent = "Disconnected";
                status.className = "disconnected";
            };
            dc.onmessage = (e) => {
                try {
                    const state = JSON.parse(e.data);
                    // Handle state updates from server
                } catch(err) {}
            };

            pc.ontrack = (e) => { video.srcObject = e.streams[0]; };

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait for ICE gathering
            await new Promise(resolve => {
                if (pc.iceGatheringState === "complete") resolve();
                else pc.onicegatheringstatechange = () => {
                    if (pc.iceGatheringState === "complete") resolve();
                };
            });

            const resp = await fetch("/api/offer", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });
            const answer = await resp.json();
            await pc.setRemoteDescription(answer);
        }

        async function disconnect() {
            if (pc) { await pc.close(); pc = null; dc = null; }
            status.textContent = "Disconnected";
            status.className = "disconnected";
        }

        function sendCommand(cmd) {
            if (dc && dc.readyState === "open") {
                dc.send(JSON.stringify(cmd));
            }
        }

        // Keyboard control
        const keyMap = {
            "w": {joint: 0, dir: 0.1}, "s": {joint: 0, dir: -0.1},
            "a": {joint: 1, dir: 0.1}, "d": {joint: 1, dir: -0.1},
            "q": {joint: 2, dir: 0.1}, "e": {joint: 2, dir: -0.1},
            "r": {joint: 3, dir: 0.1}, "f": {joint: 3, dir: -0.1},
            "t": {joint: 4, dir: 0.1}, "g": {joint: 4, dir: -0.1},
            "y": {joint: 5, dir: 0.1}, "h": {joint: 5, dir: -0.1},
            "u": {joint: 6, dir: 0.1}, "j": {joint: 6, dir: -0.1},
            "o": {joint: 7, dir: 0.04}, "l": {joint: 7, dir: -0.04},
        };

        document.addEventListener("keydown", (e) => {
            if (e.key === " ") { sendCommand({type: "reset"}); return; }
            const mapping = keyMap[e.key.toLowerCase()];
            if (mapping) {
                sendCommand({
                    type: "joint_delta",
                    joint: mapping.joint,
                    delta: mapping.dir
                });
            }
        });

        // Load env info using safe DOM methods
        fetch("/api/env-info").then(r => r.json()).then(info => {
            const el = document.getElementById("env-info");
            el.textContent = "";
            const b1 = document.createElement("strong");
            b1.textContent = "Environment:";
            el.appendChild(b1);
            el.appendChild(document.createTextNode(" " + info.name + " | "));
            const b2 = document.createElement("strong");
            b2.textContent = "Action Dim:";
            el.appendChild(b2);
            el.appendChild(document.createTextNode(" " + info.action_dim + " | "));
            const b3 = document.createElement("strong");
            b3.textContent = "Control Hz:";
            el.appendChild(b3);
            el.appendChild(document.createTextNode(" " + info.control_hz));
        });
    </script>
</body>
</html>"""
