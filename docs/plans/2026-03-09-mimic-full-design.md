# Mimic - Full System Design & Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Mimic — the universal robot learning pipeline that lets anyone teach a robot new skills from their browser. Collect demonstrations via teleoperation, train foundation models, deploy to edge hardware. `pip install mimic-robotics`.

**Architecture:** Modular Python library with 6 layers: environments (MuJoCo), teleoperation (WebRTC), data (LeRobot-compatible), training (ACT/Diffusion Policy), deployment (ONNX/TensorRT export), and hub (HuggingFace). Browser-based UI for teleoperation with gamepad/touch/keyboard support. FastAPI backend with aiortc for WebRTC streaming.

**Tech Stack:** Python 3.11+, MuJoCo, aiortc, FastAPI, PyTorch, Pydantic v2, Typer+Rich CLI, HuggingFace Hub, Vite+React frontend

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        mimic CLI                            │
│   mimic env list | mimic teleop | mimic train | mimic deploy│
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                     mimic Python API                        │
│                                                             │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────┐ ┌────────┐ │
│  │  envs   │ │  teleop   │ │  data  │ │ train │ │ deploy │ │
│  │         │ │           │ │        │ │       │ │        │ │
│  │ MuJoCo  │ │ WebRTC    │ │ LeRobot│ │ ACT   │ │ ONNX   │ │
│  │ scenes  │ │ aiortc    │ │ v3 fmt │ │ Diff. │ │ TRT    │ │
│  │ tasks   │ │ FastAPI   │ │ Parquet│ │ Policy│ │ export │ │
│  │ robots  │ │ gamepad   │ │ MP4    │ │ VLA   │ │        │ │
│  └────┬────┘ └─────┬─────┘ └───┬────┘ └───┬───┘ └───┬────┘ │
│       │            │           │           │         │      │
│  ┌────▼────────────▼───────────▼───────────▼─────────▼────┐ │
│  │                    mimic.hub                           │ │
│  │            HuggingFace Hub integration                 │ │
│  │         datasets, models, environments                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Browser UI (React)                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Video Feed   │  │  Controls    │  │  Dataset Browser │  │
│  │  (WebRTC)     │  │  Gamepad     │  │  Episode viewer  │  │
│  │  Multi-cam    │  │  Touch       │  │  Stats           │  │
│  │              │  │  Keyboard    │  │  Annotation      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
mimic/
├── pyproject.toml              # uv/pip, project metadata
├── README.md
├── LICENSE                     # MIT
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── mimic/
│       ├── __init__.py         # version, public API
│       ├── cli/
│       │   ├── __init__.py
│       │   └── app.py          # Typer CLI with rich output
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── base.py         # MimicEnv base class
│       │   ├── registry.py     # environment registry
│       │   ├── tasks/
│       │   │   ├── __init__.py
│       │   │   ├── pick_place.py
│       │   │   ├── stack.py
│       │   │   ├── push.py
│       │   │   └── pour.py
│       │   ├── robots/
│       │   │   ├── __init__.py
│       │   │   ├── panda.py
│       │   │   ├── ur5e.py
│       │   │   └── so_arm100.py
│       │   └── assets/         # MJCF XMLs, meshes
│       │       ├── scenes/
│       │       ├── robots/
│       │       └── objects/
│       ├── teleop/
│       │   ├── __init__.py
│       │   ├── server.py       # FastAPI + aiortc signaling
│       │   ├── video.py        # MuJoCo -> WebRTC video track
│       │   ├── commands.py     # data channel command handler
│       │   ├── controllers/
│       │   │   ├── __init__.py
│       │   │   ├── joint.py    # joint space control
│       │   │   ├── cartesian.py # end-effector delta control
│       │   │   └── ik.py       # inverse kinematics
│       │   └── frontend/       # built React app (static files)
│       │       └── dist/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py      # MimicDataset class
│       │   ├── recorder.py     # episode recording during teleop
│       │   ├── formats.py      # LeRobot v3 / HDF5 / RLDS export
│       │   ├── stats.py        # normalization statistics
│       │   └── video.py        # MP4 encoding/decoding
│       ├── train/
│       │   ├── __init__.py
│       │   ├── trainer.py      # unified training loop
│       │   ├── policies/
│       │   │   ├── __init__.py
│       │   │   ├── base.py     # MimicPolicy base class
│       │   │   ├── act.py      # Action Chunking Transformer
│       │   │   └── diffusion.py # Diffusion Policy
│       │   ├── dataloader.py   # efficient data loading
│       │   └── eval.py         # sim evaluation during training
│       ├── deploy/
│       │   ├── __init__.py
│       │   ├── export.py       # ONNX / TensorRT export
│       │   └── inference.py    # real-time inference server
│       ├── hub/
│       │   ├── __init__.py
│       │   └── client.py       # HF Hub push/pull
│       ├── config/
│       │   ├── __init__.py
│       │   └── models.py       # all Pydantic config models
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── visualization.py
├── frontend/                   # React source (builds to src/mimic/teleop/frontend/dist)
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── VideoStream.tsx
│   │   │   ├── Controls.tsx
│   │   │   ├── GamepadControls.tsx
│   │   │   ├── TouchJoystick.tsx
│   │   │   └── EpisodeManager.tsx
│   │   └── hooks/
│   │       ├── useWebRTC.ts
│   │       └── useGamepad.ts
│   └── index.html
├── tests/
│   ├── test_envs.py
│   ├── test_teleop.py
│   ├── test_data.py
│   ├── test_train.py
│   └── test_cli.py
└── docs/
    └── plans/
```

## Core Design Decisions

### 1. Config: Pydantic v2 (not draccus)
LeRobot's draccus pinning is a major pain point. We use Pydantic v2 for:
- Runtime validation, IDE autocomplete, JSON schema generation
- CLI override via Typer + env var support
- Serialization to/from JSON/YAML natively

### 2. Dataset Format: LeRobot v3 compatible + superset
We read/write LeRobot v3 format (Parquet + MP4) for interoperability, but extend with:
- Depth image support
- Annotation metadata (quality scores, task labels)
- Provenance tracking (which teleop session, which operator)

### 3. MuJoCo Direct (not gym wrappers)
LeRobot wraps gym packages that wrap MuJoCo. We go direct:
- `mujoco.MjModel` + `mujoco.MjData` for physics
- Custom scene composition from Menagerie models
- Direct renderer access for multi-camera + depth

### 4. Teleoperation: WebRTC (not leader-follower arms)
LeRobot requires physical leader arms. We use browser:
- Anyone with a phone/laptop can teleoperate
- Gamepad API, touch controls, keyboard fallback
- Works over LAN or internet

### 5. CLI: Typer + Rich (beautiful terminal)
```bash
mimic env list                    # list available environments
mimic teleop --env pick-place     # start browser teleoperation
mimic train --policy act --data ./my_demos
mimic deploy --model ./checkpoints/best --format onnx
mimic hub push my-dataset --repo username/my-robot-demos
```

---

## Phase 1: Foundation (Weeks 1-2)
*Project skeleton, MuJoCo environments, basic rendering*

### Task 1.1: Project Initialization

**Files:**
- Create: `pyproject.toml`
- Create: `src/mimic/__init__.py`
- Create: `src/mimic/cli/__init__.py`
- Create: `src/mimic/cli/app.py`
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`

**Step 1: Initialize git repo**

```bash
cd C:/Users/tejas/OneDrive/Desktop/mimic
git init
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mimic-robotics"
version = "0.1.0"
description = "Teach your robot anything from your browser"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Tejas" },
]
keywords = ["robotics", "imitation-learning", "teleoperation", "mujoco"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "mujoco>=3.2.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
teleop = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "aiortc>=1.9.0",
    "aiohttp>=3.9.0",
]
train = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "einops>=0.8.0",
    "wandb>=0.17.0",
]
deploy = [
    "onnx>=1.16.0",
    "onnxruntime>=1.18.0",
]
hub = [
    "huggingface-hub>=0.24.0",
    "pyarrow>=16.0.0",
]
all = [
    "mimic-robotics[teleop,train,deploy,hub]",
]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

[project.scripts]
mimic = "mimic.cli.app:app"

[tool.hatch.build.targets.wheel]
packages = ["src/mimic"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Create __init__.py**

```python
"""Mimic — Teach your robot anything from your browser."""

__version__ = "0.1.0"
```

**Step 4: Create CLI skeleton**

```python
# src/mimic/cli/app.py
import typer
from rich.console import Console

app = typer.Typer(
    name="mimic",
    help="Teach your robot anything from your browser.",
    no_args_is_help=True,
)
console = Console()

@app.command()
def version():
    """Show mimic version."""
    from mimic import __version__
    console.print(f"mimic v{__version__}")

if __name__ == "__main__":
    app()
```

**Step 5: Create .gitignore, LICENSE, README**

**Step 6: Install in dev mode and verify CLI works**

```bash
pip install -e ".[dev]"
mimic version
```
Expected: `mimic v0.1.0`

**Step 7: Commit**

```bash
git add .
git commit -m "feat: initialize mimic project skeleton"
```

---

### Task 1.2: MuJoCo Environment Base Class

**Files:**
- Create: `src/mimic/envs/__init__.py`
- Create: `src/mimic/envs/base.py`
- Create: `src/mimic/envs/registry.py`
- Create: `src/mimic/config/__init__.py`
- Create: `src/mimic/config/models.py`
- Test: `tests/test_envs.py`

**Step 1: Write config models**

```python
# src/mimic/config/models.py
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    name: str
    width: int = 320
    height: int = 240
    fovy: float = 45.0

class EnvConfig(BaseModel):
    name: str
    robot: str = "panda"
    cameras: list[CameraConfig] = Field(default_factory=lambda: [
        CameraConfig(name="front", width=320, height=240),
        CameraConfig(name="wrist", width=320, height=240),
    ])
    control_hz: int = 20
    physics_hz: int = 1000
    episode_length: int = 300  # steps at control_hz
    action_space: str = "joint"  # "joint" | "cartesian"
```

**Step 2: Write base environment class**

```python
# src/mimic/envs/base.py
from __future__ import annotations
import mujoco
import numpy as np
from pathlib import Path
from mimic.config.models import EnvConfig, CameraConfig

class MimicEnv:
    """Base environment for Mimic manipulation tasks."""

    def __init__(self, config: EnvConfig, xml_path: str | Path):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._step_count = 0
        self._steps_per_control = config.physics_hz // config.control_hz

        for cam in config.cameras:
            self._renderers[cam.name] = mujoco.Renderer(
                self.model, height=cam.height, width=cam.width
            )

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        self._apply_action(action)
        for _ in range(self._steps_per_control):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._step_count >= self.config.episode_length or self._is_success()
        info = {"is_success": self._is_success()}
        return obs, reward, done, info

    def render(self, camera: str = "front") -> np.ndarray:
        renderer = self._renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def render_all_cameras(self) -> dict[str, np.ndarray]:
        return {cam.name: self.render(cam.name) for cam in self.config.cameras}

    def _apply_action(self, action: np.ndarray):
        np.copyto(self.data.ctrl[:len(action)], action)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {
            "state": np.concatenate([
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            ]),
            "joint_pos": self.data.qpos.copy(),
            "joint_vel": self.data.qvel.copy(),
        }
        for cam in self.config.cameras:
            obs[f"image.{cam.name}"] = self.render(cam.name)
        return obs

    def _reset_task(self):
        """Override in subclass to randomize object positions etc."""
        pass

    def _compute_reward(self) -> float:
        """Override in subclass."""
        return 0.0

    def _is_success(self) -> bool:
        """Override in subclass."""
        return False

    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def state_dim(self) -> int:
        return self.model.nq + self.model.nv

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
```

**Step 3: Write environment registry**

```python
# src/mimic/envs/registry.py
from __future__ import annotations
from typing import Callable
from mimic.envs.base import MimicEnv
from mimic.config.models import EnvConfig

_REGISTRY: dict[str, Callable[[EnvConfig], MimicEnv]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def make(name: str, **kwargs) -> MimicEnv:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown env '{name}'. Available: {available}")
    config = EnvConfig(name=name, **kwargs)
    return _REGISTRY[name](config)

def list_envs() -> list[str]:
    return list(_REGISTRY.keys())
```

**Step 4: Write failing test**

```python
# tests/test_envs.py
import numpy as np
from mimic.envs.base import MimicEnv
from mimic.config.models import EnvConfig, CameraConfig

def test_env_base_class_exists():
    assert MimicEnv is not None

def test_env_config_defaults():
    config = EnvConfig(name="test")
    assert config.robot == "panda"
    assert config.control_hz == 20
    assert len(config.cameras) == 2
```

**Step 5: Run tests**

```bash
pytest tests/test_envs.py -v
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add MimicEnv base class, config models, and registry"
```

---

### Task 1.3: First Manipulation Task — Pick and Place

**Files:**
- Create: `src/mimic/envs/tasks/__init__.py`
- Create: `src/mimic/envs/tasks/pick_place.py`
- Create: `src/mimic/envs/assets/scenes/tabletop.xml`
- Create: `src/mimic/envs/robots/__init__.py`
- Create: `src/mimic/envs/robots/panda.py`
- Test: `tests/test_pick_place.py`

**Step 1: Download Panda MJCF from Menagerie**

```bash
pip install mujoco_menagerie
# Or: download franka_emika_panda XMLs from github.com/google-deepmind/mujoco_menagerie
```

**Step 2: Create tabletop scene MJCF**

Create a tabletop scene XML with:
- A table surface
- A Panda arm mounted on the table
- A cube object with freejoint
- Camera definitions (front, wrist, overhead)
- Appropriate lighting

**Step 3: Create PickPlace task class**

```python
# src/mimic/envs/tasks/pick_place.py
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register
from mimic.config.models import EnvConfig
import mujoco
import numpy as np
from pathlib import Path

SCENE_XML = Path(__file__).parent.parent / "assets" / "scenes" / "tabletop_pick_place.xml"

@register("pick-place")
class PickPlaceEnv(MimicEnv):
    """Pick up a cube and place it at a target location."""

    def __init__(self, config: EnvConfig):
        super().__init__(config, SCENE_XML)
        self._target_pos = np.array([0.4, 0.2, 0.45])

    def _reset_task(self):
        # Randomize cube position on table
        cube_x = np.random.uniform(0.3, 0.5)
        cube_y = np.random.uniform(-0.2, 0.2)
        cube_joint = self.model.joint("cube_joint")
        self.data.qpos[cube_joint.qposadr[0]:cube_joint.qposadr[0]+3] = [
            cube_x, cube_y, 0.45
        ]
        # Randomize target
        self._target_pos = np.array([
            np.random.uniform(0.3, 0.5),
            np.random.uniform(-0.2, 0.2),
            0.45,
        ])

    def _compute_reward(self) -> float:
        cube_pos = self.data.body("cube").xpos.copy()
        dist = np.linalg.norm(cube_pos - self._target_pos)
        return -dist  # dense reward

    def _is_success(self) -> bool:
        cube_pos = self.data.body("cube").xpos.copy()
        return np.linalg.norm(cube_pos - self._target_pos) < 0.05
```

**Step 4: Write test**

```python
# tests/test_pick_place.py
import numpy as np
from mimic.envs.registry import make

def test_pick_place_creates():
    env = make("pick-place")
    assert env is not None
    env.close()

def test_pick_place_reset_returns_obs():
    env = make("pick-place")
    obs = env.reset()
    assert "state" in obs
    assert "image.front" in obs
    assert obs["image.front"].shape == (240, 320, 3)
    env.close()

def test_pick_place_step():
    env = make("pick-place")
    obs = env.reset()
    action = np.zeros(env.action_dim)
    obs, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "is_success" in info
    env.close()
```

**Step 5: Run tests, iterate until passing**

```bash
pytest tests/test_pick_place.py -v
```

**Step 6: Add CLI command to list and test envs**

```python
# Add to src/mimic/cli/app.py
@app.command()
def env_list():
    """List available environments."""
    from mimic.envs import registry
    from rich.table import Table
    # import tasks to trigger registration
    import mimic.envs.tasks  # noqa
    table = Table(title="Available Environments")
    table.add_column("Name")
    for name in registry.list_envs():
        table.add_row(name)
    console.print(table)
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add pick-place manipulation environment with Panda arm"
```

---

### Task 1.4: Additional Tasks — Push, Stack

**Files:**
- Create: `src/mimic/envs/tasks/push.py`
- Create: `src/mimic/envs/tasks/stack.py`
- Create: corresponding MJCF scene files
- Test: `tests/test_tasks.py`

Same pattern as Task 1.3 but for:
- **Push**: slide a cube to a target position on the table
- **Stack**: stack one cube on top of another

**Step 1-4:** Create scene XMLs, task classes, tests
**Step 5:** Run tests
**Step 6:** Commit

```bash
git commit -m "feat: add push and stack manipulation environments"
```

---

## Phase 2: Browser Teleoperation (Weeks 3-5)
*WebRTC streaming, control interface, mobile support*

### Task 2.1: FastAPI + aiortc Signaling Server

**Files:**
- Create: `src/mimic/teleop/__init__.py`
- Create: `src/mimic/teleop/server.py`
- Create: `src/mimic/teleop/video.py`
- Test: `tests/test_teleop.py`

**Step 1: Write MuJoCo video track**

```python
# src/mimic/teleop/video.py
from aiortc import VideoStreamTrack
from av import VideoFrame
import asyncio
import numpy as np

class MuJoCoVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, frame_queue: asyncio.Queue):
        super().__init__()
        self._queue = frame_queue

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        img = await self._queue.get()
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
```

**Step 2: Write signaling server**

```python
# src/mimic/teleop/server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription
from mimic.teleop.video import MuJoCoVideoTrack
from mimic.envs.base import MimicEnv
import asyncio, json
from pathlib import Path

class TeleopServer:
    def __init__(self, env: MimicEnv):
        self.env = env
        self.app = FastAPI(title="Mimic Teleoperation")
        self._pcs: set[RTCPeerConnection] = set()
        self._frame_queue = asyncio.Queue(maxsize=2)
        self._command_callback = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/api/offer")
        async def offer(request: Request):
            params = await request.json()
            sdp = RTCSessionDescription(
                sdp=params["sdp"], type=params["type"]
            )
            pc = RTCPeerConnection()
            self._pcs.add(pc)

            video = MuJoCoVideoTrack(self._frame_queue)
            pc.addTrack(video)

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(msg):
                    if self._command_callback:
                        self._command_callback(json.loads(msg))

            await pc.setRemoteDescription(sdp)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            while pc.iceGatheringState != "complete":
                await asyncio.sleep(0.05)

            return JSONResponse({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            })

        # Serve frontend static files
        frontend_dir = Path(__file__).parent / "frontend" / "dist"
        if frontend_dir.exists():
            self.app.mount("/", StaticFiles(
                directory=str(frontend_dir), html=True
            ))

    async def push_frame(self, frame: np.ndarray):
        try:
            self._frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # drop frame

    def on_command(self, callback):
        self._command_callback = callback
```

**Step 3: Write tests**

```python
# tests/test_teleop.py
def test_teleop_server_creates():
    from mimic.teleop.server import TeleopServer
    from mimic.envs.registry import make
    import mimic.envs.tasks  # noqa
    env = make("pick-place")
    server = TeleopServer(env)
    assert server.app is not None
    env.close()
```

**Step 4: Commit**

```bash
git commit -m "feat: add WebRTC teleoperation server with MuJoCo video streaming"
```

---

### Task 2.2: Control Command Handler

**Files:**
- Create: `src/mimic/teleop/commands.py`
- Create: `src/mimic/teleop/controllers/__init__.py`
- Create: `src/mimic/teleop/controllers/joint.py`
- Create: `src/mimic/teleop/controllers/cartesian.py`
- Test: `tests/test_controllers.py`

Implement joint-space and cartesian (end-effector delta) controllers.
Cartesian uses MuJoCo's built-in Jacobian (`mj_jacSite`) for IK.

**Commit:**
```bash
git commit -m "feat: add joint and cartesian teleop controllers"
```

---

### Task 2.3: React Frontend — Video + Controls

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/hooks/useWebRTC.ts`
- Create: `frontend/src/hooks/useGamepad.ts`
- Create: `frontend/src/components/VideoStream.tsx`
- Create: `frontend/src/components/Controls.tsx`
- Create: `frontend/src/components/GamepadControls.tsx`
- Create: `frontend/src/components/TouchJoystick.tsx`

**Features:**
- WebRTC video display with multi-camera toggle
- Gamepad API polling (PS4/Xbox/generic)
- Touch joystick for mobile (dual stick: one for XY, one for Z+rotation)
- Keyboard fallback (WASD + arrow keys)
- Joint slider controls
- Episode management (start/stop/save/discard)
- Dark theme, responsive layout

**Build and bundle:**
```bash
cd frontend && npm run build
# Output goes to src/mimic/teleop/frontend/dist/
```

**Commit:**
```bash
git commit -m "feat: add React frontend with gamepad, touch, and keyboard controls"
```

---

### Task 2.4: Teleop CLI Command + Integration

**Files:**
- Modify: `src/mimic/cli/app.py`
- Create: `src/mimic/teleop/loop.py` (main teleop event loop)

**Step 1: Create the main teleop loop**

The teleop loop runs:
1. Physics at 1000Hz in a thread
2. Rendering at 30Hz, pushes frames to WebRTC
3. Receives commands from data channel at 50Hz
4. Applies commands to MuJoCo data.ctrl

**Step 2: Add CLI command**

```bash
mimic teleop --env pick-place --port 8765
# Opens browser automatically, prints QR code for phone
```

Uses `rich` to print a beautiful startup banner with:
- Local URL
- QR code for phone access (via `qrcode` library)
- Connected status

**Commit:**
```bash
git commit -m "feat: add mimic teleop CLI command with auto-browser and QR code"
```

---

## Phase 3: Dataset Pipeline (Weeks 5-7)
*Recording, storage, annotation, export*

### Task 3.1: MimicDataset Class

**Files:**
- Create: `src/mimic/data/__init__.py`
- Create: `src/mimic/data/dataset.py`
- Create: `src/mimic/data/stats.py`
- Test: `tests/test_data.py`

LeRobot v3-compatible format:
- Parquet for state/action data
- MP4 for camera streams
- JSON metadata

```python
class MimicDataset:
    def create(name, env_config, path) -> MimicDataset
    def add_frame(obs, action, reward)
    def end_episode()
    def discard_episode()
    def compute_stats()
    def to_lerobot() -> LeRobotDataset  # format conversion
    def __len__() -> int
    def __getitem__(idx) -> dict
```

**Commit:**
```bash
git commit -m "feat: add MimicDataset with LeRobot v3-compatible format"
```

---

### Task 3.2: Episode Recorder (Teleop Integration)

**Files:**
- Create: `src/mimic/data/recorder.py`
- Create: `src/mimic/data/video.py`
- Modify: `src/mimic/teleop/loop.py`

The recorder hooks into the teleop loop:
- On "start recording": begins capturing frames
- On each control step: `recorder.add_frame(obs, action)`
- On "stop recording": finalizes episode, encodes video
- On "discard": throws away current episode
- Streaming MP4 encoding (encode during capture, not post-hoc)

**Commit:**
```bash
git commit -m "feat: add episode recorder with streaming MP4 encoding"
```

---

### Task 3.3: Dataset Export Formats

**Files:**
- Create: `src/mimic/data/formats.py`
- Test: `tests/test_formats.py`

Export to:
- LeRobot v3 (native)
- HDF5 (robomimic/robosuite compatibility)
- RLDS (Open X-Embodiment compatibility)

```bash
mimic data export --input ./my_demos --format lerobot --output ./lerobot_dataset
mimic data export --input ./my_demos --format hdf5 --output demo.hdf5
```

**Commit:**
```bash
git commit -m "feat: add dataset export to LeRobot, HDF5, and RLDS formats"
```

---

### Task 3.4: Dataset Browser UI

**Files:**
- Add: `frontend/src/components/DatasetBrowser.tsx`
- Add: `frontend/src/components/EpisodeViewer.tsx`
- Modify: `src/mimic/teleop/server.py` (add dataset API routes)

A web UI tab for:
- Browsing recorded episodes
- Playing back episodes (video + state trajectory)
- Deleting bad episodes
- Viewing per-feature statistics
- Quality annotation (thumbs up/down per episode)

**Commit:**
```bash
git commit -m "feat: add dataset browser UI with episode playback and annotation"
```

---

## Phase 4: Training Pipeline (Weeks 7-10)
*ACT, Diffusion Policy, evaluation*

### Task 4.1: Training Infrastructure

**Files:**
- Create: `src/mimic/train/__init__.py`
- Create: `src/mimic/train/trainer.py`
- Create: `src/mimic/train/dataloader.py`
- Create: `src/mimic/train/policies/__init__.py`
- Create: `src/mimic/train/policies/base.py`
- Create: `src/mimic/config/train.py`
- Test: `tests/test_train.py`

```python
class MimicTrainer:
    def __init__(config: TrainConfig, dataset: MimicDataset, policy: MimicPolicy)
    def train(steps: int)  # main training loop
    def evaluate(env: MimicEnv, n_episodes: int) -> dict  # sim eval
    def save_checkpoint(path)
    def load_checkpoint(path)
```

Features:
- Configurable optimizer/scheduler
- WandB logging (optional)
- Periodic sim evaluation
- Checkpoint saving
- Mixed precision (AMP)

**Commit:**
```bash
git commit -m "feat: add training infrastructure with MimicTrainer"
```

---

### Task 4.2: ACT (Action Chunking Transformer) Policy

**Files:**
- Create: `src/mimic/train/policies/act.py`
- Test: `tests/test_act.py`

Implement ACT from scratch (not wrapping LeRobot's):
- CVAE encoder (training only)
- Transformer decoder for action chunk prediction
- Multi-camera image encoding (ResNet18 backbone)
- Temporal ensembling for smooth execution

```python
class ACTPolicy(MimicPolicy):
    def __init__(config: ACTConfig)
    def forward(batch) -> loss  # training
    def predict(obs) -> action_chunk  # inference
```

**Commit:**
```bash
git commit -m "feat: implement ACT policy for imitation learning"
```

---

### Task 4.3: Diffusion Policy

**Files:**
- Create: `src/mimic/train/policies/diffusion.py`
- Test: `tests/test_diffusion.py`

Implement Diffusion Policy:
- DDPM noise scheduler
- U-Net or Transformer denoiser backbone
- Observation conditioning (image + state)
- Action chunk prediction via denoising

**Commit:**
```bash
git commit -m "feat: implement Diffusion Policy for imitation learning"
```

---

### Task 4.4: Train CLI + Eval Loop

**Files:**
- Modify: `src/mimic/cli/app.py`
- Create: `src/mimic/train/eval.py`

```bash
mimic train \
    --policy act \
    --data ./my_demos \
    --env pick-place \
    --steps 100000 \
    --batch-size 32 \
    --eval-every 5000

mimic eval \
    --checkpoint ./outputs/best.pt \
    --env pick-place \
    --episodes 50
```

Training shows a rich progress bar with:
- Loss curve (sparkline in terminal)
- Eval success rate
- ETA

**Commit:**
```bash
git commit -m "feat: add train and eval CLI commands with rich progress display"
```

---

## Phase 5: Deployment (Weeks 10-11)
*Model export, inference server*

### Task 5.1: ONNX Export

**Files:**
- Create: `src/mimic/deploy/__init__.py`
- Create: `src/mimic/deploy/export.py`
- Test: `tests/test_deploy.py`

```bash
mimic deploy export --checkpoint ./best.pt --format onnx --output model.onnx
```

**Commit:**
```bash
git commit -m "feat: add ONNX model export"
```

---

### Task 5.2: Real-time Inference Server

**Files:**
- Create: `src/mimic/deploy/inference.py`

A lightweight inference server that:
- Loads an ONNX or PyTorch model
- Subscribes to camera topics (or receives images via API)
- Outputs actions at control frequency
- Runs on Jetson-class hardware

```bash
mimic deploy serve --model ./model.onnx --hz 20
```

**Commit:**
```bash
git commit -m "feat: add real-time inference server for edge deployment"
```

---

## Phase 6: Hub + Polish + Launch (Weeks 11-14)
*HuggingFace integration, docs, demo video, launch*

### Task 6.1: HuggingFace Hub Integration

**Files:**
- Create: `src/mimic/hub/__init__.py`
- Create: `src/mimic/hub/client.py`

```bash
mimic hub push ./my_demos --repo username/pick-place-demos
mimic hub pull username/pick-place-demos --output ./demos
mimic hub push-model ./best.pt --repo username/pick-place-act
```

**Commit:**
```bash
git commit -m "feat: add HuggingFace Hub integration for datasets and models"
```

---

### Task 6.2: Documentation Site

**Files:**
- Create: `docs/` with mkdocs or similar
- Sections: Getting Started, Environments, Teleoperation, Training, Deployment, API Reference

Key pages:
- **5-minute quickstart**: `pip install mimic-robotics[all]` -> `mimic teleop --env pick-place` -> browser opens -> collect 10 demos -> `mimic train --policy act --data ./demos` -> `mimic eval`
- **Mobile teleoperation guide**: scan QR code, use phone as controller
- **Custom environment guide**: how to add your own MJCF scene

**Commit:**
```bash
git commit -m "docs: add documentation site with quickstart and guides"
```

---

### Task 6.3: Demo Video + Launch

**Deliverables:**
- 60-second demo video showing full pipeline (teleop -> train -> eval)
- GitHub README with badges, GIF, architecture diagram
- PyPI release (`pip install mimic-robotics`)
- Twitter/X thread
- Reddit posts (r/robotics, r/ROS, r/MachineLearning)
- Hacker News submission

**Launch checklist:**
- [ ] All tests passing
- [ ] README with GIF and quick start
- [ ] PyPI package published
- [ ] Demo video on YouTube
- [ ] 3 pre-collected datasets on HuggingFace Hub
- [ ] Documentation site deployed
- [ ] GitHub Actions CI green
- [ ] License file present

---

## Success Metrics

| Metric | Target (3 months) | Stretch (6 months) |
|--------|-------------------|---------------------|
| GitHub stars | 500 | 2,000 |
| PyPI downloads/month | 1,000 | 5,000 |
| Community demos on HF Hub | 10 | 100 |
| Environments | 4 | 10+ |
| Robot models | 3 | 8+ |
| Paper submission | Draft | CoRL 2026 submission |

---

## Research Paper Track (Parallel)

While building Mimic, collect data for the paper:

**Working title:** "How Many Demonstrations Does Your Robot Really Need? An Empirical Study of Data Efficiency in Foundation Model Fine-Tuning"

**Experiments to run during development:**
1. Collect N={5, 10, 25, 50, 100} demonstrations per task
2. Train ACT + Diffusion Policy on each subset
3. Measure success rate vs. number of demonstrations
4. Vary demonstration quality (expert vs novice teleoperator)
5. Test cross-task transfer (train on pick-place, eval on stack)

**Target venue:** CoRL 2026 (submission ~June 2026)

This data falls out naturally from using and testing Mimic. The paper writes itself.
