# Mimic

[![CI](https://github.com/tejasnaladala/mimic/actions/workflows/ci.yml/badge.svg)](https://github.com/tejasnaladala/mimic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Teleoperate a simulated robot arm from your browser, record the demonstrations, train an imitation-learning policy on them, and export it for inference. The whole loop runs from four CLI commands.

The interesting part is the front end: a MuJoCo Franka Panda is rendered server-side and streamed to the browser over WebRTC at 60 FPS, with control commands flowing back over a data channel in the same connection. No plugin, no local install, no GPU on the client. You drive the arm with the keyboard, a gamepad, or a phone touchscreen, and Ctrl-click anywhere in the 3D scene to send the gripper there via Jacobian IK.

<!-- TODO: add demo GIF of the teleop HUD here -->
<!-- ![Mimic teleop](docs/assets/demo.gif) -->

```
Browser (WebRTC)  -->  Record demos  -->  Train policy  -->  Deploy
   kb/gamepad/touch     Parquet + MP4      ACT / Diffusion    ONNX / PyTorch
   server-side render   LeRobot v3 layout  transformer        real-time inference
```

## Quick start

```bash
pip install "mimic-robotics[all]"

# Teleoperate (browser opens automatically)
mimic teleop --env pick-place

# Train a policy on collected demos
mimic train --policy act --data ./demo_data

# Replay a recorded episode
mimic replay --data ./demo_data --episode 0

# Evaluate in simulation
mimic eval --checkpoint outputs/best.pt --env pick-place

# Export to ONNX
mimic deploy outputs/best.pt --output model.onnx
```

## How it works

The teleop server (`FastAPI` + `aiortc`) holds the MuJoCo simulation and a render loop. On each WebRTC negotiation it opens a video track and a `commands` data channel. The render loop steps the physics, renders the active camera, and pushes frames to the video track; incoming control messages on the data channel update the joint or Cartesian targets, which a controller interpolates toward at 60 FPS so motion stays smooth even when input is bursty.

Click-to-navigate casts a ray from the camera through the clicked pixel into the MuJoCo scene, finds the hit point, and solves Jacobian IK to move the gripper there. Camera orbit, pan, and zoom are handled the same way, server-side, so the client stays a thin video surface plus an input layer.

Recording is built into the teleop UI (REC / STOP / SAVE / DISCARD). A saved episode writes numeric data (joint positions, velocities, actions, rewards) to Parquet and the camera streams to MP4, in a layout that matches LeRobot v3:

```
demo_data/
  meta/
    info.json          # env name, dims, fps, camera list
    episodes.json      # per-episode frame counts and durations
    stats.json         # normalization statistics
  data/chunk-000/
    episode_000000.parquet
  videos/chunk-000/
    front/episode_000000.mp4
    wrist/episode_000000.mp4
```

Training reads that dataset and fits one of two policies:

- **ACT** (Action Chunking Transformer) predicts a chunk of future actions per forward pass, which trains fast and runs cheaply at inference.
- **Diffusion Policy** (DDPM) denoises an action sequence conditioned on the observation, which handles multi-modal demonstrations where ACT collapses to the mean.

Export goes through `torch.onnx` to a single `.onnx` file, with an action buffer on the inference side so chunked policies stay real-time.

## Architecture

```
+------------------+    WebRTC     +------------------+    MuJoCo    +------------------+
|     Browser      | <----------> |   Teleop server  | <---------> |   Simulation     |
|  React + Vite    |   video +    |   FastAPI         |             |   MuJoCo 3.2+    |
|  gamepad / touch |   data chan  |   aiortc          |             |   Panda arm      |
+------------------+              +------------------+             +------------------+
                                         |
                                         v
        +------------------+    PyTorch   +------------------+
        |   Data pipeline  | ----------> |    Training      |
        |   Parquet + MP4  |             |   ACT / Diffusion|
        |   LeRobot v3     |             +------------------+
        +------------------+                     |
                 |                               v
                 v                       +------------------+
          +------------------+           |    Deployment    |
          |   HuggingFace    |           |   ONNX export    |
          |   Hub            |           |   inference loop |
          +------------------+           +------------------+
```

## Simulation

MuJoCo 3.2+ with the Menagerie Franka Panda (the production mesh model, not box primitives). Three tasks ship in the registry:

| Environment | Task | Action space |
|-------------|------|--------------|
| `pick-place` | Pick up the red cube, place it on the green target | 9D (7 joints + 2 gripper fingers) |
| `push` | Push the cube to a target position | 9D |
| `stack` | Stack the red cube on the blue cube | 9D |

Adding a robot or task means dropping an MJCF/URDF model and a scene XML in, then registering an environment class in `mimic.envs.registry`. `src/mimic/envs/tasks/pick_place.py` is the reference.

## Controls

| Input | Action |
|-------|--------|
| W/S, A/D, Q/E, R/F, T/G, Y/H, U/J | Joint 0-6 control |
| O / L | Open / close gripper |
| Space | Reset environment |
| M | Toggle joint / Cartesian mode |
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| Double click | Reset camera |
| Ctrl + Click | Send gripper to the clicked 3D point |

## CLI reference

| Command | Description |
|---------|-------------|
| `mimic teleop` | Start browser-based teleoperation |
| `mimic replay` | Replay a recorded episode in the viewer |
| `mimic train` | Train a policy on demonstrations |
| `mimic eval` | Evaluate a trained policy in simulation |
| `mimic deploy` | Export a checkpoint to ONNX |
| `mimic env-list` | List available environments |
| `mimic data-info` | Show dataset information |
| `mimic data-export` | Export to LeRobot / HDF5 / RLDS |
| `mimic data-stats` | Compute dataset statistics |
| `mimic hub-push` | Push a dataset to HuggingFace |
| `mimic hub-pull` | Pull a dataset from HuggingFace |
| `mimic hub-push-model` | Push a model to HuggingFace |

## Install options

```bash
pip install "mimic-robotics[all]"        # everything

pip install "mimic-robotics[teleop]"     # browser teleoperation
pip install "mimic-robotics[train]"      # training (PyTorch)
pip install "mimic-robotics[deploy]"     # ONNX export
pip install "mimic-robotics[hub]"        # HuggingFace Hub
```

`aiortc`, `mujoco`, and `torch` are version-sensitive (a past `aiortc` 1.14 change broke the WebRTC handshake until pinned), so install into a clean virtualenv.

## Development

```bash
git clone https://github.com/tejasnaladala/mimic.git
cd mimic
pip install -e ".[all,dev]"

# Tests render MuJoCo headless, so set a software GL backend.
# Linux: osmesa (CI installs libosmesa6-dev). macOS: egl or glfw.
MUJOCO_GL=osmesa python -m pytest tests/ -v   # 103 tests
ruff check src/
```

CI runs the suite on Python 3.11 and 3.12 against the osmesa backend on every push.

## License

MIT
