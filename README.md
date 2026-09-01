# Mimic

[![CI](https://github.com/tejasnaladala/mimic/actions/workflows/ci.yml/badge.svg)](https://github.com/tejasnaladala/mimic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**Browser-based teleoperation and imitation learning for a simulated Franka Panda arm.**

Mimic runs MuJoCo simulation, rendering, inverse kinematics, data capture, and policy training on
the host machine. The browser receives WebRTC video and sends keyboard, gamepad, touch, and
Ctrl-click commands over a data channel.

The browser client requires no Mimic installation or plugin. The host must install Mimic and its
teleoperation dependencies; training and ONNX export require their corresponding optional
dependencies. `TeleopConfig.fps` defaults to `60` and paces the server render loop. That value is
a configuration target, not a measured end-to-end frame rate.

The quickstart uses five CLI operations: teleoperate and record, replay, train, evaluate, and
export.

```
Teleoperate + record  -->  Replay  -->  Train  -->  Evaluate  -->  Export
Browser + WebRTC       MP4 viewer    ACT /       MuJoCo          ONNX
Parquet + MP4                        Diffusion   simulation
```

## Quick start

```bash
git clone https://github.com/tejasnaladala/mimic.git
cd mimic
pip install -e ".[all]"

# Teleoperate (browser opens automatically)
mimic teleop --env pick-place

# Replay a recorded episode
mimic replay --data ./demo_data --episode 0

# Train a policy on collected demos
mimic train --policy act --data ./demo_data

# Evaluate in simulation
mimic eval --checkpoint outputs/final.pt --env pick-place

# Export to ONNX
mimic deploy outputs/final.pt --output model.onnx
```

## How it works

The teleop server (`FastAPI` + `aiortc`) holds the MuJoCo simulation and render loop. Each WebRTC
connection gets a video track and a `commands` data channel. Incoming messages update joint or
Cartesian targets; once per render-loop iteration, the controller advances toward the target,
steps the simulation, renders the active camera, and queues a frame for the video track. The loop
sleeps for `1 / config.fps` after that work, so delivered frame rate depends on host and transport
time.

For click-to-navigate, the server casts a ray from the camera through the clicked pixel, finds the
MuJoCo hit point, and applies Jacobian IK toward it. Camera orbit, pan, and zoom also execute on
the server; the browser decodes video and sends input events.

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

Action chunks are indexed within each episode. Near an episode boundary, the remaining positions are zero-padded and masked out of both the model context and training loss; a chunk never borrows frames from the next episode.

- **ACT** (Action Chunking Transformer) uses a conditional VAE and Transformer decoder to predict
  a chunk of future actions.
- **Diffusion Policy** (DDPM) predicts a chunk by iteratively denoising an action sequence
  conditioned on the observation.

Export goes through `torch.onnx` to a single `.onnx` file. The inference wrapper buffers a
predicted action chunk and returns its actions one at a time.

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
          |   Hub            |           | buffered actions |
          +------------------+           +------------------+
```

## Simulation

Mimic requires MuJoCo 3.2+ and bundles the mesh-based Franka Panda model used by its three
registered tasks:

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

Python 3.11 or newer is required. A separate virtual environment keeps the teleoperation,
training, and deployment dependency groups isolated from other projects.

## Development

```bash
git clone https://github.com/tejasnaladala/mimic.git
cd mimic
pip install -e ".[all,dev]"

# Tests render MuJoCo headless, so set a software GL backend.
# Linux: osmesa (CI installs libosmesa6-dev). macOS: egl or glfw.
MUJOCO_GL=osmesa python -m pytest tests/ -v
ruff check src/
```

CI runs the suite with the osmesa backend on Python 3.11 and 3.12 for pushes and pull requests
targeting `main` or `master`.

## License

MIT
