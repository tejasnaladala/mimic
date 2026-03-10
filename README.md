# Mimic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-103%20passed-brightgreen.svg)]()

**Teach your robot anything from your browser.**

Mimic is a complete robotics learning pipeline. Teleoperate a simulated robot arm from your browser, record demonstrations, train imitation learning policies (ACT, Diffusion Policy), and deploy them in real time. Four CLI commands cover the entire workflow.

```
Browser (WebRTC)  -->  Record Demos  -->  Train Policy  -->  Deploy
     |                      |                  |               |
  kb/gamepad/click     Parquet + MP4      ACT / Diffusion    ONNX / PyTorch
  free orbit camera    LeRobot v3 fmt     Transformer        real-time inference
```

## Quick Start

```bash
pip install "mimic-robotics[all]"

# 1. Teleoperate (browser opens automatically)
mimic teleop --env pick-place

# 2. Train a policy on collected demos
mimic train --policy act --data ./demo_data

# 3. Replay a recorded episode
mimic replay --data ./demo_data --episode 0

# 4. Evaluate in simulation
mimic eval --checkpoint outputs/best.pt --env pick-place

# 5. Export for deployment
mimic deploy outputs/best.pt --output model.onnx
```

## Features

### Teleoperation
- **Browser-native control** via WebRTC with keyboard, gamepad, or phone touch
- **Free orbit camera** with mouse drag (orbit), right-drag (pan), scroll (zoom), double-click (reset)
- **Click-to-navigate** (Ctrl+Click) places the gripper at any 3D point using Jacobian IK with MuJoCo ray casting
- **Joint + Cartesian control modes** with smooth exponential interpolation at 60 FPS
- **Cyberpunk HUD** with live joint readouts, FPS counter, episode tracker, and recording controls

### Data Pipeline
- **Episode recording** built into the teleop UI: REC, STOP, SAVE, DISCARD
- **Parquet + MP4 storage** per episode (numeric data + camera video)
- **Episode replay** in the browser viewer with full camera orbit support
- **Export formats**: LeRobot v3, HDF5 (robomimic), RLDS
- **HuggingFace Hub** integration for pushing/pulling datasets and models

### Training + Deployment
- **ACT (Action Chunking Transformers)** for fast imitation learning
- **Diffusion Policy** (DDPM-based) for complex multi-modal tasks
- **ONNX export** for edge inference with action buffering
- **Evaluation harness** with success rate and return metrics

### Simulation
- **MuJoCo 3.2+** with Menagerie Panda arm (production mesh files, not primitives)
- **Three environments**: pick-place, push, stack
- **Pluggable registry** for adding custom robots and tasks

## Architecture

```
+------------------+    WebRTC     +------------------+    MuJoCo    +------------------+
|     Browser      | <----------> |   Teleop Server  | <---------> |   Simulation     |
|  React + Vite    |              |   FastAPI         |             |   MuJoCo 3.2+    |
|  Gamepad/Touch   |              |   aiortc          |             |   Panda Arm      |
+------------------+              +------------------+             +------------------+
                                         |
                                         v
                                  +------------------+    PyTorch   +------------------+
                                  |   Data Pipeline  | ----------> |    Training      |
                                  |   Parquet + MP4  |             |   ACT / Diff.    |
                                  |   LeRobot v3     |             |   Transformer    |
                                  +------------------+             +------------------+
                                         |                                |
                                         v                                v
                                  +------------------+             +------------------+
                                  |   HuggingFace    |             |    Deployment    |
                                  |   Hub            |             |   ONNX export    |
                                  |   Datasets/Models|             |   Inference srv  |
                                  +------------------+             +------------------+
```

## Controls

| Input | Action |
|-------|--------|
| W/S, A/D, Q/E, R/F, T/G, Y/H, U/J | Joint 0-6 control |
| O / L | Open / close gripper |
| Space | Reset environment |
| M | Toggle joint/cartesian mode |
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom camera |
| Double click | Reset camera |
| Ctrl + Click | Navigate gripper to 3D point |

## CLI Reference

| Command | Description |
|---------|-------------|
| `mimic teleop` | Start browser-based teleoperation |
| `mimic replay` | Replay a recorded episode in the viewer |
| `mimic train` | Train a policy on demonstrations |
| `mimic eval` | Evaluate a trained policy in simulation |
| `mimic deploy` | Export model to ONNX |
| `mimic env-list` | List available environments |
| `mimic data-info` | Show dataset information |
| `mimic data-export` | Export to LeRobot/HDF5/RLDS |
| `mimic data-stats` | Compute dataset statistics |
| `mimic hub-push` | Push dataset to HuggingFace |
| `mimic hub-pull` | Pull dataset from HuggingFace |
| `mimic hub-push-model` | Push model to HuggingFace |

## Environments

| Environment | Task | Action Space |
|-------------|------|-------------|
| `pick-place` | Pick up red cube, place on green target | 9D (7 joints + 2 gripper fingers) |
| `push` | Push cube to target position | 9D |
| `stack` | Stack red cube on blue cube | 9D |

## Data Format

Episodes are stored in a LeRobot v3-compatible layout:

```
demo_data/
  meta/
    info.json          # env name, dims, fps, camera list
    episodes.json      # per-episode frame counts and durations
    stats.json         # computed normalization statistics
  data/
    chunk-000/
      episode_000000.parquet   # joint positions, velocities, actions, rewards
      episode_000001.parquet
  videos/
    chunk-000/
      front/
        episode_000000.mp4     # front camera recording
      wrist/
        episode_000000.mp4     # wrist camera recording
```

## Adding Custom Robots

1. Create an MJCF/URDF model with the robot's meshes
2. Define a scene XML that includes the robot and task objects
3. Register an environment class in the `mimic.envs.registry`
4. Run `mimic teleop --env your-env-name`

See `src/mimic/envs/tasks/pick_place.py` for a reference implementation.

## Installation Options

```bash
# Full install (everything)
pip install "mimic-robotics[all]"

# Selective install
pip install "mimic-robotics[teleop]"    # Browser teleoperation only
pip install "mimic-robotics[train]"     # Training (PyTorch)
pip install "mimic-robotics[deploy]"    # ONNX export
pip install "mimic-robotics[hub]"       # HuggingFace Hub integration
```

## Development

```bash
git clone https://github.com/nautilus4707/mimic.git
cd mimic
pip install -e ".[all,dev]"
python -m pytest tests/ -v   # 103 tests
```

## License

MIT
