# Mimic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-99%20passed-brightgreen.svg)]()

**Teach your robot anything from your browser.**

Mimic is a complete robotics learning pipeline: teleoperate a simulated robot from your browser, record demonstrations, train state-of-the-art imitation learning policies, and deploy them in real-time. The entire workflow runs with 4 CLI commands.

```
Browser (WebRTC)  -->  Record Demos  -->  Train Policy  -->  Deploy
     |                      |                  |               |
  gamepad/touch/kb    Parquet + MP4      ACT / Diffusion    ONNX / PyTorch
  mobile support      LeRobot v3 fmt     Transformer        real-time inference
```

## Quick Start

```bash
pip install "mimic-robotics[all]"

# 1. Teleoperate -- browser opens automatically
mimic teleop --env pick-place

# 2. Train a policy on collected demos
mimic train --policy act --data ./demos

# 3. Evaluate in simulation
mimic eval --checkpoint outputs/best.pt --env pick-place

# 4. Export for deployment
mimic deploy outputs/best.pt --output model.onnx
```

## Features

- **Browser teleoperation** -- Control robots via WebRTC with keyboard, gamepad, or phone touch controls
- **MuJoCo simulation** -- Panda arm with pick-place, push, and stack tasks out of the box
- **ACT + Diffusion Policy** -- Train with Action Chunking Transformers or DDPM-based diffusion
- **LeRobot-compatible data** -- Parquet + MP4 storage, export to LeRobot v3, HDF5, or RLDS
- **ONNX deployment** -- Export models for fast edge inference with action buffering
- **HuggingFace Hub** -- Push/pull datasets and models with `mimic hub-push` / `mimic hub-pull`
- **Modular install** -- Only install what you need: `pip install mimic-robotics[teleop]`

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

## CLI Reference

| Command | Description |
|---------|-------------|
| `mimic teleop` | Start browser-based teleoperation |
| `mimic train` | Train a policy on demonstrations |
| `mimic eval` | Evaluate a trained policy in sim |
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
| `pick-place` | Pick up red cube, place on green target | 8D (7 joints + gripper) |
| `push` | Push cube to target position | 8D |
| `stack` | Stack red cube on blue cube | 8D |

## Installation Options

```bash
# Full install (everything)
pip install "mimic-robotics[all]"

# Selective install
pip install "mimic-robotics[teleop]"    # Browser teleoperation
pip install "mimic-robotics[train]"     # Training (PyTorch)
pip install "mimic-robotics[deploy]"    # ONNX export
pip install "mimic-robotics[hub]"       # HuggingFace Hub
```

## Development

```bash
git clone https://github.com/tejasag/mimic.git
cd mimic
pip install -e ".[all,dev]"
python -m pytest tests/ -v
```

## License

MIT
