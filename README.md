# Mimic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**Teach your robot anything from your browser.**

Mimic is a Python robotics library for teaching robots new skills through browser-based
teleoperation and imitation learning. Control a simulated robot from your browser via WebRTC,
record demonstrations, train state-of-the-art policies, and deploy learned behaviors back to
your robot.

## Features

- **Browser-based teleoperation** -- Control robots directly from your browser via WebRTC
- **MuJoCo simulation** -- High-fidelity physics simulation with MuJoCo
- **ACT + Diffusion Policy training** -- Train with state-of-the-art imitation learning algorithms
- **ONNX export** -- Export trained policies to ONNX for fast, portable inference
- **HuggingFace Hub integration** -- Share and download datasets and models from the Hub

## Quick Start

### Install

```bash
pip install mimic-robotics
```

Install with all optional dependencies:

```bash
pip install "mimic-robotics[all]"
```

### Teleoperate

Open your browser and control a simulated robot to demonstrate a task:

```bash
mimic teleop --env PegInsertion
```

### Train

Train an imitation learning policy on collected demonstrations:

```bash
mimic train --data ./data/PegInsertion --policy act
```

### Evaluate

Evaluate a trained policy in simulation:

```bash
mimic eval --policy ./checkpoints/act_best.onnx --env PegInsertion
```

## Architecture

```
+------------+     WebRTC      +------------+     MuJoCo     +------------+
|  Browser   | <------------> |   Server   | <-----------> | Simulation |
| (teleop UI)|                | (FastAPI)  |               |  (MuJoCo)  |
+------------+                +------------+               +------------+
                                    |
                                    v
                              +------------+     ONNX       +------------+
                              |  Training  | ------------> |   Deploy   |
                              | (ACT / DP) |               | (realtime) |
                              +------------+               +------------+
                                    |
                                    v
                              +------------+
                              |  HF Hub    |
                              | (datasets) |
                              +------------+
```

## Development

```bash
git clone https://github.com/your-org/mimic.git
cd mimic
pip install -e ".[dev]"
pytest -v
ruff check src/
```

## License

MIT
