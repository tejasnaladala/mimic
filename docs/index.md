# Mimic

## Teach your robot anything from your browser

Mimic is an open-source framework for robot imitation learning. Teleoperate a simulated robot arm through your browser, collect demonstrations, train a policy, and deploy it -- all from the command line.

---

## Key Features

- **Browser-based teleoperation** -- Control a MuJoCo robot arm via WebRTC with keyboard, gamepad, or touch input. No additional hardware required.
- **Built-in manipulation tasks** -- Pick-and-place, push, and stack environments with a Franka Panda arm, ready out of the box.
- **State-of-the-art policies** -- Train with Action Chunking Transformer (ACT) or Diffusion Policy using a single CLI command.
- **LeRobot-compatible data** -- Demonstrations stored in Parquet + MP4 format, exportable to LeRobot, HDF5, and RLDS.
- **One-command deployment** -- Export trained models to ONNX for real-time inference on CPU or edge devices.
- **Extensible** -- Create custom environments by subclassing `MimicEnv` and registering them with a decorator.

---

## Quick Install

```bash
pip install mimic-robotics[all]
```

Or install only what you need:

```bash
# Core + teleoperation
pip install mimic-robotics[teleop]

# Core + training
pip install mimic-robotics[train]

# Core + deployment
pip install mimic-robotics[deploy]
```

Requires Python 3.11+.

---

## Quick Example

```bash
# 1. Start teleoperation in the pick-place environment
mimic teleop --env pick-place

# 2. Collect demonstrations via the browser UI

# 3. Train an ACT policy on your demos
mimic train --policy act --data ./demos

# 4. Evaluate the trained policy
mimic eval --checkpoint outputs/final.pt --env pick-place

# 5. Export to ONNX for deployment
mimic deploy outputs/final.pt --output model.onnx
```

See the [Quickstart](quickstart.md) for a complete walkthrough.

---

## CLI Commands

| Command | Description |
|---|---|
| `mimic teleop` | Start browser-based teleoperation |
| `mimic train` | Train a policy on collected demonstrations |
| `mimic eval` | Evaluate a trained policy in simulation |
| `mimic deploy` | Export a model to ONNX format |
| `mimic env-list` | List available environments |
| `mimic data-info <path>` | Show dataset metadata |
| `mimic data-stats <path>` | Display dataset statistics |
| `mimic data-export <src> <dst>` | Export dataset to another format |
| `mimic version` | Show mimic version |
