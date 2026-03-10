# API Reference

This page provides an overview of the key modules and classes in Mimic.

## Environments

### `mimic.envs.base.MimicEnv`

Base class for all Mimic environments. Wraps a MuJoCo model and provides the standard `reset()` / `step()` / `render()` interface.

```python
from mimic.envs.base import MimicEnv
```

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: EnvConfig, xml_path: str \| Path)` | Load MuJoCo model and create renderers |
| `reset` | `() -> dict[str, np.ndarray]` | Reset environment, return observation |
| `step` | `(action: np.ndarray) -> (obs, reward, done, info)` | Step with action array |
| `render` | `(camera: str = "front") -> np.ndarray` | Render camera as RGB array |
| `render_all_cameras` | `() -> dict[str, np.ndarray]` | Render all configured cameras |
| `close` | `()` | Clean up renderers |

| Property | Type | Description |
|---|---|---|
| `action_dim` | `int` | Number of actuators (`model.nu`) |
| `state_dim` | `int` | `model.nq + model.nv` |

Override in subclasses: `_reset_task()`, `_compute_reward()`, `_is_success()`.

### `mimic.envs.registry`

Environment registration and creation.

```python
from mimic.envs.registry import register, make, list_envs
```

| Function | Signature | Description |
|---|---|---|
| `register` | `(name: str)` | Decorator to register an environment class |
| `make` | `(name: str, **kwargs) -> MimicEnv` | Create environment by name |
| `list_envs` | `() -> list[str]` | List registered environment names |

### Built-in Environments

| Module | Class | Registered Name |
|---|---|---|
| `mimic.envs.tasks.pick_place` | `PickPlaceEnv` | `pick-place` |
| `mimic.envs.tasks.push` | `PushEnv` | `push` |
| `mimic.envs.tasks.stack` | `StackEnv` | `stack` |

---

## Policies

### `mimic.train.policies.base.MimicPolicy`

Abstract base class for all policies. Extends `torch.nn.Module`.

```python
from mimic.train.policies.base import MimicPolicy
```

| Method | Signature | Description |
|---|---|---|
| `forward` | `(batch: dict) -> dict` | Training forward pass, returns dict with `"loss"` |
| `predict` | `(obs: dict) -> torch.Tensor` | Inference: predict action(s) from observation |
| `get_optimizer` | `(lr: float = 1e-4) -> Optimizer` | Create AdamW optimizer |
| `save` | `(path: str)` | Save checkpoint (state dict + config) |
| `load` | `(path: str) -> MimicPolicy` | Class method to load from checkpoint |

### `mimic.train.policies.act.ACTPolicy`

Action Chunking Transformer. Conditional VAE architecture for multi-modal action prediction.

```python
from mimic.train.policies.act import ACTPolicy

policy = ACTPolicy(obs_dim=18, action_dim=9)
```

Constructor parameters: `obs_dim`, `action_dim`, `action_chunk_size=10`, `hidden_dim=256`, `n_heads=4`, `n_layers=4`, `latent_dim=32`, `dropout=0.1`, `kl_weight=10.0`.

### `mimic.train.policies.diffusion.DiffusionPolicy`

DDPM-based action generation via iterative denoising.

```python
from mimic.train.policies.diffusion import DiffusionPolicy

policy = DiffusionPolicy(obs_dim=18, action_dim=9)
```

Constructor parameters: `obs_dim`, `action_dim`, `action_chunk_size=10`, `hidden_dim=256`, `n_layers=4`, `n_diffusion_steps=100`, `noise_schedule="cosine"`.

---

## Training

### `mimic.train.trainer.MimicTrainer`

Unified training loop for all policies.

```python
from mimic.train.trainer import MimicTrainer

trainer = MimicTrainer(policy, config, dataset_path, output_dir="outputs")
trainer.train()
```

| Method / Property | Description |
|---|---|
| `train(steps=None)` | Run training loop |
| `save_checkpoint(name)` | Save current policy checkpoint |
| `current_step` | Current training step |
| `recent_loss` | Mean loss over last 100 steps |
| `device` | Resolved torch device |

### `mimic.train.eval.evaluate_policy`

Run policy evaluation in simulation.

```python
from mimic.train.eval import evaluate_policy

results = evaluate_policy(policy, env, n_episodes=10, device="cpu")
# Returns: {"success_rate", "mean_return", "std_return", "n_episodes"}
```

---

## Data

### `mimic.data.dataset.MimicDataset`

Dataset for storing and loading robot demonstrations in Parquet + MP4 format.

```python
from mimic.data.dataset import MimicDataset

# Create a new dataset
dataset = MimicDataset.create(
    path="./demos",
    env_name="pick-place",
    action_dim=9,
    state_dim=18,
    camera_names=["front", "wrist"],
)

# Load existing dataset
dataset = MimicDataset("./demos")
```

| Method | Description |
|---|---|
| `create(path, env_name, ...)` | Class method to create new empty dataset |
| `add_frame(obs, action, reward, done)` | Add frame to current episode |
| `end_episode(task="default")` | Save current episode, return index |
| `discard_episode()` | Discard current episode |
| `load_episode(idx)` | Load episode as PyArrow Table |
| `compute_stats()` | Compute per-feature statistics |
| `metadata` | Dataset metadata dict |
| `num_episodes` | Number of saved episodes |

### `mimic.data.recorder.EpisodeRecorder`

Records teleoperation episodes into a `MimicDataset`.

```python
from mimic.data.recorder import EpisodeRecorder

recorder = EpisodeRecorder(dataset, env)
recorder.start_recording()
recorder.record_frame(obs, action)
episode_idx = recorder.stop_recording()
```

---

## Teleoperation

### `mimic.teleop.loop.TeleopLoop`

Main teleoperation event loop. Combines WebRTC server and MuJoCo render loop.

```python
from mimic.teleop.loop import TeleopLoop

loop = TeleopLoop(env, config)
loop.run(open_browser=True)
```

### `mimic.teleop.server.TeleopServer`

FastAPI + WebRTC server. Serves the browser UI and streams video via `aiortc`.

### `mimic.teleop.commands.CommandRouter`

Routes incoming commands to `JointController` or `CartesianController`.

### `mimic.teleop.controllers.joint.JointController`

Joint-space control with delta commands.

### `mimic.teleop.controllers.cartesian.CartesianController`

End-effector Cartesian control using MuJoCo Jacobian IK.

---

## Deployment

### `mimic.deploy.export`

ONNX export utilities.

```python
from mimic.deploy.export import export_to_onnx, verify_onnx

export_to_onnx("outputs/final.pt", "model.onnx")
verify_onnx("model.onnx", obs_dim=18)
```

### `mimic.deploy.inference.InferenceServer`

Lightweight inference server with temporal ensembling (action buffering).

```python
from mimic.deploy.inference import InferenceServer

server = InferenceServer("model.onnx")
action = server.predict(state)
server.reset()
```

---

## Configuration

### `mimic.config.models`

Pydantic models for configuration.

| Class | Description |
|---|---|
| `CameraConfig` | Camera settings (name, width, height, fovy) |
| `EnvConfig` | Environment settings (robot, cameras, control/physics Hz, episode length) |
| `TrainConfig` | Training settings (policy, batch size, lr, steps, device) |
| `TeleopConfig` | Teleoperation settings (host, port, control mode, camera, fps) |

---

## CLI

Entry point: `mimic.cli.app:app` (Typer application).

Commands: `version`, `env-list`, `teleop`, `train`, `eval`, `deploy`, `data-info`, `data-stats`, `data-export`.

See `mimic --help` for full usage.
