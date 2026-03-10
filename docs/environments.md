# Environments

Mimic ships with tabletop manipulation environments using a Franka Panda arm in MuJoCo. All environments share a common observation and action interface.

## Available Environments

List registered environments:

```bash
mimic env-list
```

### pick-place

Pick up a red cube and place it at a green target location.

- **Action space:** 9-dim (7 arm joint targets + 2 gripper finger targets)
- **Observations:** `state` (qpos + qvel), `joint_pos`, `joint_vel`, `image.front`, `image.wrist`
- **Reward:** Negative Euclidean distance from cube center to target center
- **Success:** Cube center within 0.05m of target

```bash
mimic teleop --env pick-place
```

### push

Push a cube across the table to a target position without lifting it.

- **Action space:** 9-dim (7 arm joint targets + 2 gripper finger targets)
- **Observations:** `state` (qpos + qvel), `joint_pos`, `joint_vel`, `image.front`, `image.wrist`
- **Reward:** Negative 2D (XY) distance from cube center to target
- **Success:** Cube center within 0.04m of target in XY plane

```bash
mimic teleop --env push
```

### stack

Pick up a red cube and stack it on top of a blue cube.

- **Action space:** 9-dim (7 arm joint targets + 2 gripper finger targets)
- **Observations:** `state` (qpos + qvel), `joint_pos`, `joint_vel`, `image.front`, `image.wrist`
- **Reward:** Negative distance from red cube to stacking position (above blue cube)
- **Success:** Red cube within 0.05m of stack target AND red cube z > blue cube z

```bash
mimic teleop --env stack
```

## Common Configuration

All environments use `EnvConfig` from `mimic.config.models`:

```python
from mimic.config.models import EnvConfig

config = EnvConfig(
    name="pick-place",
    robot="panda",
    cameras=[
        CameraConfig(name="front", width=320, height=240),
        CameraConfig(name="wrist", width=320, height=240),
    ],
    control_hz=20,
    physics_hz=1000,
    episode_length=300,
    action_space="joint",  # "joint" or "cartesian"
)
```

## Creating Custom Environments

### 1. Subclass MimicEnv

Create a new file in your project:

```python
from pathlib import Path

import numpy as np

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register

SCENE_XML = Path(__file__).parent / "assets" / "my_scene.xml"


@register("my-task")
class MyTaskEnv(MimicEnv):
    """Custom manipulation task."""

    def __init__(self, config: EnvConfig):
        super().__init__(config, SCENE_XML)

    def _reset_task(self):
        """Randomize initial conditions for each episode."""
        # Set object positions, randomize targets, etc.
        pass

    def _compute_reward(self) -> float:
        """Return a scalar reward for the current state."""
        return 0.0

    def _is_success(self) -> bool:
        """Return True when the task is solved."""
        return False
```

The `@register("my-task")` decorator adds your environment to the global registry so it can be used with `mimic teleop --env my-task`.

### 2. MimicEnv Interface

The base class provides:

| Method | Description |
|---|---|
| `reset() -> dict` | Reset environment, return initial observation |
| `step(action) -> (obs, reward, done, info)` | Step with action array |
| `render(camera) -> np.ndarray` | Render a camera view as RGB array |
| `render_all_cameras() -> dict` | Render all configured cameras |
| `close()` | Clean up renderers |

Properties:

| Property | Description |
|---|---|
| `action_dim` | Number of actuators (from `model.nu`) |
| `state_dim` | `model.nq + model.nv` |

Override these methods in your subclass:

| Method | Purpose |
|---|---|
| `_reset_task()` | Randomize task (object positions, targets) |
| `_compute_reward() -> float` | Dense reward signal |
| `_is_success() -> bool` | Task completion check |

### 3. MJCF Scene Format

Environments use MuJoCo MJCF XML files for scene definition. Place your scene files in an `assets/scenes/` directory:

```xml
<mujoco model="my_task">
  <include file="../robots/panda.xml"/>

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.2">
      <geom type="box" size="0.4 0.4 0.2" rgba="0.9 0.85 0.7 1"/>
    </body>

    <!-- Manipulable object with freejoint -->
    <body name="cube" pos="0.5 0 0.45">
      <freejoint name="cube_joint"/>
      <geom type="box" size="0.025 0.025 0.025"
            rgba="0.8 0.2 0.2 1" mass="0.05"/>
    </body>

    <!-- Visual target marker -->
    <body name="target" pos="0.5 0.15 0.45">
      <geom type="cylinder" size="0.03 0.002"
            rgba="0.2 0.8 0.2 0.5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <!-- 7 arm joints + 2 gripper fingers -->
    <position name="j0" joint="panda_joint1" kp="100"/>
    <!-- ... -->
  </actuator>
</mujoco>
```

### 4. Using Your Environment

Once registered, your environment works with all CLI commands:

```bash
# Teleoperate
mimic teleop --env my-task

# Train
mimic train --policy act --data ./demos --env my-task

# Evaluate
mimic eval --checkpoint outputs/final.pt --env my-task
```

Or use it programmatically:

```python
from mimic.envs.registry import make

env = make("my-task")
obs = env.reset()

action = env.data.ctrl.copy()  # current actuator values
obs, reward, done, info = env.step(action)

print(f"Reward: {reward}, Success: {info['is_success']}")
env.close()
```
