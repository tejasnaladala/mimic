# Teleoperation

Mimic uses WebRTC to stream a live MuJoCo simulation to your browser. You control the robot arm and record demonstrations without installing any additional software.

## Starting the Server

```bash
mimic teleop --env pick-place
```

| Option | Default | Description |
|---|---|---|
| `--env` | `pick-place` | Environment name |
| `--port` | `8765` | Server port |
| `--mode` | `joint` | Control mode: `joint` or `cartesian` |
| `--no-browser` | `False` | Don't auto-open the browser |

The server runs on `http://localhost:<port>` using FastAPI + uvicorn. WebRTC provides low-latency video streaming via `aiortc`.

## Keyboard Controls

### Joint Mode (`--mode joint`)

Each key pair controls one joint with a position delta:

| Key | Joint | Direction |
|---|---|---|
| `W` / `S` | Joint 0 (shoulder) | +/- |
| `A` / `D` | Joint 1 (shoulder) | +/- |
| `Q` / `E` | Joint 2 (elbow) | +/- |
| `R` / `F` | Joint 3 (elbow) | +/- |
| `T` / `G` | Joint 4 (wrist) | +/- |
| `Y` / `H` | Joint 5 (wrist) | +/- |
| `U` / `J` | Joint 6 (wrist) | +/- |
| `O` / `L` | Gripper | open / close |
| `Space` | -- | Reset environment |

### Cartesian Mode (`--mode cartesian`)

In cartesian mode, commands are translated to end-effector position deltas using Jacobian-based inverse kinematics. The same key layout applies but controls Cartesian XYZ movement of the end-effector instead of individual joints.

The cartesian controller uses damped least-squares IK computed from the MuJoCo Jacobian at the `grip_site` (or `panda_hand` body as fallback).

## Gamepad Support

The browser UI accepts standard Gamepad API input. Connect a game controller and it will be detected automatically:

- **Left stick:** XY end-effector movement (cartesian mode) or Joint 0/1 deltas (joint mode)
- **Right stick:** Z movement / rotation
- **Triggers:** Gripper open/close
- **Buttons:** Reset, start/stop recording

## Mobile / Touch Controls

On mobile devices, the browser UI provides on-screen touch controls:

- **Drag** on the video feed to send directional commands
- **Pinch** to control the gripper
- **Tap** the reset button to reset the environment

## Recording Demonstrations

### Browser UI

1. Click **Connect** to establish the WebRTC connection.
2. Click **Start Recording** to begin capturing a demonstration.
3. Teleoperate the robot to complete the task.
4. Click **Stop Recording** to finalize and save the episode.
5. Use `Space` to reset the environment for the next episode.

### Programmatic Recording

Use `EpisodeRecorder` to record demonstrations from code:

```python
from mimic.data.dataset import MimicDataset
from mimic.data.recorder import EpisodeRecorder
from mimic.envs.registry import make

env = make("pick-place")
dataset = MimicDataset.create(
    path="./demos",
    env_name="pick-place",
    action_dim=env.action_dim,
    state_dim=env.state_dim,
    camera_names=["front", "wrist"],
)

recorder = EpisodeRecorder(dataset, env)

obs = env.reset()
recorder.start_recording()

for step in range(300):
    action = get_action_from_somewhere(obs)
    recorder.record_frame(obs, action)
    obs, reward, done, info = env.step(action)
    if done:
        break

episode_idx = recorder.stop_recording()
print(f"Saved episode {episode_idx}")
```

### Data Format

Demonstrations are stored in a directory with this structure:

```
demos/
  meta/
    info.json          # Dataset metadata (env name, dimensions, counts)
    stats.json         # Per-feature statistics (computed on demand)
  data/
    chunk-000/
      episode_000000.parquet   # Numeric data (state, action, reward)
      episode_000001.parquet
  videos/
    chunk-000/
      front/
        episode_000000.mp4     # Camera video (H.264)
      wrist/
        episode_000000.mp4
```

Each Parquet file contains columns: `episode_index`, `frame_index`, `timestamp`, `task`, `state`, `joint_pos`, `joint_vel`, `action`, `reward`, `done`.

### Exporting Data

Export to other formats:

```bash
# Export to LeRobot format
mimic data-export ./demos ./demos_lerobot --format lerobot

# Export to HDF5
mimic data-export ./demos ./demos.hdf5 --format hdf5

# Export to RLDS (TensorFlow Datasets)
mimic data-export ./demos ./demos_rlds --format rlds
```

## Architecture

The teleoperation stack consists of:

- **`TeleopLoop`** (`mimic.teleop.loop`) -- Main event loop combining server and render loop
- **`TeleopServer`** (`mimic.teleop.server`) -- FastAPI + WebRTC server with data channels
- **`CommandRouter`** (`mimic.teleop.commands`) -- Routes incoming commands to the active controller
- **`JointController`** (`mimic.teleop.controllers.joint`) -- Joint-space control with delta commands
- **`CartesianController`** (`mimic.teleop.controllers.cartesian`) -- End-effector control with Jacobian IK
- **`MuJoCoVideoTrack`** (`mimic.teleop.video`) -- WebRTC video track fed from MuJoCo renderer
