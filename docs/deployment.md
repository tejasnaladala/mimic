# Deployment

Mimic supports exporting trained policies to ONNX format for efficient inference on CPU or edge devices.

## ONNX Export

Export a trained checkpoint to ONNX:

```bash
mimic deploy outputs/final.pt
```

This creates `model.onnx` in the current directory. The export process auto-detects whether the checkpoint is an ACT or Diffusion policy.

Options:

```bash
# Custom output path
mimic deploy outputs/final.pt --output my_policy.onnx
```

The ONNX model wraps the policy's `predict()` method:

- **Input:** `state` -- flat float32 array of shape `[batch_size, obs_dim]`
- **Output:** `actions` -- float32 array of shape `[batch_size, chunk_size, action_dim]`

### Export from Python

```python
from mimic.deploy.export import export_to_onnx

path = export_to_onnx(
    checkpoint_path="outputs/final.pt",
    output_path="model.onnx",
)
print(f"Exported to {path}")
```

### Verify the Export

```python
from mimic.deploy.export import verify_onnx

ok = verify_onnx("model.onnx", obs_dim=18)
# Runs a dummy inference and checks output shape
```

## Inference Server

The `InferenceServer` class provides a simple interface for running inference with either ONNX or PyTorch models. It handles action chunking automatically using temporal ensembling: it predicts a full action chunk, then returns one action at a time from a buffer.

```python
from mimic.deploy.inference import InferenceServer

# Load ONNX model
server = InferenceServer("model.onnx")

# Or load a PyTorch checkpoint directly
server = InferenceServer("outputs/final.pt")

# Auto-detect backend from file extension
server = InferenceServer("model.onnx", backend="auto")  # default
```

### Predict Actions

```python
import numpy as np

state = np.random.randn(18).astype(np.float32)  # current observation

# Returns a single action (from the buffered action chunk)
action = server.predict(state)  # shape: [action_dim]

# Call reset() when starting a new episode to clear the action buffer
server.reset()
```

### Real-Time Control Loop

A complete real-time control loop using the inference server:

```python
import time

import numpy as np

from mimic.deploy.inference import InferenceServer
from mimic.envs.registry import make

# Setup
env = make("pick-place")
server = InferenceServer("model.onnx")
control_hz = 20

# Run episodes
for episode in range(10):
    obs = env.reset()
    server.reset()  # clear action buffer
    done = False

    while not done:
        t_start = time.time()

        # Get action from policy
        action = server.predict(obs["state"])

        # Step environment
        obs, reward, done, info = env.step(action)

        # Maintain control frequency
        elapsed = time.time() - t_start
        sleep_time = (1.0 / control_hz) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Episode {episode}: success={info['is_success']}")

env.close()
```

### Backend Options

| Backend | File Extension | Dependencies | Notes |
|---|---|---|---|
| `onnx` | `.onnx` | `onnxruntime` | Fast CPU inference, no PyTorch needed |
| `torch` | `.pt` | `torch` | Uses PyTorch, supports GPU |
| `auto` | any | auto-detected | Picks backend from file extension |

## ONNX Runtime Tips

For production deployment with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

# Create session with optimizations
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)

# Run inference
state = np.random.randn(1, 18).astype(np.float32)
actions = sess.run(None, {"state": state})[0]
# actions shape: [1, chunk_size, action_dim]

# Use first action
action = actions[0, 0, :]
```

For GPU inference, use `CUDAExecutionProvider`:

```python
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

## Integration Checklist

1. **Export:** `mimic deploy outputs/final.pt --output model.onnx`
2. **Verify:** Run `verify_onnx()` to check the export
3. **Integrate:** Use `InferenceServer` or raw ONNX Runtime in your control loop
4. **Timing:** Match the control frequency to the training data FPS (default 20 Hz)
5. **Reset:** Call `server.reset()` at the start of each episode to clear the action buffer
