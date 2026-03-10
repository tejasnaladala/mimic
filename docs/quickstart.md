# Quickstart

Get from zero to a trained policy in five minutes.

## 1. Install Mimic

```bash
pip install mimic-robotics[all]
```

Verify the installation:

```bash
mimic version
```

## 2. Start Teleoperation

Launch the browser-based teleoperation interface:

```bash
mimic teleop --env pick-place
```

This starts a WebRTC server on `http://localhost:8765` and opens your browser automatically. You will see a live MuJoCo simulation of a Franka Panda arm with a pick-and-place task.

Options:

```bash
# Use a different environment
mimic teleop --env push

# Change the server port
mimic teleop --env pick-place --port 9000

# Use cartesian end-effector control instead of joint control
mimic teleop --env pick-place --mode cartesian

# Don't auto-open the browser
mimic teleop --env pick-place --no-browser
```

## 3. Collect Demonstrations

Once the teleoperation UI is open in your browser:

1. Click **Connect** to establish the WebRTC connection.
2. Use keyboard controls to move the robot arm (see [Teleoperation](teleoperation.md) for full control reference).
3. Click **Start Recording** before each demonstration.
4. Perform the task (e.g., pick up the red cube and place it at the green target).
5. Click **Stop Recording** to save the episode.
6. Repeat until you have at least 10 demonstrations.

Press `Space` to reset the environment between episodes.

## 4. Inspect Your Data

Check your collected dataset:

```bash
mimic data-info ./demos
```

This shows the environment name, number of episodes, total frames, action/state dimensions, and camera names.

View statistics:

```bash
mimic data-stats ./demos
```

## 5. Train a Policy

Train an ACT (Action Chunking Transformer) policy on your demonstrations:

```bash
mimic train --policy act --data ./demos
```

The trainer will automatically detect GPU availability. Common options:

```bash
# Train a Diffusion Policy instead
mimic train --policy diffusion --data ./demos

# Customize training
mimic train \
  --policy act \
  --data ./demos \
  --steps 100000 \
  --batch-size 32 \
  --lr 1e-4 \
  --eval-every 5000 \
  --save-every 10000 \
  --output outputs

# Force CPU training
mimic train --policy act --data ./demos --device cpu
```

Checkpoints are saved to the `outputs/` directory. The final model is saved as `outputs/final.pt`.

## 6. Evaluate

Run your trained policy in simulation:

```bash
mimic eval --checkpoint outputs/final.pt --env pick-place
```

This runs 10 evaluation episodes and reports success rate, mean return, and standard deviation.

```bash
# Run more episodes
mimic eval --checkpoint outputs/final.pt --env pick-place --episodes 50

# Evaluate on GPU
mimic eval --checkpoint outputs/final.pt --env pick-place --device cuda
```

## 7. Deploy

Export your trained model to ONNX for deployment:

```bash
mimic deploy outputs/final.pt
```

This creates `model.onnx` in the current directory. Specify a custom output path:

```bash
mimic deploy outputs/final.pt --output my_policy.onnx
```

The ONNX model takes a flat state vector as input and outputs an action chunk. See [Deployment](deployment.md) for running inference with the exported model.

## Next Steps

- [Environments](environments.md) -- Available tasks and creating custom environments
- [Teleoperation](teleoperation.md) -- Full control reference and recording guide
- [Training](training.md) -- Policy architectures and training configuration
- [Deployment](deployment.md) -- ONNX export and inference server
- [API Reference](api/index.md) -- Python API documentation
