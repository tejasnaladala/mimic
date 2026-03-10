# Training

Mimic supports two policy architectures for imitation learning: **ACT** (Action Chunking Transformer) and **Diffusion Policy**. Both predict chunks of future actions from the current state observation.

## Quick Start

```bash
# Train ACT policy (default)
mimic train --policy act --data ./demos

# Train Diffusion Policy
mimic train --policy diffusion --data ./demos
```

## ACT Policy

The Action Chunking Transformer uses a conditional VAE architecture to predict multi-step action sequences.

**Architecture:**

- State encoder MLP maps observations to hidden embeddings
- CVAE encoder (TransformerEncoder, 2 layers) encodes state + action sequences into a latent distribution during training
- TransformerDecoder (4 layers, 4 heads) generates action chunks from latent + state using cross-attention
- At inference, a zero-mean latent is used (no sampling)

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `obs_dim` | -- | Observation dimension (auto-detected from dataset) |
| `action_dim` | -- | Action dimension (auto-detected from dataset) |
| `action_chunk_size` | 10 | Number of future actions to predict |
| `hidden_dim` | 256 | Transformer hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 4 | Number of decoder layers |
| `latent_dim` | 32 | CVAE latent dimension |
| `dropout` | 0.1 | Dropout rate |
| `kl_weight` | 10.0 | KL divergence loss weight |

**Loss function:** MSE reconstruction loss + weighted KL divergence loss.

**Usage from Python:**

```python
from mimic.train.policies.act import ACTPolicy

policy = ACTPolicy(
    obs_dim=18,
    action_dim=9,
    action_chunk_size=10,
    hidden_dim=256,
    n_heads=4,
    n_layers=4,
    latent_dim=32,
)

# Training forward pass
output = policy.forward({"state": state_batch, "action": action_batch})
loss = output["loss"]  # total loss
recon = output["recon_loss"]  # reconstruction loss
kl = output["kl_loss"]  # KL divergence

# Inference (uses zero latent)
actions = policy.predict({"state": obs_state})  # [B, T, action_dim]
```

## Diffusion Policy

Diffusion Policy uses DDPM (Denoising Diffusion Probabilistic Models) to generate action chunks by iteratively denoising from Gaussian noise.

**Architecture:**

- State encoder MLP maps observations to hidden embeddings
- Sinusoidal timestep embedding + MLP for diffusion step conditioning
- MLP denoiser predicts noise from concatenated (noisy actions, state embedding, time embedding)
- DDPM reverse process at inference (100 steps by default)

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `obs_dim` | -- | Observation dimension |
| `action_dim` | -- | Action dimension |
| `action_chunk_size` | 10 | Number of future actions to predict |
| `hidden_dim` | 256 | Hidden layer dimension |
| `n_layers` | 4 | Number of denoiser MLP layers |
| `n_diffusion_steps` | 100 | Number of diffusion steps |
| `noise_schedule` | `"cosine"` | Noise schedule: `"cosine"` or `"linear"` |

**Loss function:** MSE between predicted noise and actual noise.

**Usage from Python:**

```python
from mimic.train.policies.diffusion import DiffusionPolicy

policy = DiffusionPolicy(
    obs_dim=18,
    action_dim=9,
    action_chunk_size=10,
    hidden_dim=256,
    n_layers=4,
    n_diffusion_steps=100,
    noise_schedule="cosine",
)

# Training forward pass
output = policy.forward({"state": state_batch, "action": action_batch})
loss = output["loss"]

# Inference (full DDPM reverse process)
actions = policy.predict({"state": obs_state})  # [B, T, action_dim]
```

## Training Configuration

Full CLI options for `mimic train`:

| Option | Default | Description |
|---|---|---|
| `--policy` | `act` | Policy architecture: `act` or `diffusion` |
| `--data` | (required) | Path to demonstration dataset |
| `--env` | `pick-place` | Environment for evaluation |
| `--steps` | `100000` | Total training steps |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--eval-every` | `5000` | Evaluate every N steps |
| `--save-every` | `10000` | Save checkpoint every N steps |
| `--output` | `outputs` | Output directory for checkpoints |
| `--device` | `auto` | Device: `auto`, `cpu`, or `cuda` |

The `TrainConfig` model:

```python
from mimic.config.models import TrainConfig

config = TrainConfig(
    policy="act",
    batch_size=32,
    lr=1e-4,
    steps=100000,
    save_every=10000,
    seed=42,
    device="auto",
    wandb=False,
)
```

## Training from Python

For programmatic training:

```python
from mimic.config.models import TrainConfig
from mimic.train.policies.act import ACTPolicy
from mimic.train.trainer import MimicTrainer

# Create policy
policy = ACTPolicy(obs_dim=18, action_dim=9)

# Configure training
config = TrainConfig(
    policy="act",
    batch_size=32,
    lr=1e-4,
    steps=50000,
    device="auto",
)

# Train
trainer = MimicTrainer(policy, config, dataset_path="./demos", output_dir="outputs")
trainer.train()

# Check progress
print(f"Step: {trainer.current_step}, Loss: {trainer.recent_loss:.4f}")
```

The trainer uses AdamW optimizer with weight decay of 1e-4 and gradient clipping (max norm 1.0). Checkpoints are saved as `checkpoint_{step}.pt` at intervals and `final.pt` at the end.

## Evaluation

Run a trained checkpoint against the simulation environment:

```bash
mimic eval --checkpoint outputs/final.pt --env pick-place --episodes 10
```

The evaluation process:

1. Loads the checkpoint (auto-detects ACT vs Diffusion from saved config)
2. Runs N episodes in the environment
3. Reports success rate, mean return, and standard deviation

From Python:

```python
import torch
from mimic.envs.registry import make
from mimic.train.policies.act import ACTPolicy
from mimic.train.eval import evaluate_policy

env = make("pick-place")
policy = ACTPolicy.load("outputs/final.pt")

results = evaluate_policy(policy, env, n_episodes=50, device="cpu")
print(f"Success rate: {results['success_rate']:.1%}")
print(f"Mean return: {results['mean_return']:.2f} +/- {results['std_return']:.2f}")

env.close()
```

## Saving and Loading

All policies inherit from `MimicPolicy` which provides `save()` and `load()`:

```python
# Save
policy.save("my_policy.pt")

# Load (class method)
policy = ACTPolicy.load("my_policy.pt")

# Checkpoints contain:
# - state_dict: model weights
# - config: dict of constructor arguments (obs_dim, action_dim, etc.)
```
