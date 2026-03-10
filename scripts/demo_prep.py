"""Prepare demo data for screen recording.

Creates a small dataset with 5 episodes so that the training
and evaluation CLI commands produce beautiful output.

Run: python scripts/demo_prep.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def create_demo_dataset(output_dir: str = "./demo_data", n_episodes: int = 5):
    """Create a small demo dataset with random data."""
    out = Path(output_dir)
    meta_dir = out / "meta"
    data_dir = out / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_dim = 31  # Panda arm state
    action_dim = 9  # 7 joints + 2 gripper fingers

    console.print(f"\n[bold cyan]Creating demo dataset[/bold cyan] -> {output_dir}")

    all_states = []
    all_actions = []
    all_rewards = []
    episodes = []
    total_frames = 0

    for ep in range(n_episodes):
        n_frames = np.random.randint(80, 150)
        t = np.linspace(0, 2 * np.pi, n_frames)
        states = np.column_stack(
            [np.sin(t * (i + 1) * 0.3 + i) * 0.5 for i in range(state_dim)]
        ).astype(np.float32)
        actions = np.column_stack(
            [np.sin(t * (i + 1) * 0.2 + i) * 0.1 for i in range(action_dim)]
        ).astype(np.float32)
        rewards = np.linspace(0, 1, n_frames).astype(np.float32)

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)

        episodes.append({
            "episode_index": ep,
            "num_frames": n_frames,
            "length_s": n_frames / 20.0,
        })
        total_frames += n_frames
        console.print(f"  [dim]Episode {ep+1}/{n_episodes}[/dim]  frames={n_frames}  [green]ok[/green]")

    # Save as parquet-like numpy files (simplified for demo)
    states_all = np.concatenate(all_states)
    actions_all = np.concatenate(all_actions)
    rewards_all = np.concatenate(all_rewards)

    np.save(str(data_dir / "state.npy"), states_all)
    np.save(str(data_dir / "action.npy"), actions_all)
    np.save(str(data_dir / "reward.npy"), rewards_all)

    # Write metadata
    info = {
        "env_name": "pick-place",
        "num_episodes": n_episodes,
        "num_frames": total_frames,
        "fps": 20,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_names": ["front", "wrist"],
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "episodes.json", "w") as f:
        json.dump(episodes, f, indent=2)

    # Compute and save stats
    stats = {
        "state": {
            "mean": states_all.mean(axis=0).tolist(),
            "std": states_all.std(axis=0).tolist(),
            "min": states_all.min(axis=0).tolist(),
            "max": states_all.max(axis=0).tolist(),
        },
        "action": {
            "mean": actions_all.mean(axis=0).tolist(),
            "std": actions_all.std(axis=0).tolist(),
            "min": actions_all.min(axis=0).tolist(),
            "max": actions_all.max(axis=0).tolist(),
        },
        "reward": {
            "mean": float(rewards_all.mean()),
            "std": float(rewards_all.std()),
            "min": float(rewards_all.min()),
            "max": float(rewards_all.max()),
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[green]Dataset created:[/green]")
    console.print(f"  Episodes: [cyan]{n_episodes}[/cyan]")
    console.print(f"  Frames:   [cyan]{total_frames}[/cyan]")
    console.print(f"  Path:     [cyan]{output_dir}[/cyan]\n")


def create_demo_checkpoint(data_dir: str = "./demo_data", output_dir: str = "./demo_output"):
    """Train a tiny policy for a few steps to get a checkpoint."""
    import torch

    from mimic.train.policies.act import ACTPolicy

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Training demo policy (50 steps)...[/bold cyan]")

    policy = ACTPolicy(obs_dim=31, action_dim=9, action_chunk_size=10, hidden_dim=64, n_layers=2)

    # Quick training on random data to get a valid checkpoint
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    data_dir_p = Path(data_dir) / "data" / "chunk-000"
    states = torch.from_numpy(np.load(str(data_dir_p / "state.npy")))
    actions = torch.from_numpy(np.load(str(data_dir_p / "action.npy")))

    n_steps = 50
    for step in range(n_steps):
        idx = np.random.randint(0, len(states) - 10)
        batch = {
            "state": states[idx : idx + 10].unsqueeze(0),
            "action": actions[idx : idx + 10].unsqueeze(0),
        }

        outputs = policy(batch)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == n_steps - 1:
            filled = (step + 1) * 30 // n_steps
            bar = "[green]" + "=" * filled + "[/green]" + "[dim]" + "-" * (30 - filled) + "[/dim]"
            console.print(
                f"  Step [bold]{step+1:3d}[/bold]/{n_steps}  {bar}  "
                f"loss=[cyan]{loss.item():.4f}[/cyan]"
            )
        time.sleep(0.05)

    ckpt_path = str(out / "best.pt")
    policy.save(ckpt_path)

    console.print(f"\n[green]Checkpoint saved:[/green] [cyan]{ckpt_path}[/cyan]\n")
    return ckpt_path


if __name__ == "__main__":
    console.print("\n[bold]===  MIMIC DEMO PREPARATION  ===[/bold]\n")
    create_demo_dataset("./demo_data", n_episodes=5)
    create_demo_checkpoint("./demo_data", "./demo_output")
    console.print("[bold green]Demo prep complete! Ready to record.[/bold green]\n")
