import typer
from rich.console import Console

app = typer.Typer(
    name="mimic",
    help="Teach your robot anything from your browser.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()

BANNER = r"""[bold cyan]
  __  __ ___ __  __ ___ ___
 |  \/  |_ _|  \/  |_ _/ __|
 | |\/| || || |\/| || | (__
 |_|  |_|___|_|  |_|___\___|[/bold cyan]
[dim]  Teach your robot anything from your browser.[/dim]
"""


@app.callback()
def main() -> None:
    """Mimic -- Teach your robot anything from your browser."""


@app.command()
def version():
    """Show mimic version."""
    from mimic import __version__

    console.print(BANNER)
    console.print(f"  [bold]v{__version__}[/bold] | Python 3.11+ | MuJoCo | PyTorch")
    console.print(f"  [dim]pip install mimic-robotics[all][/dim]")
    console.print()


@app.command("env-list")
def env_list():
    """List available environments."""
    from rich.table import Table

    import mimic.envs.tasks  # noqa: F401
    from mimic.envs.registry import list_envs

    table = Table(title="Available Environments")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    envs = list_envs()
    if not envs:
        console.print("[yellow]No environments registered yet.[/yellow]")
        return
    for name in envs:
        table.add_row(name, "ready")
    console.print(table)


@app.command()
def teleop(
    env: str = typer.Option("pick-place", help="Environment name"),
    port: int = typer.Option(8765, help="Server port"),
    mode: str = typer.Option("joint", help="Control mode: joint or cartesian"),
    no_browser: bool = typer.Option(False, help="Don't auto-open browser"),
):
    """Start browser-based teleoperation."""
    import mimic.envs.tasks  # noqa: F401
    from mimic.config.models import TeleopConfig
    from mimic.envs.registry import make as make_env
    from mimic.teleop.loop import TeleopLoop

    console.print("[bold cyan]Mimic Teleoperation[/bold cyan]")
    console.print(f"  Environment: [green]{env}[/green]")
    console.print(f"  Control mode: [green]{mode}[/green]")
    console.print(f"  Server: [blue]http://localhost:{port}[/blue]")
    console.print()

    try:
        environment = make_env(env)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    environment.reset()
    config = TeleopConfig(port=port, control_mode=mode)
    loop = TeleopLoop(environment, config)

    console.print("[yellow]Press Ctrl+C to stop[/yellow]")
    try:
        loop.run(open_browser=not no_browser)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        environment.close()


@app.command("data-info")
def data_info(
    path: str = typer.Argument(help="Path to dataset"),
):
    """Show dataset information."""
    from rich.panel import Panel

    from mimic.data.dataset import MimicDataset

    ds = MimicDataset(path)
    meta = ds.metadata

    console.print(
        Panel(
            f"[bold]{meta.get('env_name', 'unknown')}[/bold]\n"
            f"Episodes: [cyan]{meta.get('num_episodes', 0)}[/cyan]\n"
            f"Frames: [cyan]{meta.get('num_frames', 0)}[/cyan]\n"
            f"FPS: {meta.get('fps', 20)}\n"
            f"Action dim: {meta.get('action_dim', '?')}\n"
            f"State dim: {meta.get('state_dim', '?')}\n"
            f"Cameras: {', '.join(meta.get('camera_names', []))}",
            title="Dataset Info",
        )
    )


@app.command("data-export")
def data_export(
    path: str = typer.Argument(help="Path to source dataset"),
    output: str = typer.Argument(help="Output path"),
    fmt: str = typer.Option("lerobot", "--format", help="Export format: lerobot, hdf5, rlds"),
):
    """Export dataset to another format."""
    from pathlib import Path

    from mimic.data.formats import export_to_hdf5, export_to_lerobot, export_to_rlds

    source = Path(path)
    dest = Path(output)

    exporters = {
        "lerobot": export_to_lerobot,
        "hdf5": export_to_hdf5,
        "rlds": export_to_rlds,
    }

    if fmt not in exporters:
        console.print(f"[red]Unknown format: {fmt}. Available: {', '.join(exporters.keys())}[/red]")
        raise typer.Exit(1)

    console.print(f"Exporting [cyan]{source}[/cyan] to [cyan]{fmt}[/cyan] format...")
    exporters[fmt](source, dest)
    console.print(f"[green]Exported to {dest}[/green]")


@app.command()
def train(
    policy: str = typer.Option("act", help="Policy: act or diffusion"),
    data: str = typer.Option(..., help="Path to dataset"),
    env: str = typer.Option("pick-place", help="Environment for evaluation"),
    steps: int = typer.Option(100000, help="Training steps"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    eval_every: int = typer.Option(5000, help="Eval frequency"),
    save_every: int = typer.Option(10000, help="Checkpoint save frequency"),
    output: str = typer.Option("outputs", help="Output directory"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda"),
):
    """Train a policy on collected demonstrations."""
    import json
    from pathlib import Path

    from mimic.config.models import TrainConfig
    from mimic.train.policies.act import ACTPolicy
    from mimic.train.policies.diffusion import DiffusionPolicy
    from mimic.train.trainer import MimicTrainer

    console.print("[bold cyan]Mimic Training[/bold cyan]")
    console.print(f"  Policy: [green]{policy}[/green]")
    console.print(f"  Dataset: [green]{data}[/green]")
    console.print(f"  Steps: {steps}")
    console.print()

    # Load dataset metadata
    meta_path = Path(data) / "meta" / "info.json"
    if not meta_path.exists():
        console.print(f"[red]Dataset not found at {data}[/red]")
        raise typer.Exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    obs_dim = meta["state_dim"]
    action_dim = meta["action_dim"]

    # Create policy
    policies = {
        "act": lambda: ACTPolicy(obs_dim=obs_dim, action_dim=action_dim),
        "diffusion": lambda: DiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim),
    }
    if policy not in policies:
        console.print(
            f"[red]Unknown policy: {policy}. Available: {', '.join(policies)}[/red]"
        )
        raise typer.Exit(1)

    model = policies[policy]()
    config = TrainConfig(
        policy=policy,
        batch_size=batch_size,
        lr=lr,
        steps=steps,
        eval_every=eval_every,
        save_every=save_every,
        device=device,
    )

    trainer = MimicTrainer(model, config, data, output_dir=output)
    console.print(f"[yellow]Training on {trainer.device}...[/yellow]")

    try:
        trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Saving checkpoint...[/yellow]")
        trainer.save_checkpoint("interrupted.pt")

    console.print(f"[green]Training complete! Checkpoints saved to {output}[/green]")


@app.command("eval")
def evaluate(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint"),
    env: str = typer.Option("pick-place", help="Environment name"),
    episodes: int = typer.Option(10, help="Number of evaluation episodes"),
    device: str = typer.Option("cpu", help="Device"),
):
    """Evaluate a trained policy in simulation."""
    import torch
    from rich.table import Table

    import mimic.envs.tasks  # noqa: F401
    from mimic.envs.registry import make as make_env
    from mimic.train.eval import evaluate_policy
    from mimic.train.policies.act import ACTPolicy
    from mimic.train.policies.diffusion import DiffusionPolicy

    console.print("[bold cyan]Mimic Evaluation[/bold cyan]")

    environment = make_env(env)

    # Try loading as ACT first, then Diffusion
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    if "n_diffusion_steps" in config:
        model = DiffusionPolicy.load(checkpoint)
    else:
        model = ACTPolicy.load(checkpoint)

    model = model.to(device)
    console.print(f"Evaluating on [green]{env}[/green] for {episodes} episodes...")

    results = evaluate_policy(model, environment, n_episodes=episodes, device=device)

    table = Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Success Rate", f"{results['success_rate']:.1%}")
    table.add_row("Mean Return", f"{results['mean_return']:.2f}")
    table.add_row("Std Return", f"{results['std_return']:.2f}")
    table.add_row("Episodes", str(results['n_episodes']))
    console.print(table)

    environment.close()


@app.command("data-stats")
def data_stats(
    path: str = typer.Argument(help="Path to dataset"),
):
    """Compute and display dataset statistics."""
    from pathlib import Path

    from rich.table import Table

    if (Path(path) / "meta" / "stats.json").exists():
        import json

        with open(Path(path) / "meta" / "stats.json") as f:
            stats = json.load(f)
        table = Table(title="Dataset Statistics")
        table.add_column("Feature")
        table.add_column("Mean")
        table.add_column("Std")
        table.add_column("Min")
        table.add_column("Max")
        for feature, values in stats.items():
            mean = values.get("mean", "?")
            std = values.get("std", "?")
            min_val = values.get("min", "?")
            max_val = values.get("max", "?")
            # Format arrays nicely
            if isinstance(mean, list):
                mean = f"[{len(mean)} dims]"
                std = f"[{len(std)} dims]" if isinstance(std, list) else std
                min_val = f"[{len(min_val)} dims]" if isinstance(min_val, list) else min_val
                max_val = f"[{len(max_val)} dims]" if isinstance(max_val, list) else max_val
            table.add_row(feature, str(mean), str(std), str(min_val), str(max_val))
        console.print(table)
    else:
        console.print("[yellow]No stats computed yet. Run with an active dataset.[/yellow]")


@app.command("deploy")
def deploy(
    checkpoint: str = typer.Argument(help="Path to model checkpoint (.pt)"),
    output: str = typer.Option("model.onnx", help="Output ONNX path"),
):
    """Export a trained model to ONNX format for deployment."""
    from mimic.deploy.export import export_to_onnx

    console.print("[bold cyan]Exporting model...[/bold cyan]")
    console.print(f"  Checkpoint: [green]{checkpoint}[/green]")
    console.print(f"  Output: [green]{output}[/green]")

    try:
        path = export_to_onnx(checkpoint, output)
        console.print(f"[green]Exported to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("hub-push")
def hub_push(
    path: str = typer.Argument(help="Path to dataset directory"),
    repo: str = typer.Option(..., help="HuggingFace repo ID (e.g. username/pick-place-demos)"),
    private: bool = typer.Option(False, help="Make the repository private"),
):
    """Push a dataset to HuggingFace Hub."""
    from mimic.hub.client import MimicHubClient

    client = MimicHubClient()
    console.print(f"Pushing [cyan]{path}[/cyan] to [green]{repo}[/green]...")

    try:
        url = client.push_dataset(path, repo, private=private)
        console.print(f"[green]Dataset pushed to {url}[/green]")
    except Exception as e:
        console.print(f"[red]Push failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("hub-pull")
def hub_pull(
    repo: str = typer.Argument(help="HuggingFace repo ID (e.g. username/pick-place-demos)"),
    output: str = typer.Option("./demos", help="Output directory"),
):
    """Pull a dataset from HuggingFace Hub."""
    from mimic.hub.client import MimicHubClient

    client = MimicHubClient()
    console.print(f"Pulling [green]{repo}[/green] to [cyan]{output}[/cyan]...")

    try:
        result_path = client.pull_dataset(repo, output)
        console.print(f"[green]Dataset downloaded to {result_path}[/green]")
    except Exception as e:
        console.print(f"[red]Pull failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("hub-push-model")
def hub_push_model(
    path: str = typer.Argument(help="Path to model checkpoint (.pt)"),
    repo: str = typer.Option(..., help="HuggingFace repo ID (e.g. username/pick-place-act)"),
    private: bool = typer.Option(False, help="Make the repository private"),
):
    """Push a model checkpoint to HuggingFace Hub."""
    from mimic.hub.client import MimicHubClient

    client = MimicHubClient()
    console.print(f"Pushing [cyan]{path}[/cyan] to [green]{repo}[/green]...")

    try:
        url = client.push_model(path, repo, private=private)
        console.print(f"[green]Model pushed to {url}[/green]")
    except Exception as e:
        console.print(f"[red]Push failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
