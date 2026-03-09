import typer
from rich.console import Console

app = typer.Typer(
    name="mimic",
    help="Teach your robot anything from your browser.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Mimic -- Teach your robot anything from your browser."""


@app.command()
def version():
    """Show mimic version."""
    from mimic import __version__

    console.print(f"mimic v{__version__}")


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


if __name__ == "__main__":
    app()
