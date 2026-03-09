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


if __name__ == "__main__":
    app()
