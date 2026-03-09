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


if __name__ == "__main__":
    app()
