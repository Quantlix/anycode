"""AnyCode CLI entry point."""

from __future__ import annotations

try:
    import typer
except ImportError as e:  # pragma: no cover - import-time guard
    raise RuntimeError("AnyCode CLI requires the 'cli' extras. Install with: pip install 'anycode-py[cli]'") from e

from anycode.cli.commands import init as init_cmd
from anycode.cli.commands import inspect as inspect_cmd
from anycode.cli.commands import run as run_cmd
from anycode.cli.commands import version as version_cmd
from anycode.cli.commands.eval import app as eval_app

app = typer.Typer(
    name="anycode",
    help="AnyCode — multi-agent AI orchestration CLI.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("init", help="Scaffold a new AnyCode project.")(init_cmd.command)
app.command("run", help="Run an AnyCode team or agent from a config file or flags.")(run_cmd.command)
app.command("inspect", help="Inspect built-in tools, providers, or a team config.")(inspect_cmd.command)
app.command("version", help="Print the AnyCode version and runtime info.")(version_cmd.command)
app.add_typer(eval_app, name="eval", help="Run and compare harness evaluation suites.")


def main() -> None:  # pragma: no cover - thin shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
