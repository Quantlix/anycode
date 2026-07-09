"""`anycode inspect` — inspect tools, providers, or a team config."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from anycode import BUILT_IN_TOOLS
from anycode.config.loader import load_config
from anycode.config.validator import validate_config
from anycode.plugins.discovery import discover_entry_point_plugins
from anycode.plugins.registry import list_registered_providers


def command(
    target: str = typer.Argument(
        "tools",
        help="What to inspect: 'tools', 'providers', 'plugins', 'team <path>', or 'config <path>'.",
    ),
    path: str | None = typer.Argument(None, help="Path to a config file (when target is 'team' or 'config')."),
) -> None:
    console = Console()
    target_lower = target.lower()
    if target_lower == "tools":
        _show_tools(console)
    elif target_lower == "providers":
        _show_providers(console)
    elif target_lower == "plugins":
        _show_plugins(console)
    elif target_lower in ("team", "agents", "config"):
        if not path:
            console.print("[red]Provide a config path, e.g. `anycode inspect team team.yaml`.[/red]")
            raise typer.Exit(2)
        _show_team(console, path, validate_only=target_lower == "config")
    else:
        # If `target` itself looks like a file path, treat it as a team config
        if Path(target).exists():
            _show_team(console, target, validate_only=False)
        else:
            console.print(f"[red]Unknown inspect target:[/red] {target}")
            raise typer.Exit(2)


def _show_tools(console: Console) -> None:
    table = Table(title="Built-in Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for tool in BUILT_IN_TOOLS:
        table.add_row(tool.name, tool.description)
    console.print(table)


def _show_providers(console: Console) -> None:
    builtins = ["anthropic", "openai", "google", "ollama", "azure", "bedrock"]
    plugin_providers = [p for p in list_registered_providers() if p not in builtins]
    console.print("[bold]Built-in providers:[/bold] " + ", ".join(builtins))
    if plugin_providers:
        console.print("[bold]Plugin providers:[/bold]  " + ", ".join(plugin_providers))


def _show_plugins(console: Console) -> None:
    plugins = discover_entry_point_plugins()
    if not plugins:
        console.print("[dim]No plugins discovered under the 'anycode.plugins' entry-point group.[/dim]")
        return
    table = Table(title="Discovered Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Tools", justify="right")
    table.add_column("Providers", justify="right")
    table.add_column("Sensors", justify="right")
    table.add_column("Hooks", justify="right")
    table.add_column("Description")
    for plugin in plugins:
        manifest = plugin.manifest
        table.add_row(
            manifest.name,
            manifest.version,
            str(len(list(plugin.tools()))),
            str(len(plugin.provider_factories())),
            str(len(list(plugin.sensors()))),
            str(len(list(plugin.turn_hooks()))),
            manifest.description,
        )
    console.print(table)


def _show_team(console: Console, path: str, *, validate_only: bool) -> None:
    if not Path(path).exists():
        console.print(f"[red]Config file not found:[/red] {path}")
        raise typer.Exit(1)

    issues = validate_config(path)
    if issues:
        console.print(f"[red]Config has {len(issues)} issue(s):[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        raise typer.Exit(1)

    if validate_only:
        console.print(f"[green]Config OK:[/green] {path}")
        return

    loaded = load_config(path)
    table = Table(title=f"Team: {loaded.team.name}")
    table.add_column("Agent", style="cyan")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Tools")
    for agent in loaded.team.agents:
        table.add_row(
            agent.name,
            agent.model,
            agent.provider or "(default)",
            ", ".join(agent.tools) if agent.tools else "—",
        )
    console.print(table)

    if loaded.tasks:
        task_table = Table(title="Tasks")
        task_table.add_column("Title", style="cyan")
        task_table.add_column("Assignee")
        task_table.add_column("Depends on")
        for task in loaded.tasks:
            task_table.add_row(task.title, task.assignee, ", ".join(task.depends_on or []) or "—")
        console.print(task_table)
