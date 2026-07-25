"""`anycode api` — print the public API surface as a table or as JSON."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from anycode.introspect import CORE_SURFACE, EntryKind, describe, render_entry, render_text, to_json


def command(
    symbol: str = typer.Argument(default="", help="Describe one symbol instead of the whole surface."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    core: bool = typer.Option(False, "--core", help=f"Only the {len(CORE_SURFACE)} symbols that cover most use."),
    kind: str = typer.Option("", "--kind", help="Filter by kind: class, model, protocol, function, type, constant, module."),
    compact: bool = typer.Option(False, "--compact", help="Drop signatures. Smaller output when you only need what exists."),
) -> None:
    console = Console(soft_wrap=True)

    if symbol:
        try:
            entry = describe(symbol)
        except AttributeError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        console.print(json.dumps(entry.model_dump(), indent=2) if as_json else render_entry(entry))
        raise typer.Exit(0)

    api = describe(kind=_validated_kind(kind, console), core=core)
    if as_json:
        console.print(json.dumps(to_json(api, compact=compact), indent=2, sort_keys=False))
        raise typer.Exit(0)

    console.print(f"[bold cyan]anycode[/bold cyan] {api.version} — {len(api.entries)} public symbols")
    if core:
        console.print("[dim]The core surface. Run `anycode api` for everything, `anycode api <Symbol>` for one.[/dim]")
    console.print(render_text(api, show_signature=not compact))
    raise typer.Exit(0)


def _validated_kind(kind: str, console: Console) -> EntryKind | None:
    if not kind:
        return None
    valid = ("class", "model", "protocol", "function", "type", "constant", "module")
    if kind not in valid:
        console.print(f"[red]Unknown kind '{kind}'. Choose one of: {', '.join(valid)}.[/red]")
        raise typer.Exit(1)
    return kind  # type: ignore[return-value]
