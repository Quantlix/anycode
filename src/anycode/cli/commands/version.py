"""`anycode version` command."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import typer
from rich.console import Console

PROVIDER_MODULES = ("anthropic", "openai", "google.genai", "boto3")
EXTRA_MODULES = (("telemetry", "opentelemetry"), ("cli", "typer"), ("persistence", "redis"))


def command() -> None:
    console = Console()
    try:
        pkg_version = importlib.metadata.version("anycode-py")
    except importlib.metadata.PackageNotFoundError:
        pkg_version = "0.0.0+local"

    console.print(f"[bold cyan]anycode[/bold cyan] {pkg_version}")
    console.print(f"Python {sys.version.split()[0]}")

    available_providers = []
    for mod in PROVIDER_MODULES:
        try:
            importlib.import_module(mod)
            available_providers.append(mod.split(".")[0])
        except ImportError:
            continue
    if available_providers:
        console.print(f"Providers available: {', '.join(sorted(set(available_providers)))}")

    available_extras = [name for name, mod in EXTRA_MODULES if _has_module(mod)]
    if available_extras:
        console.print(f"Extras installed: {', '.join(available_extras)}")

    raise typer.Exit(0)


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False
