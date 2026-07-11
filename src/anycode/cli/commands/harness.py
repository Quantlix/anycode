"""`anycode harness` subcommands.

Currently exposes:

- ``manifest``: emit a harness component manifest for a configured run.
- ``evolve``: experimental, dry-run-by-default harness evolution sweep.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from anycode.config.loader import load_config
from anycode.harness import build_default_registry, build_manifest, save_manifest
from anycode.security.redaction import redact_sensitive

app = typer.Typer(help="Inspect and (experimentally) evolve the AnyCode harness.", no_args_is_help=True)


@app.command("manifest")
def manifest(
    config_path: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="AnyCode config file."),
    output: Path = typer.Option(Path("artifacts/harness/manifest.json"), "--output", "-o", help="Where to write the manifest JSON."),
    notes: str = typer.Option("", "--notes", help="Optional note attached to the manifest."),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print the manifest JSON to stdout."),
) -> None:
    """Emit a deterministic harness component manifest for *config_path*."""

    console = Console()
    loaded = load_config(config_path)
    orchestrator = loaded.to_orchestrator_config()
    registry = build_default_registry(team=loaded.team, orchestrator=orchestrator)
    snapshot = build_manifest(registry, notes=notes or None)
    target = save_manifest(snapshot, output)
    console.print(f"[green]Manifest written to {target}[/green]")
    console.print(f"  components={len(snapshot.components)} checksum={snapshot.checksum[:12]}")
    if pretty:
        console.print(json.dumps(redact_sensitive(snapshot.model_dump(mode="json")), indent=2, default=str, sort_keys=True))


@app.command("evolve")
def evolve(
    suite_path: Path = typer.Argument(..., exists=True, readable=True, help="Eval suite YAML/JSON file."),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Maximum proposal cycles."),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Default dry-run; --apply is reserved for experimental tooling."),
    patch_dir: Path = typer.Option(Path("artifacts/harness/patches"), "--patch-dir", help="Where to emit reviewable patches."),
) -> None:
    """Run a controlled, dry-run harness evolution sweep.

    The default behaviour never writes back to the repository. Accepted changes
    are emitted as JSON patches under ``patch_dir`` for human review.
    """

    console = Console()
    console.print(f"[yellow]Harness evolution is experimental — running with dry_run={dry_run}.[/yellow]")
    console.print(f"  suite={suite_path}")
    console.print(f"  max_iterations={max_iterations}")
    console.print(f"  patch_dir={patch_dir}")

    if not dry_run:
        console.print("[red]--apply is reserved for experimental tooling. Use dry-run for now.[/red]")
        sys.exit(2)

    # Materialize a minimum-viable evolution call so the CLI is testable.
    from anycode.eval import load_scenarios, run_suite, write_report

    async def _baseline() -> None:
        scenarios = load_scenarios(suite_path)
        report = await run_suite(scenarios, suite_name="evolve-baseline", harness_variant="baseline")
        patch_dir.mkdir(parents=True, exist_ok=True)
        target = patch_dir / "baseline.json"
        write_report(report, target)
        console.print(f"[green]Baseline report written to {target}[/green]")

    asyncio.run(_baseline())
