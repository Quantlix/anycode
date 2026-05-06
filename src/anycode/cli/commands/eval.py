"""`anycode eval` subcommands: run a scenario suite or compare two reports."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from anycode.eval import compare_reports, load_scenarios, read_report, render_markdown, run_suite, write_report

app = typer.Typer(help="Run and compare AnyCode harness evaluation suites.", no_args_is_help=True)


@app.command("run")
def run(
    suite_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML/JSON file with scenarios."),
    variant: str = typer.Option("baseline", "--variant", "-v", help="Label for this harness variant."),
    output: Path = typer.Option(Path("artifacts/eval/report.json"), "--output", "-o", help="Where to write the JSON report."),
    suite_name: str = typer.Option("default", "--name", "-n", help="Suite name."),
    markdown: bool = typer.Option(False, "--markdown", "-m", help="Print a markdown summary table."),
) -> None:
    """Execute every scenario in `suite_path` against the configured live LLM provider."""
    console = Console()
    scenarios = load_scenarios(suite_path)
    console.print(f"[cyan]Running {len(scenarios)} scenarios as variant={variant}…[/cyan]")
    report = asyncio.run(run_suite(scenarios, suite_name=suite_name, harness_variant=variant))
    target = write_report(report, output)
    console.print(f"[green]Report written to {target}[/green]")
    console.print(f"  passed={report.passed} failed={report.failed} runtime={report.total_runtime_seconds:.3f}s")
    if markdown:
        console.print("")
        console.print(render_markdown(report))
    if report.failed > 0:
        sys.exit(1)


@app.command("compare")
def compare(
    baseline_path: Path = typer.Argument(..., exists=True, readable=True),
    candidate_path: Path = typer.Argument(..., exists=True, readable=True),
    fail_on_regression: bool = typer.Option(True, "--fail-on-regression/--no-fail-on-regression"),
) -> None:
    """Diff two evaluation reports and surface regressions or improvements."""
    console = Console()
    baseline = read_report(baseline_path)
    candidate = read_report(candidate_path)
    diff = compare_reports(baseline, candidate)

    console.print(f"baseline: variant={diff['baseline']['variant']} passed={diff['baseline']['passed']} failed={diff['baseline']['failed']}")
    console.print(f"candidate: variant={diff['candidate']['variant']} passed={diff['candidate']['passed']} failed={diff['candidate']['failed']}")
    console.print(f"runtime delta: {diff['runtime_delta_seconds']:+.3f}s")

    if diff["regressions"]:
        console.print("[red]Regressions:[/red]")
        for line in diff["regressions"]:
            console.print(f"  - {line}")
    if diff["improvements"]:
        console.print("[green]Improvements:[/green]")
        for line in diff["improvements"]:
            console.print(f"  + {line}")
    if diff["new_scenarios"]:
        console.print("[cyan]New scenarios:[/cyan]")
        for name in diff["new_scenarios"]:
            console.print(f"  * {name}")

    if fail_on_regression and diff["regressions"]:
        sys.exit(1)
