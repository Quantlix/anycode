"""`anycode run` — execute a config file or single agent prompt."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

from anycode import AgentConfig, AnyCode


def command(
    config: str | None = typer.Argument(None, help="Path to a YAML/TOML team config file."),
    goal: str | None = typer.Option(None, "--goal", "-g", help="Goal for coordinator-driven team execution."),
    agent: str | None = typer.Option(None, "--agent", help="Single-agent mode: name to display."),
    model: str | None = typer.Option(None, "--model", help="Model id for single-agent mode."),
    provider: str = typer.Option("anthropic", "--provider", help="LLM provider for single-agent mode."),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Prompt for single-agent mode."),
) -> None:
    console = Console()

    if agent or model or prompt:
        if not (agent and model and prompt):
            console.print("[red]Single-agent mode requires --agent, --model, and --prompt.[/red]")
            raise typer.Exit(2)
        asyncio.run(_run_single(console, agent, model, provider, prompt))
        return

    if not config:
        console.print("[red]Provide a config file or single-agent flags. See `anycode run --help`.[/red]")
        raise typer.Exit(2)

    if not Path(config).exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(1)

    asyncio.run(_run_team(console, config, goal))


async def _run_team(console: Console, config_path: str, goal: str | None) -> None:
    async with AnyCode.from_config(config_path) as engine:
        console.print(f"[cyan]Loaded team from[/cyan] {config_path}")
        result = await engine.run_team_from_config(goal=goal)
    console.print(f"\n[green]Done.[/green] success={result.success}")
    console.print(f"Tokens: in={result.total_token_usage.input_tokens} out={result.total_token_usage.output_tokens}")
    if result.cost_report:
        console.print(f"Cost: ${result.cost_report.total_cost_usd:.4f}")
    if result.handoffs:
        console.print(f"Handoffs: {len(result.handoffs)}")
    for name, agent_result in result.agent_results.items():
        status = "[green]ok[/green]" if agent_result.success else "[red]fail[/red]"
        console.print(f"  {name}: {status} — {agent_result.output[:120]}")


async def _run_single(console: Console, name: str, model: str, provider: str, prompt: str) -> None:
    cfg = AgentConfig(name=name, model=model, provider=provider, tools=[])  # type: ignore[arg-type]
    async with AnyCode() as engine:
        runtime_agent = engine.build_agent(cfg)
        console.print(f"[cyan]Running agent[/cyan] {name} ({provider}/{model})")
        result = await runtime_agent.run(prompt)
    console.print(f"\n[bold]Output:[/bold]\n{result.output}")
    console.print(f"\nTokens: in={result.token_usage.input_tokens} out={result.token_usage.output_tokens}")


def _resolve_api_key(provider: str) -> str | None:  # pragma: no cover - reserved for future direct adapter use
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    return os.environ.get(env_var) if env_var else None
