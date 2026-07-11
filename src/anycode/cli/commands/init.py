"""`anycode init` — scaffold a new AnyCode project."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

TEMPLATE_TEAM_YAML = """\
# AnyCode team configuration
format_version: 1
name: backend-crew

agents:
  - name: planner
    model: claude-haiku-4-5
    provider: anthropic
    system_prompt: |
      You are a senior software architect.
      Decompose user goals into clear, actionable tasks.
    tools: []

  - name: builder
    model: claude-haiku-4-5
    provider: anthropic
    system_prompt: |
      You are a careful Python implementer.
      Produce high-quality, idiomatic code.
    tools:
      - file_read
      - file_write
      - file_edit

# Optional ordered tasks. Omit to use coordinator-driven planning.
tasks:
  - title: Plan API
    description: Sketch a REST API for a todo app.
    assignee: planner
  - title: Implement endpoints
    description: Implement the planned endpoints in Python.
    assignee: builder
    depends_on:
      - Plan API

# Optional features (uncomment as needed)
# cost:
#   budget_usd: 1.00
#   on_budget_exceeded: warn
# routing:
#   enabled: true
# rag:
#   enabled: false
"""

TEMPLATE_MAIN_PY = '''\
"""Entry point for an AnyCode project scaffolded by `anycode init`."""

import asyncio

from dotenv import load_dotenv

from anycode import AnyCode

load_dotenv()


async def main() -> None:
    engine = AnyCode.from_config("team.yaml")
    result = await engine.run_team_from_config()
    print(result.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())
'''

TEMPLATE_ENV = """\
# Anthropic
ANTHROPIC_API_KEY=

# OpenAI
OPENAI_API_KEY=
"""

TEMPLATE_TOOLS_INIT = '"""Custom tools for this AnyCode project."""\n'

TEMPLATE_GITIGNORE = ".env\n__pycache__/\n*.pyc\n.venv/\n"


def command(
    name: str = typer.Argument(..., help="Project directory to create."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
) -> None:
    console = Console()
    target = Path(name)
    if target.exists() and not force:
        if any(target.iterdir()):
            console.print(f"[red]Directory '{target}' is not empty. Pass --force to overwrite.[/red]")
            raise typer.Exit(1)

    target.mkdir(parents=True, exist_ok=True)
    (target / "tools").mkdir(exist_ok=True)

    files = {
        target / "team.yaml": TEMPLATE_TEAM_YAML,
        target / "main.py": TEMPLATE_MAIN_PY,
        target / ".env.example": TEMPLATE_ENV,
        target / "tools" / "__init__.py": TEMPLATE_TOOLS_INIT,
        target / ".gitignore": TEMPLATE_GITIGNORE,
    }

    for path, content in files.items():
        if path.exists() and not force:
            continue
        path.write_text(content, encoding="utf-8")

    console.print(f"[green]Created project:[/green] {target}/")
    for path in files:
        console.print(f"  [dim]{path.relative_to(target)}[/dim]")
    console.print(f"\nNext: [cyan]cd {target} && cp .env.example .env && anycode run team.yaml[/cyan]")
