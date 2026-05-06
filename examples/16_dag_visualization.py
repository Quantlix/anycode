# Demo 16 — DAG Visualization (Phase 5.3)
# Execute: uv run python examples/16_dag_visualization.py
#
# Demonstrates rendering a task DAG in mermaid, dot, JSON, and ASCII,
# then renders a timeline from a real team run.
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

import asyncio
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, TaskQueue, TaskSpec, TeamConfig, render_dag, render_timeline
from anycode.tasks.task import create_task

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ProviderName = Literal["anthropic", "openai"]


def _resolve_provider() -> tuple[ProviderName, str] | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


def _build_demo_queue() -> TaskQueue:
    queue = TaskQueue()
    plan = create_task(title="Plan", description="Plan the build.", assignee="planner")
    queue.add(plan)
    queue.update(plan.id, status="completed")

    build = create_task(title="Build", description="Implement the design.", assignee="builder", depends_on=[plan.id])
    queue.add(build)
    queue.update(build.id, status="in_progress")

    test = create_task(title="Test", description="Verify the build.", assignee="qa", depends_on=[build.id])
    queue.add(test)

    deploy = create_task(title="Deploy", description="Ship it.", assignee="ops", depends_on=[test.id])
    queue.add(deploy)
    return queue


async def _run_live_timeline(provider: ProviderName, model: str):
    engine = AnyCode()
    team = engine.create_team(
        "viz-demo",
        TeamConfig(
            name="viz-demo",
            agents=[
                AgentConfig(
                    name="planner",
                    model=model,
                    provider=provider,
                    system_prompt="You produce concise execution plans.",
                    tools=[],
                ),
                AgentConfig(
                    name="builder",
                    model=model,
                    provider=provider,
                    system_prompt="You describe implementation steps clearly and briefly.",
                    tools=[],
                ),
                AgentConfig(
                    name="qa",
                    model=model,
                    provider=provider,
                    system_prompt="You describe validation checks in concise terms.",
                    tools=[],
                ),
            ],
        ),
    )
    tasks = [
        TaskSpec(
            title="Plan",
            description="In one short sentence, outline a release plan for a demo CLI tool.",
            assignee="planner",
        ),
        TaskSpec(
            title="Build",
            description="In two short sentences, describe the implementation approach for that CLI tool.",
            assignee="builder",
            depends_on=["Plan"],
        ),
        TaskSpec(
            title="Test",
            description="In one short sentence, describe the most important validation for that CLI tool.",
            assignee="qa",
            depends_on=["Build"],
        ),
    ]
    return await engine.run_tasks(team, tasks)


class _OutputCapture:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._buffer.write(text)

    def flush(self) -> None:
        self._stdout.flush()

    def get_output(self) -> str:
        return self._buffer.getvalue()

    def restore(self) -> None:
        sys.stdout = self._stdout


def _write_artifacts(queue: TaskQueue, *, timeline_text: str, agent_outputs_text: str, console_output: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "16-dag-visualization-ascii.txt": render_dag(queue, format="ascii"),
        "16-dag-visualization.mmd": render_dag(queue, format="mermaid"),
        "16-dag-visualization.dot": render_dag(queue, format="dot"),
        "16-dag-visualization.json": render_dag(queue, format="json"),
        "16-dag-visualization-timeline.txt": timeline_text,
        "16-dag-visualization-agent-outputs.txt": agent_outputs_text,
        "16-dag-visualization-output.txt": console_output,
    }
    for name, content in artifacts.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")


async def main() -> None:
    capture = _OutputCapture()
    sys.stdout = capture  # type: ignore[assignment]

    resolved = _resolve_provider()
    provider_label = resolved[0] if resolved is not None else "none"
    model_label = resolved[1] if resolved is not None else "none"

    queue = _build_demo_queue()
    ascii_dag = render_dag(queue, format="ascii")
    mermaid_dag = render_dag(queue, format="mermaid")
    dot_dag = render_dag(queue, format="dot")
    json_dag = render_dag(queue, format="json")

    print(f"=== DAG Visualization Demo (provider={provider_label}, model={model_label}) ===")
    print("=== ASCII tree ===")
    print(ascii_dag)
    print("\n=== Mermaid (paste into a Markdown viewer) ===")
    print(mermaid_dag)
    print("\n=== Graphviz DOT ===")
    print(dot_dag)
    print("\n=== JSON ===")
    print(json_dag)

    timeline_text = "Skipped live timeline: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env."
    agent_outputs_text = timeline_text
    if resolved is None:
        print("\n=== Timeline (live team run) ===")
        print(timeline_text)
    else:
        provider, model = resolved
        print("\n=== Timeline (live team run) ===")
        result = await _run_live_timeline(provider, model)
        timeline_text = render_timeline(result, width=40)
        print(timeline_text)

        agent_lines = [f"  {name}: {agent_result.output[:140]}" for name, agent_result in result.agent_results.items()]
        agent_outputs_text = "\n".join(agent_lines)

        print("\nAgent outputs:")
        for line in agent_lines:
            print(line)

    capture.restore()
    _write_artifacts(queue, timeline_text=timeline_text, agent_outputs_text=agent_outputs_text, console_output=capture.get_output())
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
