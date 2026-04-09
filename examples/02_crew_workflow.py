# Demo 02 — Coordinated Agent Crew
# Execute: uv run python examples/02_crew_workflow.py

import asyncio
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, OrchestratorConfig, OrchestratorEvent, TeamConfig

load_dotenv()


def _resolve_provider() -> tuple[str, str]:
    """Return (provider, model) based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    print("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    sys.exit(1)


PROVIDER, MODEL = _resolve_provider()

# --- Define specialized crew members ---

planner = AgentConfig(
    name="planner",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a backend architect experienced with Python microservices.\n"
        "Design clean API specifications and directory layouts.\n"
        "Keep plans concise — output markdown only, skip filler text."
    ),
    tools=["bash", "file_write"],
    max_turns=4,
    temperature=0.15,
)

builder = AgentConfig(
    name="builder",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a seasoned Python backend engineer. Build exactly what the planner describes.\n"
        "Focus on clean, runnable code with sensible error handling. Use the tools to write and verify files."
    ),
    tools=["bash", "file_read", "file_write", "file_edit"],
    max_turns=10,
    temperature=0.1,
)

auditor = AgentConfig(
    name="auditor",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a meticulous code auditor. Examine code for bugs, security gaps, and clarity issues.\n"
        "Structure your report: Approved items, Suggestions, Blockers.\n"
        "Always read the files before giving your verdict."
    ),
    tools=["bash", "file_read", "grep"],
    max_turns=4,
    temperature=0.25,
)

# --- Progress tracking ---

agent_timers: dict[str, float] = {}


def handle_progress(ev: OrchestratorEvent) -> None:
    clock = datetime.now(UTC).isoformat()[11:23]

    match ev.type:
        case "agent_start":
            agent_timers[ev.agent or ""] = datetime.now(UTC).timestamp() * 1000
            print(f"[{clock}] STARTED   >> {ev.agent}")
        case "agent_complete":
            now = datetime.now(UTC).timestamp() * 1000
            duration = now - agent_timers.get(ev.agent or "", now)
            print(f"[{clock}] COMPLETED << {ev.agent} ({duration:.0f}ms)")
        case "task_start":
            print(f"[{clock}] TASK ON   >> {ev.task}")
        case "task_complete":
            print(f"[{clock}] TASK OFF  << {ev.task}")
        case "error":
            print(f"[{clock}] FAULT     !! agent={ev.agent} task={ev.task}")
            if isinstance(ev.data, Exception):
                print(f"             {ev.data}")


async def main() -> None:
    engine = AnyCode(
        config=OrchestratorConfig(
            default_model=MODEL,
            max_concurrency=1,
            on_progress=handle_progress,
        )
    )

    crew = engine.create_team(
        "backend-crew",
        TeamConfig(
            name="backend-crew",
            agents=[planner, builder, auditor],
            shared_memory=True,
            max_concurrency=1,
        ),
    )

    agent_names = ", ".join(a.name for a in crew.get_agents())
    print(f'Crew "{crew.name}" ready — members: {agent_names}')
    print("\nKicking off crew workflow...\n")
    print("#" * 55)

    objective = (
        "Build a lightweight FastAPI-style REST API in /tmp/task-api/ with:\n"
        "- GET  /ping          -> { message: 'pong' }\n"
        "- GET  /tasks         -> returns a hardcoded list of 3 task dicts (id, title, done)\n"
        "- POST /tasks         -> accepts { title } in body, logs it, responds with 201\n"
        "- A global error handler that returns structured JSON errors\n"
        "- Server binds to port 3005\n"
        "- Include a requirements.txt listing necessary dependencies"
    )

    report = await engine.run_team(crew, objective)

    print("\n" + "#" * 55)
    print("\nCrew workflow complete.")
    print(f"Outcome: {'SUCCESS' if report.success else 'FAILURE'}")
    print(f"Tokens consumed — input: {report.total_token_usage.input_tokens}, output: {report.total_token_usage.output_tokens}")

    print("\nBreakdown by member:")
    for member_name, member_report in report.agent_results.items():
        badge = "PASS" if member_report.success else "FAIL"
        tool_count = len(member_report.tool_calls)
        print(f"  {member_name:<10} [{badge}]  tool_calls={tool_count}")
        if not member_report.success:
            print(f"    Detail: {member_report.output[:100]}")

    builder_report = report.agent_results.get("builder")
    if builder_report and builder_report.success:
        print("\nBuilder output (last 500 chars):")
        print("-" * 55)
        txt = builder_report.output
        print(("…" + txt[-500:]) if len(txt) > 500 else txt)
        print("-" * 55)

    audit_report = report.agent_results.get("auditor")
    if audit_report and audit_report.success:
        print("\nAudit report:")
        print("-" * 55)
        print(audit_report.output)
        print("-" * 55)


if __name__ == "__main__":
    asyncio.run(main())
