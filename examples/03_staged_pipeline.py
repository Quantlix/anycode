# Demo 03 — Staged Task Pipeline with Dependency Graph
# Execute: uv run python examples/03_staged_pipeline.py

import asyncio
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, OrchestratorConfig, OrchestratorEvent, Task, TaskSpec, TeamConfig

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

# --- Specialist agent configurations ---

spec_writer = AgentConfig(
    name="spec-writer",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a technical specification author. Produce concise markdown specs\n"
        "detailing interfaces, data structures, and file layout. Avoid filler text."
    ),
    tools=["file_write"],
    max_turns=3,
)

coder = AgentConfig(
    name="coder",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a backend Python developer. Read the spec produced earlier\nand implement it to /tmp/cache-svc/. Use only the provided tools."
    ),
    tools=["bash", "file_read", "file_write"],
    max_turns=8,
)

validator = AgentConfig(
    name="validator",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a QA specialist. Execute the implemented code and verify it behaves correctly.\n"
        "Summarize: what succeeded, what broke, and any bugs discovered."
    ),
    tools=["bash", "file_read", "grep"],
    max_turns=5,
)

critic = AgentConfig(
    name="critic",
    model=MODEL,
    provider=PROVIDER,
    system_prompt=(
        "You are a senior code critic. Read source files and deliver a structured verdict.\n"
        "Sections: Overview, Positives, Concerns (if any), Decision (APPROVE / REVISE)."
    ),
    tools=["file_read", "grep"],
    max_turns=3,
)

# --- Timing instrumentation ---

timers: dict[str, float] = {}


def on_event(ev: OrchestratorEvent) -> None:
    ts = datetime.now(UTC).isoformat()[11:23]

    match ev.type:
        case "task_start":
            timers[ev.task or ""] = datetime.now(UTC).timestamp() * 1000
            detail = ev.data if isinstance(ev.data, Task) else None
            title = detail.title if detail else ev.task
            assignee = detail.assignee if detail else "unassigned"
            print(f'[{ts}] STAGE ON    "{title}" → {assignee}')
        case "task_complete":
            now = datetime.now(UTC).timestamp() * 1000
            elapsed = now - timers.get(ev.task or "", now)
            detail = ev.data if isinstance(ev.data, Task) else None
            title = detail.title if detail else ev.task
            print(f'[{ts}] STAGE OFF   "{title}" ({elapsed:.0f}ms)')
        case "agent_start":
            print(f"[{ts}] AGENT ON    {ev.agent}")
        case "agent_complete":
            print(f"[{ts}] AGENT OFF   {ev.agent}")
        case "error":
            print(f"[{ts}] PROBLEM     {ev.agent or ''}  stage={ev.task}")


async def main() -> None:
    engine = AnyCode(
        config=OrchestratorConfig(
            default_model=MODEL,
            max_concurrency=2,
            on_progress=on_event,
        )
    )

    crew = engine.create_team(
        "pipeline-crew",
        TeamConfig(
            name="pipeline-crew",
            agents=[spec_writer, coder, validator, critic],
            shared_memory=True,
        ),
    )

    # --- Task definitions with explicit dependency edges ---

    SPEC_PATH = "/tmp/cache-svc/spec.md"

    stages = [
        TaskSpec(
            title="Specify: in-memory cache service",
            description=(
                f"Draft a short technical spec and write it to {SPEC_PATH}.\n"
                "Cover the following:\n"
                "- Python dataclass/TypedDict for CacheEntry and CacheOptions\n"
                "- Eviction strategy (LRU with max-size)\n"
                "- Public API: set, get, has, remove, clear, stats\n"
                "Keep the spec under 25 lines of markdown."
            ),
            assignee="spec-writer",
        ),
        TaskSpec(
            title="Build: cache service implementation",
            description=(
                f"Read the spec at {SPEC_PATH}.\n"
                "Implement the cache service in /tmp/cache-svc/src/:\n"
                "- cache.py: core LRU cache class with all methods from the spec\n"
                "- helpers.py: utility for generating timestamps and computing entry ages\n"
                "- main.py: demo script that creates a cache, inserts items, triggers eviction, prints stats\n"
                "No third-party packages — only Python standard library."
            ),
            assignee="coder",
            depends_on=["Specify: in-memory cache service"],
        ),
        TaskSpec(
            title="Validate: cache service behavior",
            description=(
                "Execute the cache implementation:\n"
                "1. Run the demo script: python /tmp/cache-svc/src/main.py\n"
                "2. Confirm set/get/has/remove operations produce expected results\n"
                "3. Check that LRU eviction triggers when capacity is exceeded\n"
                "4. Report pass/fail for each check."
            ),
            assignee="validator",
            depends_on=["Build: cache service implementation"],
        ),
        TaskSpec(
            title="Critique: cache service code quality",
            description=(
                f"Read all .py files under /tmp/cache-svc/src/ and the spec at {SPEC_PATH}.\n"
                "Deliver a structured review:\n"
                "- Overview (two sentences)\n"
                "- Positives (bullet list)\n"
                '- Concerns (bullet list, or "None")\n'
                "- Decision: APPROVE or REVISE"
            ),
            assignee="critic",
            depends_on=["Build: cache service implementation"],
        ),
    ]

    # --- Launch the pipeline ---

    print("Launching 4-stage pipeline...\n")
    print("Flow: specify → build → [validate + critique] (parallel)")
    print("#" * 55)

    outcome = await engine.run_tasks(crew, stages)

    print("\n" + "#" * 55)
    print("Pipeline complete.\n")
    print(f"Overall result: {'OK' if outcome.success else 'FAIL'}")
    print(f"Token totals — input: {outcome.total_token_usage.input_tokens}, output: {outcome.total_token_usage.output_tokens}")

    print("\nAgent summaries:")
    for name, res in outcome.agent_results.items():
        tag = "PASS" if res.success else "FAIL"
        tool_list = ", ".join(c.tool_name for c in res.tool_calls)
        print(f"  [{tag}] {name:<14}  tools: {tool_list or '(none)'}")

    critique = outcome.agent_results.get("critic")
    if critique and critique.success:
        print("\nCode critique:")
        print("-" * 55)
        print(critique.output)
        print("-" * 55)


if __name__ == "__main__":
    asyncio.run(main())
