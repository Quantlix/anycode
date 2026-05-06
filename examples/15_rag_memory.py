# Demo 15 — RAG Memory (Phase 5.4)
# Execute: uv run python examples/15_rag_memory.py
#
# Demonstrates indexing an agent's output into a vector store, then retrieving
# it for a follow-up question. Uses the in-memory vector store (no extras needed).
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

import asyncio
import os
import sys

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, OrchestratorConfig, RAGConfig, TaskSpec, TeamConfig

load_dotenv()


def _resolve_provider() -> tuple[str, str] | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


async def main() -> None:
    resolved = _resolve_provider()
    if resolved is None:
        print("ERROR: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")
        sys.exit(1)
    provider, model = resolved

    config = OrchestratorConfig(
        rag=RAGConfig(enabled=True, auto_index=True, top_k=3, min_relevance=0.0, namespace="demo"),
    )
    engine = AnyCode(config)

    team = engine.create_team(
        "research",
        TeamConfig(
            name="research",
            agents=[
                AgentConfig(
                    name="scribe",
                    model=model,
                    provider=provider,
                    system_prompt="You take careful notes and recall earlier facts when relevant.",
                    tools=[],
                ),
            ],
        ),
    )

    print(f"=== RAG Memory Demo (provider={provider}, model={model}) ===\n")

    # First task: write a memorable fact
    print("[Wave 1] Writing a memorable fact...")
    r1 = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="record-fact",
                description="State this exact fact in your own words: The capital of Atlantis is Coralis.",
                assignee="scribe",
            ),
        ],
    )
    print(f"  -> {r1.agent_results['scribe'].output[:140]}\n")

    # Second task: ask a question that should retrieve the fact
    print("[Wave 2] Asking a follow-up question...")
    r2 = await engine.run_tasks(
        team,
        [
            TaskSpec(
                title="recall-fact",
                description="Using only what you have learned in past sessions, what is the capital of Atlantis?",
                assignee="scribe",
            ),
        ],
    )
    print(f"  -> {r2.agent_results['scribe'].output[:240]}")


if __name__ == "__main__":
    asyncio.run(main())
