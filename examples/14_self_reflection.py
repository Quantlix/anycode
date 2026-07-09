# Demo 14 — Self-Reflection
# Execute: uv run python examples/14_self_reflection.py
#
# Demonstrates a critic loop that retries until quality threshold is met.
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

import asyncio
import os
import sys

from dotenv import load_dotenv

from anycode import AgentConfig, AnyCode, OrchestratorConfig, ReflectionConfig

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
        reflection=ReflectionConfig(
            enabled=True,
            mode="self",
            quality_threshold=0.8,
            max_reflections=2,
        ),
    )
    engine = AnyCode(config)

    agent = engine.build_agent(
        AgentConfig(
            name="explainer",
            model=model,
            provider=provider,
            system_prompt="You are a precise technical writer. Write clear, accurate explanations.",
            tools=[],
        ),
    )

    print(f"=== Self-Reflection Demo (provider={provider}, model={model}) ===\n")
    # Run via the reflection loop directly so we can show counts.
    from anycode.reflection.loop import ReflectionLoop
    from anycode.types import AgentInfo

    loop = ReflectionLoop(config.reflection)
    info = AgentInfo(name="explainer", role="technical writer", model=model)
    result = await loop.run(
        agent,
        "Explain the CAP theorem to a junior engineer in exactly 3 sentences.",
        agent_info=info,
        agent_provider=provider,
    )

    print(f"reflections_count = {result.reflections_count}")
    print(f"quality_score     = {result.quality_score}")
    print(f"\nFinal output:\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
