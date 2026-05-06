# Demo 17 — YAML Config + CLI loading (Phase 4.2)
# Execute: uv run python examples/17_yaml_config.py
#
# Demonstrates building an AnyCode engine from a YAML config file.
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from anycode import AnyCode

load_dotenv()


def _resolve_provider() -> tuple[str, str] | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


YAML_TEMPLATE = """\
name: yaml-demo
agents:
  - name: assistant
    model: {model}
    provider: {provider}
    system_prompt: You are a friendly assistant. Answer concisely.
    tools: []
tasks:
  - title: greet
    description: Greet the user in one short sentence and mention the value of YAML configs.
    assignee: assistant
"""


async def main() -> None:
    resolved = _resolve_provider()
    if resolved is None:
        print("ERROR: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")
        sys.exit(1)
    provider, model = resolved

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "team.yaml"
        cfg_path.write_text(YAML_TEMPLATE.format(model=model, provider=provider), encoding="utf-8")

        engine = AnyCode.from_config(str(cfg_path))
        result = await engine.run_team_from_config()

        print(f"success={result.success}")
        for name, agent_result in result.agent_results.items():
            print(f"  {name}: {agent_result.output[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
