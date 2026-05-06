# Demo 19 — Adaptive Context Lifecycle
# Execute: uv run python examples/19_adaptive_context.py
"""
Demonstrates the ContextManager:
  - Pressure classification (normal/trim/offload/compact/handoff) per ContextPolicy
  - Large tool outputs are offloaded to disk with digest + head/tail excerpts
  - Older dialogue is compacted into a structured summary
  - Handoff mode emits a recoverable JSON file that rebuilds a fresh prompt

Real LLM calls are made when ANTHROPIC_API_KEY or OPENAI_API_KEY is present.
Run artifacts (offloaded payloads, handoff files, manifests) live under
./artifacts/context/ so successive runs accumulate inspectable history.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from anycode import (
    Agent,
    AgentConfig,
    ContextManager,
    ContextPolicy,
    LLMMessage,
    TextBlock,
    ToolExecutor,
    ToolRegistry,
    ToolResultBlock,
    rebuild_from_handoff,
)

load_dotenv()

ARTIFACT_DIR = Path("artifacts/context")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _ts_label() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _persist(name: str, payload: dict) -> Path:
    target = ARTIFACT_DIR / f"{_ts_label()}_{name}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _select_provider() -> tuple[str | None, str | None]:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-sonnet-4-5"
    if os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None, None


# ----- Section A: pressure classification & offload (no LLM) ----------


async def section_offload() -> None:
    print("\n=== A. ContextPolicy → tool output offload ===")
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=8_000,
        offload_ratio=0.3,
        compact_ratio=0.95,
        handoff_ratio=0.99,
        max_tool_output_tokens=200,
        artifact_dir=str(ARTIFACT_DIR / "offload"),
    )
    manager = ContextManager(policy)

    big_tool_output = "stdout: " + ("data block " * 1500)
    messages = [
        LLMMessage(role="user", content=[TextBlock(text="Run the database scan")]),
        LLMMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_db", content=big_tool_output)],
        ),
    ]
    new_messages, manifest = manager.assemble(messages)
    print(f"pressure={manifest.pressure} estimated_tokens={manifest.estimated_tokens} offloaded={len(manifest.offloaded)}")
    if manifest.offloaded:
        artifact = manifest.offloaded[0]
        print(f"  artifact path={artifact.path} bytes={artifact.bytes} digest={artifact.digest[:12]}…")
    target = _persist("offload_manifest", manifest.model_dump())
    print(f"  manifest persisted -> {target}")


# ----- Section B: compaction (no LLM) ---------------------------------


async def section_compact() -> None:
    print("\n=== B. Compaction summary preserves recent turns ===")
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=600,
        compact_ratio=0.4,
        handoff_ratio=0.99,
        keep_recent_messages=2,
        artifact_dir=str(ARTIFACT_DIR / "compact"),
    )
    manager = ContextManager(policy)
    history: list[LLMMessage] = []
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        history.append(LLMMessage(role=role, content=[TextBlock(text=f"step {i} " + "lorem " * 25)]))
    new_messages, manifest = manager.assemble(history)
    print(f"pressure={manifest.pressure} kept_messages={len(new_messages)} summary_present={manifest.compaction_summary is not None}")
    if manifest.compaction_summary:
        first_lines = "\n  ".join(manifest.compaction_summary.splitlines()[:3])
        print(f"  summary head:\n  {first_lines}")
    _persist("compact_manifest", manifest.model_dump())


# ----- Section C: handoff (no LLM) ------------------------------------


async def section_handoff() -> None:
    print("\n=== C. Handoff file is recoverable ===")
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=200,
        handoff_ratio=0.4,
        compact_ratio=0.3,
        offload_ratio=0.2,
        trim_ratio=0.1,
        keep_recent_messages=2,
        artifact_dir=str(ARTIFACT_DIR / "handoff"),
    )
    manager = ContextManager(policy)
    history = [LLMMessage(role="user", content=[TextBlock(text=f"plan step {i} " + "context " * 40)]) for i in range(6)]
    _, manifest = manager.assemble(history)
    print(f"pressure={manifest.pressure} handoff_path={manifest.handoff_path}")
    assert manifest.handoff_path is not None
    restored = rebuild_from_handoff(manifest.handoff_path)
    print(f"  rebuilt {len(restored)} messages from handoff file")


# ----- Section D: live LLM run with policy enabled --------------------


async def section_live_run() -> None:
    provider, model = _select_provider()
    if provider is None or model is None:
        print("\n=== D. SKIP live run (no API key in .env) ===")
        return
    print(f"\n=== D. Live LLM run with ContextPolicy (provider={provider} model={model}) ===")
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=4_000,
        trim_ratio=0.5,
        offload_ratio=0.6,
        compact_ratio=0.7,
        handoff_ratio=0.95,
        keep_recent_messages=4,
        max_tool_output_tokens=500,
        artifact_dir=str(ARTIFACT_DIR / "live"),
    )
    config = AgentConfig(
        name="context-aware-agent",
        provider=provider,  # type: ignore[arg-type]
        model=model,
        max_turns=2,
        context_policy=policy,
    )
    agent = Agent(config, tool_registry=ToolRegistry(), tool_executor=ToolExecutor(ToolRegistry()))
    result = await agent.run("Reply with the single word 'context-ok' and nothing else.")
    print(f"output={result.output!r}")
    print(f"context_manifests={len(result.context_manifests)}")
    if result.context_manifests:
        first = result.context_manifests[0]
        print(f"  first manifest: pressure={first.pressure} estimated_tokens={first.estimated_tokens}")
    _persist(
        "live_run_summary",
        {
            "provider": provider,
            "model": model,
            "output": result.output,
            "manifests": [m.model_dump() for m in result.context_manifests],
            "stop_reason": result.stop_reason.model_dump() if result.stop_reason else None,
        },
    )


async def main() -> int:
    await section_offload()
    await section_compact()
    await section_handoff()
    await section_live_run()
    print("\nAll sections complete. Artifacts under ./artifacts/context/")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
