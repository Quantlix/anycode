# Demo 20 — Verification Sensors & Quality Gates
# Execute: uv run python examples/20_quality_gates.py
"""
Demonstrates verification sensors:
  - Built-in computational sensors (ruff, schema)
  - QualityGate decisions: pass / warn / retry / block / escalate
  - Inferential (LLM-backed) sensor that judges output quality

Real LLM calls are made when ANTHROPIC_API_KEY or OPENAI_API_KEY is present.
Decisions and sensor results are persisted to ./artifacts/verification/ for
audit and replay.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from anycode import (
    Agent,
    AgentConfig,
    QualityGate,
    Sensor,
    SensorContext,
    ToolExecutor,
    ToolRegistry,
    VerificationResult,
    VerificationSensorConfig,
    create_adapter,
    ruff_sensor,
    schema_sensor,
)

load_dotenv()

ARTIFACT_DIR = Path("artifacts/verification")
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


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    summary: str


# ----- Section A: built-in computational gate (no LLM) ----------------


async def section_computational_gate() -> None:
    print("\n=== A. Computational gate (ruff + schema) ===")
    gate = QualityGate(
        [
            ruff_sensor("src/anycode/verification/", phases=("after_task",)),
            schema_sensor(WeatherReport, phases=("after_task",)),
        ]
    )
    ctx_good = SensorContext(
        phase="after_task",
        agent_name="demo",
        run_id="run-good",
        output='{"city": "Paris", "temperature_celsius": 18.5, "summary": "mild and clear"}',
    )
    decision = await gate.evaluate(ctx_good)
    print(f"good output -> {decision.outcome}: {decision.message}")
    for r in decision.results:
        print(f"  - {r.sensor_name}: passed={r.passed} severity={r.severity}")

    ctx_bad = SensorContext(
        phase="after_task",
        agent_name="demo",
        run_id="run-bad",
        output='{"city": "Paris"}',  # missing required fields
    )
    decision_bad = await gate.evaluate(ctx_bad)
    print(f"bad output  -> {decision_bad.outcome}: {decision_bad.message}")
    _persist(
        "computational_gate",
        {
            "good": decision.model_dump(),
            "bad": decision_bad.model_dump(),
        },
    )


# ----- Section B: inferential (LLM judge) sensor ----------------------


async def section_inferential_sensor() -> None:
    provider, model = _select_provider()
    if provider is None or model is None:
        print("\n=== B. SKIP inferential sensor (no API key in .env) ===")
        return
    print(f"\n=== B. Inferential sensor with provider={provider} model={model} ===")
    adapter = await create_adapter(provider)

    async def _judge(ctx: SensorContext) -> VerificationResult:
        from anycode import LLMChatOptions, LLMMessage, TextBlock

        prompt = (
            "You are a strict reviewer. Reply with only 'PASS' or 'FAIL'. "
            f"Does the following output answer 'What is 2 + 2?' correctly?\n\nOutput: {ctx.output!r}"
        )
        response = await adapter.chat(
            [LLMMessage(role="user", content=[TextBlock(text=prompt)])],
            LLMChatOptions(model=model, max_tokens=10, temperature=0.0),
        )
        verdict = "".join(b.text for b in response.content if hasattr(b, "text")).strip().upper()
        passed = verdict.startswith("PASS")
        return VerificationResult(
            sensor_name="llm_judge",
            kind="inferential",
            passed=passed,
            severity="info" if passed else "error",
            message=f"judge said: {verdict}",
            feedback_for_agent=None if passed else "Re-answer with a correct numeric result.",
        )

    judge = Sensor(
        config=VerificationSensorConfig(name="llm_judge", kind="inferential", phases=("after_task",)),
        fn=_judge,
    )
    gate = QualityGate([judge])

    correct_ctx = SensorContext(phase="after_task", agent_name="demo", run_id="r1", output="The answer is 4.")
    wrong_ctx = SensorContext(phase="after_task", agent_name="demo", run_id="r2", output="It is 22.")

    correct_decision = await gate.evaluate(correct_ctx)
    print(f"correct answer -> {correct_decision.outcome}: {correct_decision.message}")
    wrong_decision = await gate.evaluate(wrong_ctx)
    print(f"wrong answer   -> {wrong_decision.outcome}: {wrong_decision.message}")
    _persist(
        "inferential_gate",
        {
            "correct": correct_decision.model_dump(),
            "wrong": wrong_decision.model_dump(),
        },
    )


# ----- Section C: live agent run gated by schema ----------------------


async def section_live_agent_with_gate() -> None:
    provider, model = _select_provider()
    if provider is None or model is None:
        print("\n=== C. SKIP live agent (no API key in .env) ===")
        return
    print(f"\n=== C. Live agent gated by schema sensor (provider={provider} model={model}) ===")
    config = AgentConfig(
        name="weather-agent",
        provider=provider,  # type: ignore[arg-type]
        model=model,
        max_turns=1,
        system_prompt=(
            "Always respond with a single JSON object containing fields "
            "city (string), temperature_celsius (number), summary (string). "
            "Do not include any other text or code fences."
        ),
    )
    agent = Agent(config, tool_registry=ToolRegistry(), tool_executor=ToolExecutor(ToolRegistry()))
    result = await agent.run("Report the current weather for Paris.")
    print(f"agent output: {result.output!r}")

    gate = QualityGate([schema_sensor(WeatherReport)])
    decision = await gate.evaluate(
        SensorContext(
            phase="after_task",
            agent_name=config.name,
            run_id="live",
            output=result.output,
        )
    )
    print(f"gate decision -> {decision.outcome}: {decision.message}")
    _persist(
        "live_agent_gate",
        {
            "output": result.output,
            "decision": decision.model_dump(),
            "stop_reason": result.stop_reason.model_dump() if result.stop_reason else None,
        },
    )


async def main() -> int:
    await section_computational_gate()
    await section_inferential_sensor()
    await section_live_agent_with_gate()
    print("\nAll sections complete. Artifacts under ./artifacts/verification/")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
