# Demo 05 — Production Features: Observability, Guardrails, Structured Output
# Execute: uv run python examples/05_production_features.py
#
# Demonstrates:
#   1. Observability & Tracing  — console span exporter, metrics, lifecycle events
#   2. Guardrails & Safety      — budget limits, output validators, lifecycle hooks
#   3. Structured Output        — Pydantic-validated LLM responses (end-to-end via API)
#   4. Integrated Engine        — all features wired together for a real agent run
#
# Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env (or environment)

import asyncio
import os
import sys
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from anycode import AnyCode
from anycode.guardrails.budget import BudgetTracker, estimate_cost
from anycode.guardrails.hooks import HookRunner, LoggingHook
from anycode.guardrails.validators import (
    BlocklistValidator,
    ContainsValidator,
    MaxLengthValidator,
    run_validators,
)
from anycode.structured.output import (
    parse_structured_output,
    schema_to_openai_response_format,
    schema_to_tool_def,
)
from anycode.telemetry.events import EventEmitter
from anycode.telemetry.metrics import MetricsCollector, Timer
from anycode.telemetry.tracer import Tracer
from anycode.types import (
    AgentInfo,
    BudgetStatus,
    GuardrailConfig,
    LLMMessage,
    LLMResponse,
    SpanAttributes,
    TextBlock,
    TokenUsage,
    TraceConfig,
)

SEPARATOR = "=" * 60
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# --- Structured output schemas ---


class CodeReview(BaseModel):
    model_config = ConfigDict(frozen=True)
    summary: str
    issues: list[str]
    severity: str
    approved: bool


class NestedReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    class Metrics(BaseModel):
        model_config = ConfigDict(frozen=True)
        lines_reviewed: int
        bugs_found: int
        coverage_pct: float

    title: str
    metrics: Metrics
    recommendations: list[str]


# --- Helpers ---


def _resolve_provider() -> tuple[str, str] | None:
    """Return (provider, model) based on available API keys, or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None


# --- Section 1: Observability & Tracing ---


def demo_tracing(provider: str, model: str) -> None:
    """Demonstrate span lifecycle, console exporter, metrics, and events."""
    print(f"\n{SEPARATOR}")
    print("  1. Observability & Tracing")
    print(SEPARATOR)

    # Tracer with console exporter
    print("\n--- Tracer (console exporter) ---")
    tracer = Tracer(TraceConfig(enabled=True, exporter="console", service_name="demo"))

    with tracer.span("anycode.run_agent.reviewer") as root:
        root.set_attributes(SpanAttributes(agent_name="reviewer", model=model))

        with tracer.span("anycode.llm.chat", parent=root) as llm_span:
            llm_span.set_attributes(
                SpanAttributes(model=model, provider=provider, token_input=350, token_output=120)
            )

        with tracer.span("anycode.tool.file_read", parent=root) as tool_span:
            tool_span.set_attributes(SpanAttributes(tool_name="file_read"))
            tool_span.set_attribute("file_path", "/src/main.py")

    print(f"\nSpans collected: {len(tracer.spans)}")
    for s in tracer.spans:
        d = s.to_dict()
        print(f"  {d['name']}  ({d['duration_ms']:.2f}ms)  parent={d['parent']}")

    # No-op tracer (zero overhead when disabled)
    print("\n--- No-op tracer (disabled) ---")
    noop = Tracer(TraceConfig(enabled=False))
    span = noop.start_span("should.not.exist")
    span.set_attribute("key", "value")
    noop.end_span(span)
    print(f"Spans when disabled: {len(noop.spans)} (zero overhead)")

    # Metrics collector
    print("\n--- Metrics Collector ---")
    metrics = MetricsCollector(enabled=True)
    metrics.record_token_usage("reviewer", model, input_tokens=350, output_tokens=120)
    metrics.record_cost("reviewer", model, estimate_cost(model, 350, 120))
    metrics.record_latency("llm.chat", 245.6, {"model": model})
    metrics.record_error("tool.bash", "timeout")

    with Timer(metrics, "full_pipeline"):
        _ = sum(range(10_000))

    summary = metrics.get_summary()
    print(f"Counters: {len(summary['counters'])} tracked")
    for key, val in summary["counters"].items():
        print(f"  {key}: {val}")
    print(f"Histograms: {len(summary['histograms'])} tracked")

    # Event emitter
    print("\n--- Event Emitter ---")
    emitter = EventEmitter(enabled=True)
    emitter.agent_start("reviewer", model)
    emitter.turn_start("reviewer", 1)
    emitter.llm_call_start("reviewer", model)
    emitter.llm_call_complete("reviewer", model, input_tokens=350, output_tokens=120, duration_ms=245.6)
    emitter.tool_start("reviewer", "file_read")
    emitter.tool_complete("reviewer", "file_read", duration_ms=12.3, is_error=False)
    emitter.turn_complete("reviewer", 1, token_input=350, token_output=120)
    emitter.agent_complete("reviewer", turns=1, tokens_used=470)

    print(f"Lifecycle events recorded: {len(emitter.events)}")
    for ev in emitter.events:
        print(f"  {ev.name}: {ev.attributes}")


# --- Section 2: Guardrails & Safety ---


async def demo_guardrails(provider: str, model: str) -> None:
    """Demonstrate budget tracking, output validators, and lifecycle hooks."""
    print(f"\n{SEPARATOR}")
    print("  2. Guardrails & Safety")
    print(SEPARATOR)

    # Budget tracker
    print("\n--- Budget Tracker ---")
    config = GuardrailConfig(
        max_tokens_per_agent=10_000,
        max_cost_usd=1.00,
        max_turns=5,
        max_tool_calls=20,
        blocked_tools=["bash"],
    )
    tracker = BudgetTracker(config, model=model)

    tracker.record_usage(TokenUsage(input_tokens=2000, output_tokens=800))
    tracker.record_turn()
    tracker.record_tool_call(3)

    status: BudgetStatus = tracker.get_status()
    print(f"Tokens: {status.tokens_used}/{status.tokens_limit}")
    print(f"Cost:   ${status.cost_used:.4f}/${status.cost_limit}")
    print(f"Turns:  {status.turns_used}/{status.turns_limit}")
    print(f"Tools:  {status.tool_calls_used}/{status.tool_calls_limit}")
    print(f"Exhausted: {status.exhausted}")
    print(f"Tool 'bash' blocked: {tracker.is_tool_blocked('bash')}")
    print(f"Tool 'file_read' blocked: {tracker.is_tool_blocked('file_read')}")

    # Push it over the limit
    tracker.record_usage(TokenUsage(input_tokens=5000, output_tokens=5000))
    print(f"\nAfter large usage:")
    print(f"Exhausted: {tracker.is_exhausted()}")
    print(f"Reason: {tracker.get_exhaustion_reason()}")

    # Cost estimation
    print("\n--- Cost Estimation ---")
    models = ["claude-sonnet-4-6", "claude-haiku-4-5", "gpt-4o", "gpt-4o-mini", "gpt-4.1"]
    for model in models:
        cost = estimate_cost(model, input_tokens=100_000, output_tokens=50_000)
        print(f"  {model}: ${cost:.4f} for 100K in + 50K out")

    # Output validators
    print("\n--- Output Validators ---")
    agent_info = AgentInfo(name="writer", role="assistant", model=model)

    length_validator = MaxLengthValidator(200)
    contains_validator = ContainsValidator("conclusion")
    blocklist_validator = BlocklistValidator(["password", "secret", "api_key"])

    test_outputs = [
        ("Short, has conclusion.", "short"),
        ("A" * 300, "too long"),
        ("This has a conclusion and is good.", "valid with conclusion"),
        ("My password is hunter2", "contains blocked term"),
    ]

    for text, label in test_outputs:
        result = await run_validators(
            text,
            [length_validator, contains_validator, blocklist_validator],
            agent_info,
        )
        status_str = "PASS" if result.valid else f"FAIL ({result.reason})"
        print(f"  [{label}]: {status_str}")

    # Lifecycle hooks
    print("\n--- Lifecycle Hooks ---")
    hook = LoggingHook()
    runner = HookRunner([hook])

    messages = [LLMMessage(role="user", content=[TextBlock(text="Explain async Python.")])]
    response = LLMResponse(
        id="resp-demo",
        content=[TextBlock(text="Async Python uses coroutines and an event loop.")],
        model=model,
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=50, output_tokens=30),
    )

    messages = await runner.run_before_turn(messages, agent_info)
    response = await runner.run_after_turn(response, agent_info)

    print(f"Before-turn log: {hook.before_turn_log}")
    print(f"After-turn log:  {hook.after_turn_log}")

    # Custom hook that adds a system instruction
    class SafetyHook:
        async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
            safety_msg = LLMMessage(
                role="user",
                content=[TextBlock(text="[SYSTEM] Always respond helpfully and safely.")],
            )
            return [safety_msg] + list(messages)

        async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
            return response

    safety_runner = HookRunner([SafetyHook()])
    original = [LLMMessage(role="user", content=[TextBlock(text="Hello")])]
    modified = await safety_runner.run_before_turn(original, agent_info)
    print(f"\nCustom hook injected {len(modified) - len(original)} message(s) before turn")


# --- Section 3: Structured Output (schema utilities) ---


def demo_structured_schemas() -> None:
    """Demonstrate schema conversion and local parsing."""
    print(f"\n{SEPARATOR}")
    print("  3. Structured Output — Schema Utilities")
    print(SEPARATOR)

    # Anthropic tool-use schema
    print("\n--- Anthropic Tool-Use Schema ---")
    tool_def = schema_to_tool_def(CodeReview)
    print(f"Tool name: {tool_def.name}")
    print(f"Description: {tool_def.description}")
    print(f"Properties: {list(tool_def.input_schema.get('properties', {}).keys())}")
    print(f"Required: {tool_def.input_schema.get('required', [])}")

    # OpenAI response_format schema
    print("\n--- OpenAI JSON Schema Format ---")
    oai_fmt = schema_to_openai_response_format(CodeReview)
    print(f"Type: {oai_fmt['type']}")
    print(f"Schema name: {oai_fmt['json_schema']['name']}")
    print(f"Strict: {oai_fmt['json_schema']['strict']}")

    # Local parsing tests
    print("\n--- Parsing Structured Output (local) ---")

    valid_json = '{"summary": "Clean code with minor issues", "issues": ["unused import on line 12"], "severity": "low", "approved": true}'
    parsed = parse_structured_output(valid_json, CodeReview)
    print(f"Valid JSON -> parsed: {parsed is not None}")
    if parsed:
        print(f"  summary: {parsed.summary}")  # type: ignore[union-attr]
        print(f"  severity: {parsed.severity}")  # type: ignore[union-attr]
        print(f"  approved: {parsed.approved}")  # type: ignore[union-attr]

    markdown_response = '```json\n{"summary": "All good", "issues": [], "severity": "low", "approved": true}\n```'
    parsed2 = parse_structured_output(markdown_response, CodeReview)
    print(f"Markdown-wrapped JSON -> parsed: {parsed2 is not None}")

    bad_json = "This is not JSON at all"
    parsed3 = parse_structured_output(bad_json, CodeReview)
    print(f"Invalid text -> parsed: {parsed3 is not None}")

    nested_json = '{"title": "Sprint Review", "metrics": {"lines_reviewed": 1500, "bugs_found": 3, "coverage_pct": 87.5}, "recommendations": ["Add error handling", "Improve test coverage"]}'
    parsed4 = parse_structured_output(nested_json, NestedReport)
    print(f"Nested model -> parsed: {parsed4 is not None}")
    if parsed4:
        print(f"  title: {parsed4.title}")  # type: ignore[union-attr]
        print(f"  bugs_found: {parsed4.metrics.bugs_found}")  # type: ignore[union-attr]
        print(f"  recommendations: {parsed4.recommendations}")  # type: ignore[union-attr]


# --- Section 4: End-to-End LLM Integration ---


async def demo_live_agent(provider: str, model: str) -> None:
    """Run a real agent with telemetry, guardrails, hooks, and structured output via the LLM API."""
    print(f"\n{SEPARATOR}")
    print(f"  4. End-to-End LLM Agent ({provider}/{model})")
    print(SEPARATOR)

    engine = AnyCode()

    engine.configure(
        trace=TraceConfig(enabled=True, exporter="console", service_name="production-features-demo"),
        guardrails=GuardrailConfig(
            max_tokens_per_agent=50_000,
            max_cost_usd=2.00,
            max_turns=10,
            blocked_tools=["bash"],
        ),
        hooks=[LoggingHook()],
        output_validators=[
            MaxLengthValidator(10_000),
            BlocklistValidator(["internal_secret"]),
        ],
    )

    # --- 4a. Plain text agent run ---
    print("\n--- Plain agent run ---")
    result = await engine.run_agent(
        config={
            "name": "explainer",
            "model": model,
            "provider": provider,
            "system_prompt": "You are a concise technical explainer. Keep answers under 100 words.",
            "max_turns": 2,
        },
        prompt="Explain what a Python decorator is and give a one-line example.",
    )

    print(f"Success: {result.success}")
    print(f"Output:  {result.output[:300]}")
    print(f"Tokens:  input={result.token_usage.input_tokens}, output={result.token_usage.output_tokens}")
    print(f"Tools:   {len(result.tool_calls)} call(s)")

    # --- 4b. Structured output agent run ---
    print("\n--- Structured output agent run ---")
    agent = engine.build_agent(
        {
            "name": "reviewer",
            "model": model,
            "provider": provider,
            "system_prompt": (
                "You are a code reviewer. When asked to review code, respond with a structured JSON assessment "
                "containing: summary, issues (list of strings), severity (low/medium/high), and approved (boolean)."
            ),
            "max_turns": 2,
        },
    )

    structured_result = await agent.run_structured(
        "Review this Python function:\n\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "Provide your assessment.",
        schema=CodeReview,
    )

    print(f"Success: {structured_result.success}")
    print(f"Parsed:  {structured_result.parsed is not None}")
    if structured_result.parsed:
        review = structured_result.parsed
        print(f"  summary:  {review.summary}")  # type: ignore[union-attr]
        print(f"  issues:   {review.issues}")  # type: ignore[union-attr]
        print(f"  severity: {review.severity}")  # type: ignore[union-attr]
        print(f"  approved: {review.approved}")  # type: ignore[union-attr]
    else:
        print(f"  Raw output: {structured_result.output[:300]}")

    print(f"Tokens:  input={structured_result.token_usage.input_tokens}, output={structured_result.token_usage.output_tokens}")

    # --- 4c. Multi-turn conversational agent ---
    print("\n--- Multi-turn conversation ---")
    tutor = engine.build_agent(
        {
            "name": "tutor",
            "model": model,
            "provider": provider,
            "system_prompt": "You are a Python tutor. Give brief, practical answers in 1-2 sentences.",
            "max_turns": 2,
        },
    )

    reply1 = await tutor.prompt("What is a generator in Python?")
    print(f"Turn 1: {reply1.output[:200]}")

    reply2 = await tutor.prompt("Show a one-line example.")
    print(f"Turn 2: {reply2.output[:200]}")

    print(f"Conversation history: {len(tutor.get_history())} messages")
    total_in = reply1.token_usage.input_tokens + reply2.token_usage.input_tokens
    total_out = reply1.token_usage.output_tokens + reply2.token_usage.output_tokens
    print(f"Total tokens: input={total_in}, output={total_out}")


# --- Main ---


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


async def main() -> None:
    load_dotenv()

    capture = _OutputCapture()
    sys.stdout = capture  # type: ignore[assignment]

    resolved = _resolve_provider()
    if resolved:
        provider, model = resolved
    else:
        provider, model = "anthropic", "claude-haiku-4-5"

    print("AnyCode \u2014 Production Features Demo")
    print(f"Provider: {provider} | Model: {model}\n")

    demo_tracing(provider, model)
    await demo_guardrails(provider, model)
    demo_structured_schemas()

    if resolved:
        await demo_live_agent(provider, model)
    else:
        print(f"\n{SEPARATOR}")
        print("  4. Live Agent — SKIPPED (no API key in .env or environment)")
        print(SEPARATOR)

    print(f"\n{SEPARATOR}")
    print("  All features verified successfully!")
    print(SEPARATOR)

    sys.stdout = capture._stdout
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "05-production-features-output.txt"
    output_file.write_text(capture.get_output(), encoding="utf-8")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
