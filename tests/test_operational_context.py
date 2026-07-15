"""Identity propagation, external policy, and GenAI telemetry privacy tests."""

from __future__ import annotations

from collections.abc import AsyncIterable

import pytest
from pydantic import BaseModel, ValidationError

from anycode.contracts import PolicyDecision, PolicyObligation
from anycode.core.runner import AgentRunner
from anycode.helpers.uuid7 import uuid7
from anycode.identity.context import DelegationGrant, ExecutionContext
from anycode.identity.policy import InMemoryPolicyAuditSink, PolicyEnforcer, PolicyRequest
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.telemetry.genai import BoundedTelemetryBuffer, GenAITelemetryConfig, GenAITelemetryMapper
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import LLMChatOptions, LLMMessage, LLMResponse, LLMStreamOptions, RunnerOptions, StreamEvent, TextBlock, ToolResult, ToolUseContext


def _context() -> ExecutionContext:
    return ExecutionContext(
        principal="user:42",
        subject="customer:7",
        workload_identity="spiffe://example/worker",
        tenant_scope="tenant-a",
        delegation=(DelegationGrant(delegator="user:42", delegatee="agent:writer", scopes=("artifact:write",)),),
        classification="confidential",
        allowed_regions=("eu-west-1",),
        required_region="eu-west-1",
        credential_references=("env:OPENAI_API_KEY",),
        trace_id="a" * 32,
        attributes={"department": "research"},
    )


def test_execution_context_rejects_raw_credentials_but_accepts_references() -> None:
    context = _context()
    assert context.audit_attributes() == {
        "principal": "user:42",
        "subject": "customer:7",
        "workload_identity": "spiffe://example/worker",
        "tenant_scope": "tenant-a",
        "classification": "confidential",
        "region": "eu-west-1",
    }
    assert "OPENAI_API_KEY" in context.policy_json()

    with pytest.raises(ValidationError):
        ExecutionContext(principal="user", attributes={"api_key": "plain-value"})
    with pytest.raises(ValidationError):
        ExecutionContext(principal="user", attributes={"note": "Bearer abcdefghijklmnop"})


class AllowAdapter:
    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome="allow",
            policy_version="opa/bundle-7",
            reason_codes=("tenant_allowed",),
            obligations=(PolicyObligation(type="record_evidence", parameters={"level": "metadata"}),),
            correlation_id=request.correlation_id,
            generation=request.generation,
            attempt=request.attempt,
        )


class BrokenAdapter:
    async def decide(self, request: PolicyRequest) -> PolicyDecision:
        del request
        raise RuntimeError("policy endpoint unavailable")


def _policy_request() -> PolicyRequest:
    return PolicyRequest(
        run_id="run-1",
        task_id="task-1",
        action="execute",
        resource="tool:publish",
        boundary="tool",
        context=_context(),
        correlation_id="corr-1",
        input={"document_id": "doc-1"},
    )


async def test_policy_enforcer_applies_obligations_and_audits_metadata() -> None:
    sink = InMemoryPolicyAuditSink()

    async def record_evidence(obligation: PolicyObligation, request: PolicyRequest) -> bool:
        return obligation.parameters["level"] == "metadata" and request.context.tenant_scope == "tenant-a"

    enforcer = PolicyEnforcer(AllowAdapter(), audit_sink=sink, obligation_handlers={"record_evidence": record_evidence})
    result = await enforcer.enforce(_policy_request())

    assert result.allowed and result.applied_obligations == ("record_evidence",)
    assert len(sink.events) == 1
    assert sink.events[0].context["tenant_scope"] == "tenant-a"
    assert "input" not in sink.events[0].model_dump()


async def test_required_policy_fails_closed_and_unfulfilled_obligations_block() -> None:
    failed = await PolicyEnforcer(BrokenAdapter(), fail_closed=True).enforce(_policy_request())
    unfulfilled = await PolicyEnforcer(AllowAdapter()).enforce(_policy_request())

    assert not failed.allowed and failed.decision.reason_codes == ("policy_unavailable",)
    assert not unfulfilled.allowed and unfulfilled.error is not None
    assert unfulfilled.error.code == "policy_obligation_unfulfilled"


class RecordingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__(responses=[FakeResponse(tool_calls=(("capture", {}),)), FakeResponse(text="done")])
        self.contexts: list[ExecutionContext | None] = []

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        self.contexts.append(options.execution_context)
        return await super().chat(messages, options)

    def stream(self, messages: list[LLMMessage], options: LLMStreamOptions) -> AsyncIterable[StreamEvent]:
        return super().stream(messages, options)


async def test_runner_propagates_execution_context_to_model_and_tool_boundaries() -> None:
    seen_tool_contexts: list[ExecutionContext | None] = []

    class EmptyInput(BaseModel):
        pass

    async def capture(_input: EmptyInput, context: ToolUseContext) -> ToolResult:
        seen_tool_contexts.append(context.execution_context)
        return ToolResult(data="captured")

    registry = ToolRegistry()
    registry.register(define_tool(name="capture", description="capture context", input_model=EmptyInput, execute=capture))
    adapter = RecordingAdapter()
    context = _context()
    runner = AgentRunner(
        adapter,
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake", agent_name="context-test", max_turns=3, execution_context=context),
    )

    await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])

    assert adapter.contexts and all(item == context for item in adapter.contexts)
    assert seen_tool_contexts == [context]


def test_metadata_telemetry_excludes_all_content_and_credentials() -> None:
    mapper = GenAITelemetryMapper(GenAITelemetryConfig(profile="metadata"))
    record = mapper.map(
        "model.completed",
        {
            "provider": "openai",
            "model": "gpt-test",
            "input_tokens": 12,
            "output_tokens": 3,
            "prompt": "private prompt",
            "response": "private response",
            "tool_arguments": {"path": "secret.txt"},
            "artifact_body": "private artifact",
            "api_key": "plain-value",
        },
        context=_context(),
    )
    assert record is not None
    serialized = record.model_dump_json()

    assert record.attributes["gen_ai.operation.name"] == "chat"
    assert record.attributes["gen_ai.provider.name"] == "openai"
    assert record.attributes["gen_ai.usage.input_tokens"] == 12
    assert record.attributes["gen_ai.usage.output_tokens"] == 3
    assert record.trace_id == "a" * 32
    for forbidden in ("private prompt", "private response", "secret.txt", "private artifact", "plain-value", "OPENAI_API_KEY"):
        assert forbidden not in serialized


def test_redacted_full_off_hashing_truncation_and_bounded_buffer() -> None:
    secret = "sk-1234567890abcdef1234567890"
    redacted = GenAITelemetryMapper(GenAITelemetryConfig(profile="redacted")).map("tool.completed", {"tool_output": secret})
    full = GenAITelemetryMapper(GenAITelemetryConfig(profile="full")).map("tool.completed", {"tool_output": secret})
    off = GenAITelemetryMapper(GenAITelemetryConfig(profile="off")).map("tool.completed", {})
    hashed = GenAITelemetryMapper(
        GenAITelemetryConfig(profile="metadata", hash_fields=("session_id",), max_string_length=16)
    ).map("run.started", {"session_id": "customer-session", "safe": "x" * 30})

    assert redacted is not None and secret not in redacted.model_dump_json()
    assert full is not None and secret not in full.model_dump_json()
    assert off is None
    assert hashed is not None and str(hashed.attributes["session_id"]).startswith("sha256:")
    assert hashed.attributes["safe"] == "x" * 16

    buffer = BoundedTelemetryBuffer(capacity=1)
    buffer.append(redacted)
    buffer.append(full)
    assert buffer.dropped == 1 and buffer.records == (full,)


async def test_telemetry_export_failure_is_isolated_and_retriable() -> None:
    record = GenAITelemetryMapper().map("run.started", {"run_id": "run-1"})
    assert record is not None
    buffer = BoundedTelemetryBuffer(capacity=2)
    buffer.append(record)

    async def fail(_batch: object) -> None:
        raise RuntimeError("collector offline")

    exported: list[object] = []

    async def succeed(batch: object) -> None:
        exported.append(batch)

    assert not await buffer.flush(fail)
    assert buffer.records == (record,) and buffer.export_failures == 1
    assert await buffer.flush(succeed)
    assert not buffer.records and len(exported) == 1
