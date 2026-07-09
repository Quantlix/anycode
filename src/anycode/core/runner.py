"""Agentic dialogue driver — manages LLM interactions, tool dispatch, and turn looping."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from anycode.checkpoint.serializer import _serialize_message
from anycode.constants import DEFAULT_TURN_LIMIT, HANDOFF_TOOL_NAME, MAX_VALIDATION_RETRIES, MS_PER_SECOND
from anycode.context.profiles import resolve_profile
from anycode.core.context_manager import ContextManager
from anycode.core.lifecycle import LifecycleEmitter, LifecycleEvent, LifecycleListener, LoopDetector, fingerprint_call
from anycode.core.stop_reason import (
    budget_exceeded as stop_budget_exceeded,
)
from anycode.core.stop_reason import (
    doom_loop as stop_doom_loop,
)
from anycode.core.stop_reason import (
    max_turns as stop_max_turns,
)
from anycode.core.stop_reason import (
    provider_unavailable as stop_provider_unavailable,
)
from anycode.core.stop_reason import (
    success as stop_success,
)
from anycode.core.stop_reason import (
    unknown as stop_unknown,
)
from anycode.guardrails.budget import BudgetTracker
from anycode.guardrails.hooks import HookRunner
from anycode.guardrails.validators import run_validators
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.helpers.uuid7 import uuid7
from anycode.providers.resilience import ProviderUnavailableError
from anycode.runstore.store import FilesystemRunStore
from anycode.structured.output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    build_retry_prompt,
    parse_structured_output,
    schema_to_tool_def,
)
from anycode.telemetry.tracer import Span, Tracer
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentInfo,
    ContentBlock,
    ContextManifest,
    ContextPolicy,
    DurabilityConfig,
    GuardrailConfig,
    HandoffRequest,
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    OutputValidator,
    QualityGateDecision,
    RunnerOptions,
    RunResult,
    SpanAttributes,
    StopReason,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
    TurnCheckpoint,
    TurnHook,
    VerificationResult,
    VerificationSensorConfig,
    WakeCondition,
)
from anycode.verification.gate import QualityGate
from anycode.verification.registry import build_sensors
from anycode.verification.sensor import SensorContext


def _pull_text(blocks: list[ContentBlock]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def _filter_tool_calls(blocks: list[ContentBlock]) -> list[ToolUseBlock]:
    return [b for b in blocks if isinstance(b, ToolUseBlock)]


class AgentRunner:
    """Orchestrates the full model-tool-model turn loop until completion or limit."""

    def __init__(
        self,
        adapter: LLMAdapter,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        options: RunnerOptions,
        *,
        tracer: Tracer | None = None,
        guardrail_config: GuardrailConfig | None = None,
        hooks: list[TurnHook] | None = None,
        output_validators: list[OutputValidator] | None = None,
        output_schema: type[BaseModel] | None = None,
        lifecycle_listeners: list[LifecycleListener] | None = None,
        context_policy: ContextPolicy | None = None,
        durability: DurabilityConfig | None = None,
        run_store: FilesystemRunStore | None = None,
        resume_from: TurnCheckpoint | None = None,
    ) -> None:
        self._adapter = adapter
        self._registry = tool_registry
        self._executor = tool_executor
        self._options = options
        self._turn_limit = options.max_turns or DEFAULT_TURN_LIMIT
        self._tracer = tracer or Tracer()
        self._budget = BudgetTracker(guardrail_config, model=options.model)
        self._hook_runner = HookRunner(hooks)
        self._validators = list(output_validators) if output_validators else []
        self._output_schema = output_schema
        self._lifecycle_listeners = list(lifecycle_listeners or [])
        self._context_manager: ContextManager | None = (
            ContextManager(context_policy, provider=adapter.name, model=options.model)
            if context_policy and (context_policy.enabled or context_policy.mode == "auto")
            else None
        )
        self._sensor_configs: tuple[VerificationSensorConfig, ...] = tuple(options.verification or ())
        self._gate: QualityGate | None = None
        if self._sensor_configs:
            self._gate = QualityGate(build_sensors(self._sensor_configs))
        self._durability = durability if (durability and durability.enabled) else None
        self._run_store: FilesystemRunStore | None = None
        if self._durability is not None:
            self._run_store = run_store or FilesystemRunStore(self._durability.run_root)
        self._resume_from = resume_from

    @property
    def budget_tracker(self) -> BudgetTracker:
        return self._budget

    async def run(
        self,
        messages: list[LLMMessage],
        on_message: Callable[[LLMMessage], None] | None = None,
    ) -> RunResult:
        fallback = RunResult(messages=[], output="", tool_calls=[], token_usage=EMPTY_USAGE, turns=0)
        async for event in self.stream(messages, on_message=on_message):
            if event.type == "done":
                return event.data  # type: ignore[return-value]
        return fallback

    async def stream(
        self,
        seed_messages: list[LLMMessage],
        on_message: Callable[[LLMMessage], None] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        conversation = list(seed_messages)
        cumulative_usage: TokenUsage = EMPTY_USAGE
        tool_calls: list[ToolCallRecord] = []
        context_manifests: list[ContextManifest] = []
        last_output = ""
        turn_count = 0
        validation_retries = 0
        structured_retries = 0
        base_retries = 0
        agent_name = self._options.agent_name or "runner"

        restored = self._resume_from
        run_id = restored.run_id if restored is not None else str(uuid7())
        emitter = LifecycleEmitter(
            run_id=run_id,
            agent_name=agent_name,
            listeners=self._lifecycle_listeners,
        )
        loop_detector = LoopDetector()
        terminal_stop: StopReason | None = None
        verification_results: list[VerificationResult] = []
        gate_decisions: list[QualityGateDecision] = []
        restored_lifecycle: list[object] = []
        verification_retries = 0
        max_verification_retries = MAX_VALIDATION_RETRIES

        if restored is not None:
            # Continue a durable run from its last turn boundary: history,
            # accounting, and loop-detection state all carry over.
            conversation = list(restored.messages)
            cumulative_usage = restored.token_usage
            turn_count = restored.turn
            last_output = restored.last_output
            base_retries = restored.retries
            self._budget.restore(restored.budget)
            loop_detector.restore_window(restored.loop_window)
            context_manifests = list(restored.context_manifests)
            verification_results = list(restored.verification_results)
            gate_decisions = list(restored.gate_decisions)
            restored_lifecycle = list(restored.lifecycle_events)
        seed_len = len(conversation) if restored is not None else len(seed_messages)

        store = self._run_store
        durability = self._durability
        if store is not None and durability is not None:
            if store.read_record(run_id) is None:
                store.create_run(run_id, agent_name=agent_name, model=self._options.model)
            else:
                store.update_status(run_id, "running")

            def _on_lifecycle(ev: LifecycleEvent) -> None:
                if store is not None:
                    store.append_event(run_id, "lifecycle", ev.model_dump(mode="json"))

            emitter.add_listener(_on_lifecycle)

        def _persist(kind: str, payload: dict[str, object]) -> None:
            if store is not None:
                store.append_event(run_id, kind, payload)  # type: ignore[arg-type]

        def _all_lifecycle() -> list[object]:
            return [*restored_lifecycle, *emitter.events]

        def _save_turn_checkpoint() -> None:
            if store is None or durability is None:
                return
            checkpoint = TurnCheckpoint(
                run_id=run_id,
                turn=turn_count,
                messages=list(conversation),
                token_usage=cumulative_usage,
                budget=self._budget.snapshot(),
                loop_window=loop_detector.export_window(),
                last_output=last_output,
                retries=base_retries + validation_retries + structured_retries + verification_retries,
                lifecycle_events=_all_lifecycle(),  # type: ignore[arg-type]
                context_manifests=list(context_manifests),
                verification_results=list(verification_results),
                gate_decisions=list(gate_decisions),
                created_at=datetime.now(UTC),
            )
            store.save_checkpoint(checkpoint, keep_last=durability.keep_last_checkpoints)
            _persist("checkpoint", {"turn": turn_count})

        all_defs = self._registry.to_tool_defs()
        active_defs = [d for d in all_defs if d.name in self._options.allowed_tools] if self._options.allowed_tools else all_defs

        if self._output_schema:
            structured_tool = schema_to_tool_def(self._output_schema)
            active_defs = list(active_defs) + [structured_tool] if active_defs else [structured_tool]

        cache_profile, _ = resolve_profile(provider=self._adapter.name, model=self._options.model)
        chat_params = LLMChatOptions(
            model=self._options.model,
            tools=active_defs if active_defs else None,
            max_tokens=self._options.max_tokens,
            temperature=self._options.temperature,
            system_prompt=self._options.system_prompt,
            enable_prompt_cache=cache_profile.supports_prompt_cache,
        )

        try:
            while turn_count < self._turn_limit:
                if self._budget.is_exhausted():
                    reason = self._budget.get_exhaustion_reason() or "Budget exhausted."
                    last_output = reason
                    terminal_stop = stop_budget_exceeded(reason)
                    yield StreamEvent(type="text", data=reason)
                    break

                turn_count += 1
                self._budget.record_turn()
                if store is not None:
                    # Liveness at turn start too: an LLM call can legitimately
                    # run for minutes, and watchdogs kill on stale heartbeats.
                    store.touch_heartbeat(run_id)

                ctx_info = self._build_agent_info()
                conversation = await self._hook_runner.run_before_turn(conversation, ctx_info)

                if emitter.phase != "executing":
                    emitter.transition("executing", metadata={"turn": turn_count})

                async with self._tracer.async_span(f"anycode.agent.{agent_name}.turn.{turn_count}") as turn_span:
                    turn_span.set_attributes(
                        SpanAttributes(
                            agent_name=agent_name,
                            model=self._options.model,
                            turn_number=turn_count,
                            phase="executing",
                        )
                    )

                    llm_start = time.monotonic()
                    if self._context_manager is not None:
                        tool_defs_text: str | None = None
                        if active_defs:
                            tool_defs_text = json.dumps(
                                [{"name": d.name, "description": d.description, "input_schema": d.input_schema} for d in active_defs],
                                default=str,
                                sort_keys=True,
                            )
                        prepared_messages, manifest = self._context_manager.assemble(
                            conversation,
                            system_prompt=self._options.system_prompt,
                            tool_definitions_text=tool_defs_text,
                        )
                    else:
                        prepared_messages = conversation
                        manifest = None
                    async with self._tracer.async_span("anycode.llm.chat", parent=turn_span) as llm_span:
                        response = await self._adapter.chat(prepared_messages, chat_params)
                        llm_span.set_attributes(
                            SpanAttributes(
                                model=self._options.model,
                                provider=self._adapter.name,
                                token_input=response.usage.input_tokens,
                                token_output=response.usage.output_tokens,
                            )
                        )
                    if manifest is not None:
                        manifest = ContextManager.reconcile(manifest, response.usage)
                        context_manifests.append(manifest)
                        if self._context_manager is not None:
                            # Provider-actual counts calibrate future pressure
                            # classification so compaction never fires late.
                            self._context_manager.note_actual(manifest)
                        if manifest.pressure in ("compact", "handoff"):
                            _persist(
                                "compaction",
                                {
                                    "pressure": manifest.pressure,
                                    "estimated_tokens": manifest.estimated_tokens,
                                    "archive_path": manifest.archive_path,
                                },
                            )

                        # Automatic context reset: at `handoff` pressure the
                        # context manager has archived a rebuildable artifact.
                        # Continue the same run (identity, budget, accounting)
                        # in a fresh, compact context instead of letting the
                        # conversation grow past the window.
                        if (
                            manifest.pressure == "handoff"
                            and manifest.handoff_path
                            and self._context_manager is not None
                            and self._context_manager.policy.auto_reset_on_handoff
                        ):
                            from anycode.core.context_manager import build_invariant_message, rebuild_from_handoff

                            conversation = rebuild_from_handoff(manifest.handoff_path)
                            conversation.append(
                                build_invariant_message(
                                    self._context_manager.policy,
                                    notice=(
                                        "Context was reset from a handoff artifact to stay within the "
                                        f"model window (archived at {manifest.handoff_path})."
                                    ),
                                )
                            )
                            seed_len = min(seed_len, len(conversation))
                            _persist(
                                "compaction",
                                {"reset": True, "handoff_path": manifest.handoff_path, "turn": turn_count},
                            )
                            _save_turn_checkpoint()

                    turn_span.set_attribute("llm_duration_ms", (time.monotonic() - llm_start) * MS_PER_SECOND)
                    cumulative_usage = merge_usage(cumulative_usage, response.usage)
                    self._budget.record_usage(response.usage)

                    response = await self._hook_runner.run_after_turn(response, ctx_info)

                    assistant_msg = LLMMessage(role="assistant", content=response.content)
                    conversation.append(assistant_msg)
                    if on_message:
                        on_message(assistant_msg)
                    _persist("message", _serialize_message(assistant_msg))

                    turn_text = _pull_text(response.content)
                    if turn_text:
                        yield StreamEvent(type="text", data=turn_text)

                    tool_blocks = _filter_tool_calls(response.content)

                    if self._output_schema:
                        structured_block = next((b for b in tool_blocks if b.name == STRUCTURED_OUTPUT_TOOL_NAME), None)
                        if structured_block:
                            raw_json = json.dumps(structured_block.input)
                            parsed = parse_structured_output(raw_json, self._output_schema)
                            if parsed is not None:
                                last_output = raw_json
                                turn_span.set_attribute("structured_output.valid", True)
                                tool_blocks = [b for b in tool_blocks if b.name != STRUCTURED_OUTPUT_TOOL_NAME]
                                if not tool_blocks:
                                    break
                            else:
                                turn_span.set_attribute("structured_output.valid", False)
                                tool_blocks = [b for b in tool_blocks if b.name != STRUCTURED_OUTPUT_TOOL_NAME]
                                structured_retries += 1
                                if structured_retries <= MAX_VALIDATION_RETRIES and turn_count < self._turn_limit:
                                    retry_msg = LLMMessage(
                                        role="user",
                                        content=[
                                            TextBlock(
                                                text=build_retry_prompt(
                                                    "",
                                                    "Response did not match the required schema. Return valid JSON matching the schema exactly.",
                                                )
                                            )
                                        ],
                                    )
                                    conversation.append(retry_msg)
                                    if on_message:
                                        on_message(retry_msg)
                                    if not tool_blocks:
                                        continue

                    for block in tool_blocks:
                        yield StreamEvent(type="tool_use", data=block)

                    if not tool_blocks:
                        last_output = turn_text or last_output

                        if self._validators and last_output:
                            validation = await run_validators(last_output, self._validators, ctx_info)
                            if not validation.valid and validation.retry:
                                validation_retries += 1
                                if validation_retries <= MAX_VALIDATION_RETRIES and turn_count < self._turn_limit:
                                    retry_msg = LLMMessage(
                                        role="user",
                                        content=[TextBlock(text=build_retry_prompt("", validation.reason or "Output validation failed."))],
                                    )
                                    conversation.append(retry_msg)
                                    if on_message:
                                        on_message(retry_msg)
                                    continue

                        if self._gate is not None:
                            emitter.transition(
                                "verifying",
                                metadata={"sensors": len(self._gate.sensors), "phase": "after_task"},
                            )
                            turn_span.set_attribute("phase", "verifying")
                            gate_ctx = SensorContext(
                                phase="after_task",
                                agent_name=agent_name,
                                run_id=emitter.events[0].run_id,
                                output=last_output,
                                messages=list(conversation),
                                tool_calls=list(tool_calls),
                                lifecycle_events=list(emitter.events),
                            )
                            decision = await self._gate.evaluate(gate_ctx)
                            gate_decisions.append(decision)
                            _persist("gate_decision", decision.model_dump(mode="json"))
                            verification_results.extend(decision.results)
                            for r in decision.results:
                                turn_span.add_event(
                                    f"sensor.{r.sensor_name}",
                                    {"passed": r.passed, "severity": r.severity, "kind": r.kind},
                                )
                            if decision.outcome == "block":
                                terminal_stop = StopReason(
                                    code="verification_failed",
                                    message=f"Quality gate blocked: {decision.message}",
                                    recoverable=False,
                                )
                                turn_span.set_attribute("stop_reason", terminal_stop.code)
                                turn_span.set_attribute("recoverable", terminal_stop.recoverable)
                                break
                            if decision.outcome == "escalate":
                                terminal_stop = StopReason(
                                    code="verification_failed",
                                    message=f"Quality gate escalated: {decision.message}",
                                    recoverable=True,
                                )
                                turn_span.set_attribute("stop_reason", terminal_stop.code)
                                turn_span.set_attribute("recoverable", terminal_stop.recoverable)
                                break
                            if decision.outcome == "retry" and verification_retries < max_verification_retries and turn_count < self._turn_limit:
                                verification_retries += 1
                                feedback_lines = [
                                    f"Verification '{r.sensor_name}' failed: {r.feedback_for_agent or r.message}"
                                    for r in decision.results
                                    if not r.passed and (r.feedback_for_agent or r.message)
                                ]
                                feedback = "\n".join(feedback_lines) or decision.message
                                retry_msg = LLMMessage(
                                    role="user",
                                    content=[TextBlock(text=build_retry_prompt("", feedback))],
                                )
                                conversation.append(retry_msg)
                                if on_message:
                                    on_message(retry_msg)
                                continue
                        terminal_stop = stop_success()
                        break

                    pre_decision = await self._evaluate_tool_gate(
                        phase="before_tool",
                        agent_name=agent_name,
                        run_id=emitter.events[0].run_id,
                        last_output=last_output,
                        conversation=conversation,
                        tool_calls=tool_calls,
                        emitter=emitter,
                        turn_span=turn_span,
                        verification_results=verification_results,
                        gate_decisions=gate_decisions,
                    )
                    if pre_decision is not None:
                        _persist("gate_decision", pre_decision.model_dump(mode="json"))
                    if pre_decision is not None and pre_decision.outcome in ("block", "escalate"):
                        recoverable = pre_decision.outcome == "escalate"
                        terminal_stop = StopReason(
                            code="verification_failed",
                            message=f"Quality gate {pre_decision.outcome} before tool use: {pre_decision.message}",
                            recoverable=recoverable,
                        )
                        turn_span.set_attribute("stop_reason", terminal_stop.code)
                        turn_span.set_attribute("recoverable", terminal_stop.recoverable)
                        break

                    ctx = self._build_context()
                    results = await self._execute_tool_blocks(tool_blocks, turn_span, ctx)

                    self._budget.record_tool_call(len(results))
                    for _rb, _record in results:
                        _persist(
                            "tool_result",
                            {
                                "tool_name": _record.tool_name,
                                "input": _record.input,
                                "output": _record.output[:16_000],
                                "duration": _record.duration,
                            },
                        )

                    post_decision = await self._evaluate_tool_gate(
                        phase="after_tool",
                        agent_name=agent_name,
                        run_id=emitter.events[0].run_id,
                        last_output=last_output,
                        conversation=conversation,
                        tool_calls=tool_calls + [r[1] for r in results],
                        emitter=emitter,
                        turn_span=turn_span,
                        verification_results=verification_results,
                        gate_decisions=gate_decisions,
                    )
                    if post_decision is not None:
                        _persist("gate_decision", post_decision.model_dump(mode="json"))
                    if post_decision is not None and post_decision.outcome in ("block", "escalate"):
                        recoverable = post_decision.outcome == "escalate"
                        terminal_stop = StopReason(
                            code="verification_failed",
                            message=f"Quality gate {post_decision.outcome} after tool use: {post_decision.message}",
                            recoverable=recoverable,
                        )
                        turn_span.set_attribute("stop_reason", terminal_stop.code)
                        turn_span.set_attribute("recoverable", terminal_stop.recoverable)
                        for _, record in results:
                            tool_calls.append(record)
                            yield StreamEvent(type="tool_result", data=record)
                        break

                    for block in tool_blocks:
                        loop_detector.record(fingerprint_call(block.name, block.input))
                    looping, pattern, repeats = loop_detector.is_looping()
                    if looping and pattern is not None:
                        terminal_stop = stop_doom_loop(pattern, repeats)
                        emitter.transition(
                            "recovering",
                            stop_reason=terminal_stop,
                            metadata={"pattern": pattern, "repeats": repeats},
                        )
                        for _, record in results:
                            tool_calls.append(record)
                            yield StreamEvent(type="tool_result", data=record)
                        break

                    handoff = self._detect_handoff(results)
                    if handoff is not None:
                        handoff_req, _ = handoff
                        # Preserve full audit trail: append every executed tool call (including
                        # any non-handoff calls in the same batch) and emit their tool_result
                        # events before terminating the run.
                        for _, record in results:
                            tool_calls.append(record)
                            yield StreamEvent(type="tool_result", data=record)
                        yield StreamEvent(type="handoff", data=handoff_req)
                        terminal_stop = stop_success("Handoff to downstream agent.")
                        final_event = emitter.transition("completed", stop_reason=terminal_stop)
                        if store is not None:
                            _save_turn_checkpoint()
                            _persist("stop", {"code": terminal_stop.code, "message": terminal_stop.message})
                            store.update_status(run_id, "completed")
                        yield StreamEvent(
                            type="done",
                            data=RunResult(
                                messages=conversation[seed_len:],
                                output=last_output or handoff_req.summary,
                                tool_calls=tool_calls,
                                token_usage=cumulative_usage,
                                turns=turn_count,
                                handoff_request=handoff_req,
                                terminal_phase=final_event.phase,
                                stop_reason=terminal_stop,
                                lifecycle_events=_all_lifecycle(),  # type: ignore[arg-type]
                                context_manifests=list(context_manifests),
                                verification_results=list(verification_results),
                                gate_decisions=list(gate_decisions),
                                retries=base_retries + validation_retries + structured_retries + verification_retries,
                            ),
                        )
                        return

                    emitter.transition("observing", metadata={"tool_calls": len(results)})

                    result_blocks: list[ContentBlock] = [r[0] for r in results]
                    for _, record in results:
                        tool_calls.append(record)
                        yield StreamEvent(type="tool_result", data=record)

                    tool_msg = LLMMessage(role="user", content=result_blocks)
                    conversation.append(tool_msg)
                    if on_message:
                        on_message(tool_msg)

                    if store is not None and durability is not None:
                        store.touch_heartbeat(run_id)
                        if turn_count % durability.checkpoint_every_turns == 0:
                            _save_turn_checkpoint()

            if terminal_stop is None:
                # Loop exited without an explicit reason: turn limit reached.
                terminal_stop = stop_max_turns(self._turn_limit)

        except asyncio.CancelledError:
            cancel_stop = StopReason(code="user_cancelled", message="Run cancelled by caller.", recoverable=False)
            try:
                emitter.transition("cancelled", stop_reason=cancel_stop)
            except Exception:  # noqa: BLE001
                pass
            if store is not None:
                try:
                    _save_turn_checkpoint()
                    _persist("stop", {"code": cancel_stop.code, "message": cancel_stop.message})
                    store.update_status(run_id, "cancelled")
                except Exception:  # noqa: BLE001 - persistence must not mask cancellation
                    pass
            yield StreamEvent(
                type="done",
                data=RunResult(
                    messages=conversation[seed_len:],
                    output=last_output or cancel_stop.message,
                    tool_calls=tool_calls,
                    token_usage=cumulative_usage,
                    turns=turn_count,
                    terminal_phase="cancelled",
                    stop_reason=cancel_stop,
                    lifecycle_events=_all_lifecycle(),  # type: ignore[arg-type]
                    context_manifests=list(context_manifests),
                    verification_results=list(verification_results),
                    gate_decisions=list(gate_decisions),
                    retries=base_retries + validation_retries + structured_retries + verification_retries,
                ),
            )
            raise
        except ProviderUnavailableError as e:
            # Transient-failure retries exhausted or circuit open: surface a
            # structured, recoverable stop reason instead of a raw error event.
            terminal_stop = stop_provider_unavailable(str(e))
        except Exception as e:
            failure_stop = StopReason(code="unknown", message=str(e), recoverable=False)
            try:
                emitter.transition("failed", stop_reason=failure_stop)
            except Exception:  # noqa: BLE001 - never mask the real error
                pass
            if store is not None:
                try:
                    _save_turn_checkpoint()
                    _persist("stop", {"code": failure_stop.code, "message": failure_stop.message})
                    store.update_status(run_id, "failed")
                except Exception:  # noqa: BLE001 - never mask the real error
                    pass
            yield StreamEvent(type="error", data=e)
            return

        if not last_output and conversation:
            for msg in reversed(conversation):
                if msg.role == "assistant":
                    last_output = _pull_text(msg.content)
                    break

        if terminal_stop is None:
            terminal_stop = stop_unknown()

        terminal_phase = "completed" if terminal_stop.code == "success" else "failed"
        try:
            final_event = emitter.transition(terminal_phase, stop_reason=terminal_stop)
            terminal_phase_name = final_event.phase
        except Exception:  # noqa: BLE001 - keep run result even if listener/transition fails
            terminal_phase_name = terminal_phase

        # Attach lifecycle attributes to the root tracer span for the run.
        async with self._tracer.async_span(f"anycode.agent.{agent_name}.terminal") as terminal_span:
            terminal_span.set_attributes(
                SpanAttributes(
                    agent_name=agent_name,
                    model=self._options.model,
                    phase=terminal_phase_name,
                    stop_reason=terminal_stop.code,
                    recoverable=terminal_stop.recoverable,
                )
            )

        if store is not None:
            _save_turn_checkpoint()
            _persist("stop", {"code": terminal_stop.code, "message": terminal_stop.message})
            if terminal_stop.code == "provider_unavailable":
                # Pause-not-die: the provider circuit is open. Park the run with
                # a timed wake so a scheduler sweep resumes it after the
                # provider has had time to recover.
                store.pause_run(
                    run_id,
                    WakeCondition(
                        kind="on_provider_recovery",
                        wake_at=datetime.now(UTC) + timedelta(seconds=120),
                        note=terminal_stop.message,
                    ),
                )
            else:
                store.update_status(run_id, "completed" if terminal_stop.code == "success" else "failed")

        yield StreamEvent(
            type="done",
            data=RunResult(
                messages=conversation[seed_len:],
                output=last_output,
                tool_calls=tool_calls,
                token_usage=cumulative_usage,
                turns=turn_count,
                terminal_phase=terminal_phase_name,
                stop_reason=terminal_stop,
                lifecycle_events=_all_lifecycle(),  # type: ignore[arg-type]
                context_manifests=list(context_manifests),
                verification_results=list(verification_results),
                gate_decisions=list(gate_decisions),
                retries=base_retries + validation_retries + structured_retries + verification_retries,
            ),
        )

    async def _evaluate_tool_gate(
        self,
        *,
        phase: str,
        agent_name: str,
        run_id: str,
        last_output: str,
        conversation: list[LLMMessage],
        tool_calls: list[ToolCallRecord],
        emitter: LifecycleEmitter,
        turn_span: Span,
        verification_results: list[VerificationResult],
        gate_decisions: list[QualityGateDecision],
    ) -> QualityGateDecision | None:
        """Run the configured QualityGate at a tool-boundary phase.

        Returns the decision (or None if no gate / no sensors registered for the phase).
        Caller decides whether to terminate the run on block/escalate outcomes.
        """
        if self._gate is None:
            return None
        from anycode.types import SensorPhase as _SP  # noqa: F401  (typing only)

        active = self._gate._sensors_for_phase(phase)  # type: ignore[arg-type]
        if not active:
            return None
        emitter.transition(
            "verifying",
            metadata={"sensors": len(active), "phase": phase},
        )
        turn_span.set_attribute("phase", "verifying")
        gate_ctx = SensorContext(
            phase=phase,  # type: ignore[arg-type]
            agent_name=agent_name,
            run_id=run_id,
            output=last_output,
            messages=list(conversation),
            tool_calls=list(tool_calls),
            lifecycle_events=list(emitter.events),
        )
        decision = await self._gate.evaluate(gate_ctx)
        gate_decisions.append(decision)
        verification_results.extend(decision.results)
        for r in decision.results:
            turn_span.add_event(
                f"sensor.{r.sensor_name}",
                {"passed": r.passed, "severity": r.severity, "kind": r.kind, "phase": phase},
            )
        return decision

    async def _execute_tool_blocks(
        self,
        blocks: list[ToolUseBlock],
        turn_span: Span,
        ctx: ToolUseContext,
    ) -> list[tuple[ToolResultBlock, ToolCallRecord]]:
        """Execute a batch of tool calls, respecting budget guardrails."""
        results: list[tuple[ToolResultBlock, ToolCallRecord]] = []
        for block in blocks:
            if self._budget.is_tool_blocked(block.name):
                result = ToolResult(data=f'Tool "{block.name}" is blocked by guardrail policy.', is_error=True)
                result_block = ToolResultBlock(tool_use_id=block.id, content=result.data, is_error=True)
                record = ToolCallRecord(tool_name=block.name, input=block.input, output=result.data, duration=0.0)
                results.append((result_block, record))
                continue

            began = time.monotonic()
            async with self._tracer.async_span(f"anycode.tool.{block.name}", parent=turn_span) as tool_span:
                try:
                    result = await self._executor.execute(block.name, block.input, ctx)
                except Exception as e:
                    result = ToolResult(data=str(e), is_error=True)
                    tool_span.set_error(str(e))

                duration = time.monotonic() - began
                tool_span.set_attributes(SpanAttributes(tool_name=block.name))
                tool_span.set_attribute("duration_ms", duration * MS_PER_SECOND)
                tool_span.set_attribute("is_error", bool(result.is_error))

            result_block = ToolResultBlock(
                tool_use_id=block.id,
                content=result.data,
                is_error=result.is_error,
            )
            record = ToolCallRecord(
                tool_name=block.name,
                input=block.input,
                output=result.data,
                duration=duration,
            )
            results.append((result_block, record))
        return results

    @staticmethod
    def _detect_handoff(results: list[tuple[ToolResultBlock, ToolCallRecord]]) -> tuple[HandoffRequest, ToolCallRecord] | None:
        """Scan tool results for a handoff sentinel. Returns (request, record) or None."""
        from anycode.handoff.tool import decode_handoff_payload

        for _, record in results:
            if record.tool_name != HANDOFF_TOOL_NAME:
                continue
            payload = decode_handoff_payload(record.output)
            if payload is None:
                continue
            return (
                HandoffRequest(
                    to_agent=payload["to_agent"],
                    summary=payload["summary"],
                    reason=payload["reason"],
                ),
                record,
            )
        return None

    def _build_context(self) -> ToolUseContext:
        return ToolUseContext(agent=self._build_agent_info())

    def _build_agent_info(self) -> AgentInfo:
        return AgentInfo(
            name=self._options.agent_name or "runner",
            role=self._options.agent_role or "assistant",
            model=self._options.model,
        )
