"""External policy decision, obligation, and audit enforcement."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, JsonValue

from anycode.contracts.models import ContractError, ContractModel, PolicyDecision, PolicyObligation, utc_now
from anycode.helpers.uuid7 import uuid7
from anycode.identity.context import BoundaryKind, ExecutionContext


class PolicyRequest(ContractModel):
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    action: str = Field(min_length=1)
    resource: str = Field(min_length=1)
    boundary: BoundaryKind
    context: ExecutionContext
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)


class PolicyAuditEvent(ContractModel):
    decision_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    action: str = Field(min_length=1)
    resource: str = Field(min_length=1)
    boundary: BoundaryKind
    outcome: str = Field(pattern=r"^(allow|deny)$")
    policy_version: str = Field(min_length=1)
    reason_codes: tuple[str, ...] = ()
    obligation_types: tuple[str, ...] = ()
    context: dict[str, str] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)


class PolicyEnforcementResult(ContractModel):
    allowed: bool
    decision: PolicyDecision
    applied_obligations: tuple[str, ...] = ()
    error: ContractError | None = None


@runtime_checkable
class ExternalPolicyAdapter(Protocol):
    async def decide(self, request: PolicyRequest) -> PolicyDecision: ...


@runtime_checkable
class PolicyAuditSink(Protocol):
    async def record(self, event: PolicyAuditEvent) -> None: ...


class InMemoryPolicyAuditSink:
    def __init__(self) -> None:
        self._events: list[PolicyAuditEvent] = []

    @property
    def events(self) -> tuple[PolicyAuditEvent, ...]:
        return tuple(self._events)

    async def record(self, event: PolicyAuditEvent) -> None:
        self._events.append(event)


ObligationHandler = Callable[[PolicyObligation, PolicyRequest], Awaitable[bool]]


class PolicyEnforcer:
    """Evaluates external policy and fails closed when configured to require it."""

    def __init__(
        self,
        adapter: ExternalPolicyAdapter | None = None,
        *,
        fail_closed: bool = False,
        audit_sink: PolicyAuditSink | None = None,
        obligation_handlers: dict[str, ObligationHandler] | None = None,
    ) -> None:
        self._adapter = adapter
        self._fail_closed = fail_closed
        self._audit_sink = audit_sink
        self._obligation_handlers = dict(obligation_handlers or {})

    def _local_decision(self, request: PolicyRequest, *, outcome: Literal["allow", "deny"], reason: str) -> PolicyDecision:
        return PolicyDecision(
            id=str(uuid7()),
            run_id=request.run_id,
            task_id=request.task_id,
            outcome=outcome,
            policy_version="anycode-local/1",
            reason_codes=(reason,),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            generation=request.generation,
            attempt=request.attempt,
        )

    async def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        if self._adapter is None:
            decision = self._local_decision(
                request,
                outcome="deny" if self._fail_closed else "allow",
                reason="policy_required" if self._fail_closed else "policy_not_configured",
            )
        else:
            try:
                decision = await self._adapter.decide(request)
            except Exception:
                decision = self._local_decision(
                    request,
                    outcome="deny" if self._fail_closed else "allow",
                    reason="policy_unavailable",
                )
            else:
                if (
                    decision.run_id != request.run_id
                    or decision.task_id != request.task_id
                    or decision.correlation_id != request.correlation_id
                    or decision.generation != request.generation
                    or decision.attempt != request.attempt
                ):
                    decision = self._local_decision(request, outcome="deny", reason="invalid_policy_decision_context")
        await self._audit(decision, request)
        return decision

    async def enforce(self, request: PolicyRequest) -> PolicyEnforcementResult:
        decision = await self.evaluate(request)
        if decision.outcome == "deny":
            return PolicyEnforcementResult(
                allowed=False,
                decision=decision,
                error=ContractError(code="policy_denied", message="The external policy decision denied this operation."),
            )
        applied: list[str] = []
        for obligation in decision.obligations:
            handler = self._obligation_handlers.get(obligation.type)
            fulfilled = False
            if handler is not None:
                try:
                    fulfilled = await handler(obligation, request)
                except Exception:
                    fulfilled = False
            if not fulfilled:
                return PolicyEnforcementResult(
                    allowed=False,
                    decision=decision,
                    applied_obligations=tuple(applied),
                    error=ContractError(
                        code="policy_obligation_unfulfilled",
                        message=f"Required policy obligation could not be fulfilled: {obligation.type}",
                    ),
                )
            applied.append(obligation.type)
        return PolicyEnforcementResult(allowed=True, decision=decision, applied_obligations=tuple(applied))

    async def _audit(self, decision: PolicyDecision, request: PolicyRequest) -> None:
        if self._audit_sink is None:
            return
        event = PolicyAuditEvent(
            decision_id=decision.id,
            run_id=decision.run_id,
            task_id=decision.task_id,
            action=request.action,
            resource=request.resource,
            boundary=request.boundary,
            outcome=decision.outcome,
            policy_version=decision.policy_version,
            reason_codes=decision.reason_codes,
            obligation_types=tuple(obligation.type for obligation in decision.obligations),
            context=request.context.audit_attributes(),
        )
        try:
            await self._audit_sink.record(event)
        except Exception:
            return
