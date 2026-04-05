"""ApprovalManager — config-aware wrapper that enforces timeout and emits events."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from anycode.helpers.uuid7 import uuid7
from anycode.types import ApprovalConfig, ApprovalGate, ApprovalRequest, ApprovalResponse


class ApprovalManager:
    """Checks config to decide whether approval is needed, then delegates to a gate."""

    def __init__(self, config: ApprovalConfig, gate: ApprovalGate) -> None:
        self._config = config
        self._gate = gate
        self._history: list[tuple[ApprovalRequest, ApprovalResponse]] = []

    @property
    def history(self) -> list[tuple[ApprovalRequest, ApprovalResponse]]:
        return list(self._history)

    async def check_and_request(
        self,
        *,
        request_type: str,
        agent: str,
        description: str,
        context: dict[str, object] | None = None,
    ) -> ApprovalResponse | None:
        if not self._config.enabled:
            return None

        if request_type == "tool_call" and self._config.require_approval_tools:
            tool_name = (context or {}).get("tool_name", "")
            if tool_name not in self._config.require_approval_tools:
                return None

        if request_type == "task" and not self._config.require_approval_tasks:
            return None

        request = ApprovalRequest(
            id=str(uuid7()),
            type=request_type,  # type: ignore[arg-type]
            agent=agent,
            description=description,
            context=dict(context) if context else None,
            created_at=datetime.now(UTC),
        )

        try:
            response = await asyncio.wait_for(self._gate.request_approval(request), timeout=self._config.timeout_seconds)
        except TimeoutError:
            response = ApprovalResponse(
                request_id=request.id,
                approved=self._config.default_on_timeout == "approve",
                reason=f"Approval timed out after {self._config.timeout_seconds}s — default: {self._config.default_on_timeout}",
                responded_at=datetime.now(UTC),
            )

        self._history.append((request, response))
        return response
