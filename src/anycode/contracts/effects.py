"""Fenced operation claims for AnyCode-committed side-effect results."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, JsonValue

from anycode.contracts.models import ContractError, ContractModel, utc_now

OperationState = Literal["claimed", "committed", "released", "uncertain"]
ClaimOutcome = Literal["acquired", "committed", "replay", "busy", "conflict", "uncertain", "stale", "not_found", "invalid"]


class OperationClaim(ContractModel):
    operation_key: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    input_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    owner_id: str = Field(min_length=1)
    fencing_token: int = Field(ge=1)
    state: OperationState
    lease_expires_at: datetime
    result_artifact_id: str | None = None
    created_at: datetime
    updated_at: datetime


class OperationClaimResult(ContractModel):
    ok: bool
    outcome: ClaimOutcome
    claim: OperationClaim | None = None
    result_artifact_id: str | None = None
    error: ContractError | None = None


def canonical_input_digest(value: JsonValue) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class InMemoryOperationStore:
    """Deterministic reference claim store with leases and fencing tokens."""

    def __init__(self) -> None:
        self._claims: dict[str, OperationClaim] = {}
        self._lock = asyncio.Lock()

    async def claim(
        self,
        *,
        operation_key: str,
        run_id: str,
        task_id: str | None,
        input_digest: str,
        owner_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> OperationClaimResult:
        if lease_seconds <= 0:
            return OperationClaimResult(
                ok=False,
                outcome="invalid",
                error=ContractError(code="invalid_lease", message="A claim lease must be greater than zero seconds."),
            )
        timestamp = now or utc_now()
        async with self._lock:
            current = self._claims.get(operation_key)
            if current is not None and current.input_digest != input_digest:
                return OperationClaimResult(
                    ok=False,
                    outcome="conflict",
                    claim=current,
                    error=ContractError(code="operation_conflict", message="Operation key was reused with different input."),
                )
            if current is not None and current.state == "committed":
                return OperationClaimResult(
                    ok=True,
                    outcome="replay",
                    claim=current,
                    result_artifact_id=current.result_artifact_id,
                )
            if current is not None and current.state == "uncertain":
                return OperationClaimResult(
                    ok=False,
                    outcome="uncertain",
                    claim=current,
                    error=ContractError(
                        code="side_effect_unknown",
                        message="The prior owner may have completed the external effect; reconcile before retry.",
                    ),
                )
            if current is not None and current.state == "claimed" and current.lease_expires_at > timestamp:
                return OperationClaimResult(
                    ok=False,
                    outcome="busy",
                    claim=current,
                    error=ContractError(code="operation_busy", message="Operation is owned by an active lease.", retryable=True),
                )

            fencing_token = 1 if current is None else current.fencing_token + 1
            claim = OperationClaim(
                operation_key=operation_key,
                run_id=run_id,
                task_id=task_id,
                input_digest=input_digest,
                owner_id=owner_id,
                fencing_token=fencing_token,
                state="claimed",
                lease_expires_at=timestamp + timedelta(seconds=lease_seconds),
                created_at=current.created_at if current is not None else timestamp,
                updated_at=timestamp,
            )
            self._claims[operation_key] = claim
            return OperationClaimResult(ok=True, outcome="acquired", claim=claim)

    async def commit(
        self,
        *,
        operation_key: str,
        owner_id: str,
        fencing_token: int,
        result_artifact_id: str,
        now: datetime | None = None,
    ) -> OperationClaimResult:
        timestamp = now or utc_now()
        async with self._lock:
            current = self._claims.get(operation_key)
            if current is None:
                return OperationClaimResult(
                    ok=False,
                    outcome="not_found",
                    error=ContractError(code="claim_not_found", message="Operation claim does not exist."),
                )
            if current.state == "committed":
                if current.owner_id == owner_id and current.fencing_token == fencing_token and current.result_artifact_id == result_artifact_id:
                    return OperationClaimResult(
                        ok=True,
                        outcome="replay",
                        claim=current,
                        result_artifact_id=current.result_artifact_id,
                    )
                return self._stale(current)
            if current.owner_id != owner_id or current.fencing_token != fencing_token or current.lease_expires_at <= timestamp:
                return self._stale(current)
            if current.state != "claimed":
                return OperationClaimResult(
                    ok=False,
                    outcome="uncertain" if current.state == "uncertain" else "stale",
                    claim=current,
                    error=ContractError(code="claim_not_committable", message=f"A {current.state} claim cannot commit a result."),
                )

            committed = current.model_copy(update={"state": "committed", "result_artifact_id": result_artifact_id, "updated_at": timestamp})
            self._claims[operation_key] = committed
            return OperationClaimResult(
                ok=True,
                outcome="committed",
                claim=committed,
                result_artifact_id=result_artifact_id,
            )

    async def mark_uncertain(
        self,
        *,
        operation_key: str,
        owner_id: str,
        fencing_token: int,
        now: datetime | None = None,
    ) -> OperationClaimResult:
        timestamp = now or utc_now()
        async with self._lock:
            current = self._claims.get(operation_key)
            if current is None:
                return OperationClaimResult(
                    ok=False,
                    outcome="not_found",
                    error=ContractError(code="claim_not_found", message="Operation claim does not exist."),
                )
            if current.owner_id != owner_id or current.fencing_token != fencing_token or current.state != "claimed":
                return self._stale(current)
            uncertain = current.model_copy(update={"state": "uncertain", "updated_at": timestamp})
            self._claims[operation_key] = uncertain
            return OperationClaimResult(
                ok=False,
                outcome="uncertain",
                claim=uncertain,
                error=ContractError(code="side_effect_unknown", message="External effect outcome requires reconciliation."),
            )

    async def get(self, operation_key: str) -> OperationClaim | None:
        async with self._lock:
            return self._claims.get(operation_key)

    @staticmethod
    def _stale(current: OperationClaim) -> OperationClaimResult:
        return OperationClaimResult(
            ok=False,
            outcome="stale",
            claim=current,
            error=ContractError(code="stale_fencing_token", message="Claim owner or fencing token is no longer authoritative."),
        )
