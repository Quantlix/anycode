"""Approval gate implementations — callback, stdin, and webhook channels."""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any, cast

from anycode.constants import POLL_INTERVAL_S, WEBHOOK_TIMEOUT_S
from anycode.hitl.review import format_approval_request
from anycode.types import ApprovalRequest, ApprovalResponse

ApprovalCallback = object  # placeholder for callable type


class CallbackApprovalGate:
    """Wraps a user-provided async callable as an ApprovalGate."""

    def __init__(self, handler: object) -> None:
        self._handler = handler

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return await self._handler(request)  # type: ignore[operator]


class StdinApprovalGate:
    """Interactive console approval gate — reads from stdin."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        if not sys.stdin.isatty():
            raise RuntimeError("StdinApprovalGate requires an interactive terminal (stdin is not a TTY).")

        display = format_approval_request(request)
        print(display, flush=True)  # noqa: T201

        choice = await asyncio.get_event_loop().run_in_executor(None, lambda: input("[a]pprove / [r]eject / [m]odify: ").strip().lower())

        if choice.startswith("a"):
            return ApprovalResponse(request_id=request.id, approved=True, responded_at=datetime.now(UTC))
        if choice.startswith("m"):
            raw = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Modified input (JSON): ").strip())
            try:
                modified = json.loads(raw)
            except json.JSONDecodeError:
                modified = None
            return ApprovalResponse(request_id=request.id, approved=True, modified_input=modified, responded_at=datetime.now(UTC))
        reason = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Reason (optional): ").strip())
        return ApprovalResponse(request_id=request.id, approved=False, reason=reason or None, responded_at=datetime.now(UTC))


class WebhookApprovalGate:
    """Posts approval requests to a webhook and polls for a response."""

    def __init__(
        self,
        request_url: str,
        poll_url: str,
        poll_interval: float = POLL_INTERVAL_S,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._request_url = request_url
        self._poll_url = poll_url
        self._poll_interval = poll_interval
        self._headers = headers or {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        payload = request.model_dump(mode="json")
        await asyncio.to_thread(self._post, self._request_url, payload)

        poll_target = self._poll_url.replace("{request_id}", request.id)
        while True:
            await asyncio.sleep(self._poll_interval)
            data = await asyncio.to_thread(self._get, poll_target)
            if data and data.get("responded_at"):
                return ApprovalResponse(
                    request_id=request.id,
                    approved=bool(data.get("approved", False)),
                    reason=str(data["reason"]) if data.get("reason") else None,
                    modified_input=cast(dict[str, Any], data["modified_input"]) if isinstance(data.get("modified_input"), dict) else None,
                    responded_at=datetime.fromisoformat(str(data.get("responded_at", datetime.now(UTC).isoformat()))),
                )

    def _post(self, url: str, payload: dict[str, object]) -> None:
        body = json.dumps(payload, default=str).encode()
        req = urllib.request.Request(url, data=body, headers={**self._headers, "Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S)  # noqa: S310
        except urllib.error.URLError as exc:
            raise ConnectionError(f"WebhookApprovalGate: POST to {url} failed: {exc}") from exc

    def _get(self, url: str) -> dict[str, object] | None:
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S) as resp:  # noqa: S310
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]
        except urllib.error.URLError:
            return None
