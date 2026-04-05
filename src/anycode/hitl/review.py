"""Formatting utilities for rendering approval requests to console or JSON."""

from __future__ import annotations

from anycode.constants import (
    APPROVAL_BOX_WIDTH,
    APPROVAL_CONTEXT_MAX_DISPLAY,
    APPROVAL_CONTEXT_TRUNCATE_AT,
    BOX_BOTTOM_LEFT,
    BOX_BOTTOM_RIGHT,
    BOX_HORIZONTAL,
    BOX_TOP_LEFT,
    BOX_TOP_RIGHT,
    BOX_VERTICAL,
)
from anycode.types import ApprovalRequest


def format_approval_request(request: ApprovalRequest) -> str:
    lines = [
        "APPROVAL REQUIRED",
        f"Agent: {request.agent}",
        f"Type: {request.type}",
        f"Action: {request.description}",
    ]
    if request.context:
        for k, v in request.context.items():
            display_v = str(v)
            if len(display_v) > APPROVAL_CONTEXT_MAX_DISPLAY:
                display_v = display_v[:APPROVAL_CONTEXT_TRUNCATE_AT] + "..."
            lines.append(f"{k}: {display_v}")
    lines.append("")
    lines.append("[a]pprove  [r]eject  [m]odify")

    top = BOX_TOP_LEFT + BOX_HORIZONTAL * APPROVAL_BOX_WIDTH + BOX_TOP_RIGHT
    bottom = BOX_BOTTOM_LEFT + BOX_HORIZONTAL * APPROVAL_BOX_WIDTH + BOX_BOTTOM_RIGHT
    body = "\n".join(f"{BOX_VERTICAL} {line:<{APPROVAL_BOX_WIDTH - 1}}{BOX_VERTICAL}" for line in lines)
    return f"{top}\n{body}\n{bottom}"
