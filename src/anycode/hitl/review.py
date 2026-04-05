"""Formatting utilities for rendering approval requests to console or JSON."""

from __future__ import annotations

from anycode.types import ApprovalRequest

_BOX_WIDTH = 55


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
            if len(display_v) > 80:
                display_v = display_v[:77] + "..."
            lines.append(f"{k}: {display_v}")
    lines.append("")
    lines.append("[a]pprove  [r]eject  [m]odify")

    top = "\u250c" + "\u2500" * _BOX_WIDTH + "\u2510"
    bottom = "\u2514" + "\u2500" * _BOX_WIDTH + "\u2518"
    body = "\n".join(f"\u2502 {line:<{_BOX_WIDTH - 1}}\u2502" for line in lines)
    return f"{top}\n{body}\n{bottom}"
