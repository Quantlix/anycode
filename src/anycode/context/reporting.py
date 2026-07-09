"""Render :class:`ContextUsageReport` instances for humans and machines.

Two surfaces:

* :func:`format_usage_report` — structured dict suitable for JSON manifests.
* :func:`render_usage_report_table` — Markdown-style table for CLI/log output.

The renderers are pure functions; they never mutate the report.
"""

from __future__ import annotations

from typing import Any

from anycode.types import ContextSectionUsage, ContextUsageReport

_HEADER_ROW = ("Section", "Tokens", "Share", "Status")
_LABELS: dict[str, str] = {
    "reserved_response": "Reserved for response",
    "system_instructions": "System Instructions",
    "tool_definitions": "Tool Definitions",
    "user_messages": "User Messages",
    "files": "Files",
    "tool_results": "Tool Results",
    "memory_rag": "Memory / RAG",
    "task_state": "Task State",
    "verification": "Verification",
    "offloaded_artifacts": "Offloaded Artifacts",
}


def format_usage_report(report: ContextUsageReport) -> dict[str, Any]:
    return {
        "max_context_tokens": report.max_context_tokens,
        "reserved_response_tokens": report.reserved_response_tokens,
        "used_tokens": report.used_tokens,
        "available_tokens": report.available_tokens,
        "counting_confidence": report.counting_confidence,
        "profile": report.profile.model_dump() if report.profile else None,
        "sections": [s.model_dump() for s in report.sections],
    }


def _format_share(section: ContextSectionUsage) -> str:
    return f"{section.percentage_of_window * 100:.1f}%"


def _row(section: ContextSectionUsage) -> tuple[str, str, str, str]:
    label = _LABELS.get(section.kind, section.kind)
    tokens = section.included_tokens or section.estimated_tokens
    status = section.strategy_applied or ("reserved" if section.kind == "reserved_response" else "included")
    return (label, f"{tokens:,}", _format_share(section), status)


def render_usage_report_table(report: ContextUsageReport) -> str:
    """Render a Markdown-compatible table for the report's sections."""
    rows: list[tuple[str, str, str, str]] = [_HEADER_ROW]
    rows.extend(_row(s) for s in report.sections)
    if not report.sections:
        rows.append(("(no sections recorded)", "0", "0.0%", ""))

    widths = [max(len(row[i]) for row in rows) for i in range(4)]

    def _format_row(row: tuple[str, str, str, str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[i] for i in range(4)) + " |"
    body = [_format_row(rows[0]), separator, *(_format_row(r) for r in rows[1:])]

    summary_lines = [
        f"Window: {report.max_context_tokens or 'unbounded'} tokens "
        f"(reserved {report.reserved_response_tokens:,} for response, "
        f"used {report.used_tokens:,}, available "
        f"{report.available_tokens if report.available_tokens is not None else 'unbounded'}).",
        f"Counting confidence: {report.counting_confidence}.",
    ]
    return "\n".join([*summary_lines, "", *body])
