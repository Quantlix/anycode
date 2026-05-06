"""Adaptive context lifecycle manager.

Responsibilities:
  * Estimate context pressure before a model call.
  * Trim, offload, compact, or hand off when pressure crosses configured ratios.
  * Return a structured ContextManifest describing the assembled prompt so
    downstream observability and reproducibility tools can audit each call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

from anycode.core.context_artifacts import offload_text, render_placeholder
from anycode.types import (
    ContextArtifact,
    ContextManifest,
    ContextPolicy,
    ContextPressure,
    ContextSource,
    LLMMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# Conservative tokens-per-character estimate when no model tokenizer is wired in.
_TOKENS_PER_CHAR: Final[float] = 0.25
# Avoid emitting empty placeholder messages when compaction collapses to zero kept history.
_COMPACTION_PROMPT: Final[str] = "[CONTEXT COMPACTED] Earlier dialogue was summarized below to conserve tokens."


def estimate_message_tokens(message: LLMMessage) -> int:
    total = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            total += int(len(block.text) * _TOKENS_PER_CHAR)
        elif isinstance(block, ToolUseBlock):
            total += int(len(json.dumps(block.input, default=str)) * _TOKENS_PER_CHAR) + 16
        elif isinstance(block, ToolResultBlock):
            total += int(len(block.content) * _TOKENS_PER_CHAR) + 8
        else:
            total += 16
    return total


def estimate_messages_tokens(messages: list[LLMMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _classify_pressure(ratio: float, policy: ContextPolicy) -> ContextPressure:
    if ratio >= policy.handoff_ratio:
        return "handoff"
    if ratio >= policy.compact_ratio:
        return "compact"
    if ratio >= policy.offload_ratio:
        return "offload"
    if ratio >= policy.trim_ratio:
        return "trim"
    return "normal"


class ContextManager:
    """Assembles model prompts under a ContextPolicy."""

    def __init__(self, policy: ContextPolicy) -> None:
        self._policy = policy
        self._artifacts: list[ContextArtifact] = []

    @property
    def policy(self) -> ContextPolicy:
        return self._policy

    @property
    def artifacts(self) -> list[ContextArtifact]:
        return list(self._artifacts)

    def _offload_oversized_blocks(
        self,
        message: LLMMessage,
        threshold_tokens: int,
    ) -> tuple[LLMMessage, list[ContextArtifact]]:
        """Offload any tool result block whose estimated tokens exceed `threshold_tokens`."""
        new_blocks: list = []
        offloaded: list[ContextArtifact] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                block_tokens = int(len(block.content) * _TOKENS_PER_CHAR)
                if block_tokens > threshold_tokens:
                    artifact = offload_text(
                        block.content,
                        self._policy.artifact_dir,
                        label="tool_result",
                    )
                    offloaded.append(artifact)
                    new_blocks.append(
                        ToolResultBlock(
                            tool_use_id=block.tool_use_id,
                            content=render_placeholder(artifact),
                            is_error=block.is_error,
                        )
                    )
                    continue
            new_blocks.append(block)
        return LLMMessage(role=message.role, content=new_blocks), offloaded

    def _compact_history(
        self,
        messages: list[LLMMessage],
    ) -> tuple[list[LLMMessage], str]:
        """Replace older messages with a single textual summary, keeping recent turns verbatim."""
        keep_n = max(self._policy.keep_recent_messages, 2)
        if len(messages) <= keep_n + 1:
            return list(messages), ""
        head = messages[:-keep_n]
        tail = messages[-keep_n:]
        bullets: list[str] = []
        for msg in head:
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    text_parts.append(f"<tool:{block.name}>")
                elif isinstance(block, ToolResultBlock):
                    text_parts.append("<tool_result>")
            joined = " ".join(t.strip() for t in text_parts if t.strip())
            if joined:
                bullets.append(f"- [{msg.role}] {joined[:240]}")
        summary = _COMPACTION_PROMPT + "\n" + "\n".join(bullets[: self._policy.summary_target_tokens // 8])
        compact_msg = LLMMessage(role="user", content=[TextBlock(text=summary)])
        return [compact_msg, *tail], summary

    def _write_handoff(self, messages: list[LLMMessage], pressure: ContextPressure) -> str:
        target_dir = Path(self._policy.artifact_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"handoff-{int(time.time() * 1000)}.json"
        payload = {
            "pressure": pressure,
            "policy": self._policy.model_dump(),
            "messages": [m.model_dump() for m in messages],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)

    def assemble(self, messages: list[LLMMessage]) -> tuple[list[LLMMessage], ContextManifest]:
        """Return possibly transformed messages plus a manifest describing the decision."""
        max_tokens = self._policy.max_context_tokens
        estimated = estimate_messages_tokens(messages)
        ratio = estimated / max_tokens if max_tokens > 0 else 0.0
        pressure = _classify_pressure(ratio, self._policy) if self._policy.enabled else "normal"

        sources = [
            ContextSource(
                kind="working_memory",
                label=f"messages[{len(messages)}]",
                estimated_tokens=estimated,
                preserved=True,
            )
        ]
        offloaded: list[ContextArtifact] = []
        compaction_summary: str | None = None
        handoff_path: str | None = None
        new_messages = list(messages)

        if pressure in ("offload", "compact", "handoff"):
            transformed: list[LLMMessage] = []
            for msg in new_messages:
                updated_msg, msg_offloads = self._offload_oversized_blocks(msg, self._policy.max_tool_output_tokens)
                if msg_offloads:
                    offloaded.extend(msg_offloads)
                    self._artifacts.extend(msg_offloads)
                    sources.append(
                        ContextSource(
                            kind="offloaded_artifact",
                            label=msg_offloads[0].artifact_id,
                            estimated_tokens=int(len(render_placeholder(msg_offloads[0])) * _TOKENS_PER_CHAR),
                        )
                    )
                transformed.append(updated_msg)
            new_messages = transformed

        if pressure in ("compact", "handoff"):
            new_messages, compaction_summary = self._compact_history(new_messages)
            if compaction_summary:
                sources.append(
                    ContextSource(
                        kind="task_state",
                        label="compaction_summary",
                        estimated_tokens=int(len(compaction_summary) * _TOKENS_PER_CHAR),
                    )
                )

        if pressure == "handoff":
            handoff_path = self._write_handoff(new_messages, pressure)
            sources.append(
                ContextSource(
                    kind="task_state",
                    label="handoff_file",
                    estimated_tokens=0,
                )
            )

        final_estimated = estimate_messages_tokens(new_messages)
        manifest = ContextManifest(
            pressure=pressure,
            estimated_tokens=final_estimated,
            max_tokens=max_tokens,
            sources=sources,
            offloaded=offloaded,
            compaction_summary=compaction_summary,
            handoff_path=handoff_path,
        )
        return new_messages, manifest


def rebuild_from_handoff(path: str | Path) -> list[LLMMessage]:
    """Restore a list of messages from a handoff artifact written by ContextManager."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_messages = payload.get("messages") or []
    return [LLMMessage.model_validate(item) for item in raw_messages]
