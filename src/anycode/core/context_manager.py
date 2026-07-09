"""Adaptive context lifecycle manager.

Responsibilities
----------------
* Resolve an effective :class:`ModelContextProfile` for the (provider, model)
  pair the runner is about to call.
* Reserve response tokens before assembling input sections.
* Classify content into typed :class:`ContextSectionKind` sections, track
  estimated tokens, and apply per-section overflow policies (`trim`,
  `summarize`, `offload`, `drop`, `error`).
* Apply pressure-based macro strategies (`trim` -> `offload` -> `compact`
  -> `handoff`) so long-running runs stay within the resolved window.
* Produce a structured :class:`ContextManifest` and embedded
  :class:`ContextUsageReport` for every model call.
* Reconcile provider-actual usage back into the manifest after the call.

Auto mode (`ContextPolicy.mode == "auto"`) removes the AnyCode-imposed
ceiling and uses the resolved profile's max as the effective window. When
no profile bounds the window the engine reports `max_tokens=0` (treated as
unbounded) and never trims unless a section budget is explicit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

from anycode.context.profiles import resolve_profile
from anycode.context.tokenizer import Tokenizer, select_tokenizer
from anycode.core.context_artifacts import offload_text, render_placeholder
from anycode.types import (
    ContentBlock,
    ContextArtifact,
    ContextManifest,
    ContextPolicy,
    ContextPressure,
    ContextSectionBudget,
    ContextSectionInput,
    ContextSectionKind,
    ContextSectionUsage,
    ContextSource,
    ContextSourceKind,
    ContextUsageReport,
    LLMMessage,
    ModelContextProfile,
    SectionOverflow,
    SectionPriority,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)

# Conservative tokens-per-character estimate when no model tokenizer is wired in.
_TOKENS_PER_CHAR: Final[float] = 0.25
_COMPACTION_PROMPT: Final[str] = "[CONTEXT COMPACTED] Earlier dialogue was summarized below to conserve tokens."
_TRIM_MARKER: Final[str] = "\n…[TRIMMED CONTENT]…\n"
_INVARIANT_MARKER: Final[str] = "[context-invariants]"
# Tool results shorter than this are never masked — the pointer would not be
# meaningfully smaller than the content.
_MASK_MIN_CHARS: Final[int] = 240

# Default proportional allocation for sections when the policy doesn't define
# explicit budgets. Values must sum to ~1.0; remaining slack is left as the
# flexible tail so sections can borrow when one comes in under budget.
_DEFAULT_ALLOCATION: dict[ContextSectionKind, float] = {
    "system_instructions": 0.10,
    "tool_definitions": 0.05,
    "user_messages": 0.25,
    "tool_results": 0.30,
    "memory_rag": 0.08,
    "task_state": 0.05,
    "files": 0.10,
    "verification": 0.02,
    "offloaded_artifacts": 0.02,
}

_DEFAULT_PRIORITY: dict[ContextSectionKind, SectionPriority] = {
    "reserved_response": "required",
    "system_instructions": "required",
    "task_state": "required",
    "verification": "high",
    "user_messages": "high",
    "tool_definitions": "high",
    "tool_results": "medium",
    "memory_rag": "medium",
    "files": "medium",
    "offloaded_artifacts": "low",
}

_DEFAULT_OVERFLOW: dict[ContextSectionKind, SectionOverflow] = {
    "reserved_response": "error",
    "system_instructions": "error",
    "task_state": "trim",
    "verification": "trim",
    "user_messages": "trim",
    "tool_definitions": "trim",
    "tool_results": "summarize",
    "memory_rag": "trim",
    "files": "offload",
    "offloaded_artifacts": "drop",
}

_SECTION_TITLES: dict[ContextSectionKind, str] = {
    "files": "Files and Code Excerpts",
    "memory_rag": "Memory and RAG Context",
    "task_state": "Task and Checkpoint State",
    "verification": "Verification State",
    "offloaded_artifacts": "Offloaded Artifacts",
}

_SECTION_SOURCE_KIND: dict[ContextSectionKind, str] = {
    "files": "files",
    "memory_rag": "external_memory",
    "task_state": "task_state",
    "verification": "verification",
    "offloaded_artifacts": "offloaded_artifact",
}


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
    if ratio >= policy.mask_ratio:
        return "mask"
    if ratio >= policy.trim_ratio:
        return "trim"
    return "normal"


def _classify_block(role: str, block: ContentBlock) -> ContextSectionKind:
    """Map a content block to its primary :class:`ContextSectionKind`."""
    if isinstance(block, ToolResultBlock):
        return "tool_results"
    if isinstance(block, ToolUseBlock):
        return "user_messages" if role == "assistant" else "tool_results"
    return "user_messages"


def _render_context_input(item: ContextSectionInput) -> str:
    title = _SECTION_TITLES.get(item.kind, item.kind.replace("_", " ").title())
    label = item.label.strip() or item.kind
    content = item.content.strip()
    if not content:
        return ""
    return f"## {title}: {label}\n{content}"


def _format_mapping(mapping: Mapping[str, object]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in mapping.items())


def _format_sequence(values: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _truncate_text(text: str, target_tokens: int, tokenizer: Tokenizer) -> str:
    """Trim text to roughly `target_tokens` while preserving head and tail."""
    if target_tokens <= 0 or not text:
        return ""
    current = tokenizer.count_text(text)
    if current <= target_tokens:
        return text
    keep_chars = max(64, int(len(text) * (target_tokens / max(current, 1))))
    if keep_chars >= len(text):
        return text
    half = keep_chars // 2
    return text[:half] + _TRIM_MARKER + text[-half:]


class ContextManager:
    """Assembles model prompts under a :class:`ContextPolicy`.

    The manager is created per-run and is safe to reuse across turns of the
    same agent. It is intentionally synchronous so prompt assembly does not
    introduce additional async hops on hot paths.
    """

    def __init__(
        self,
        policy: ContextPolicy,
        *,
        provider: str | None = None,
        model: str | None = None,
        summarizer: Callable[[list[LLMMessage]], str] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_policy = policy
        self._policy = policy.for_provider(provider)
        self._artifacts: list[ContextArtifact] = []
        self._profile, self._profile_warning = resolve_profile(
            provider=provider,
            model=model or "",
            override=self._policy.model_profile,
            custom_profiles=self._policy.custom_profiles,
        )
        self._tokenizer: Tokenizer = select_tokenizer(self._profile.tokenizer_strategy, model=model)
        self._effective_window = self._resolve_effective_window()
        # Optional richer compaction (e.g. an LLM call). On failure the
        # deterministic extractive path is the fallback — degrade, never corrupt.
        self._summarizer = summarizer
        # Running correction factor from provider-reported actual token counts:
        # local estimates run systematically low, which fires compaction late.
        self._calibration = 1.0

    # -- public introspection ------------------------------------------------

    @property
    def policy(self) -> ContextPolicy:
        return self._policy

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def profile(self) -> ModelContextProfile:
        return self._profile

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def effective_window(self) -> int | None:
        """Return the resolved context window in tokens (None means unbounded)."""
        return self._effective_window

    def section_budget(self, kind: ContextSectionKind) -> ContextSectionBudget:
        """Return the effective budget for a context section."""
        return self._section_budget(kind)

    @property
    def artifacts(self) -> list[ContextArtifact]:
        return list(self._artifacts)

    # -- resolution helpers --------------------------------------------------

    def _resolve_effective_window(self) -> int | None:
        # Auto mode: profile is authoritative; None => unbounded.
        if self._policy.mode == "auto":
            return self._profile.max_context_tokens
        # Manual mode (default): policy max wins, but a stricter profile cap clips it.
        explicit = self._policy.max_context_tokens
        profile_cap = self._profile.max_context_tokens
        if profile_cap is None:
            return explicit
        return min(explicit, profile_cap)

    def _section_budget(self, kind: ContextSectionKind) -> ContextSectionBudget:
        """Resolve the budget for `kind`, falling back to defaults derived from window."""
        if kind in self._policy.sections:
            return self._policy.sections[kind]
        max_tokens: int | None = None
        if self._effective_window is not None and self._effective_window > 0:
            available = max(self._effective_window - self._policy.reserved_response_tokens, 0)
            share = _DEFAULT_ALLOCATION.get(kind, 0.0)
            max_tokens = int(available * share) if share > 0 else None
        return ContextSectionBudget(
            kind=kind,
            max_tokens=max_tokens,
            priority=_DEFAULT_PRIORITY.get(kind, "medium"),
            overflow=_DEFAULT_OVERFLOW.get(kind, "trim"),
        )

    # -- section accounting --------------------------------------------------

    def _section_usage(
        self,
        kind: ContextSectionKind,
        estimated: int,
        included: int,
        strategy: str | None = None,
    ) -> ContextSectionUsage:
        window = self._effective_window or 0
        share = (included / window) if window > 0 else 0.0
        return ContextSectionUsage(
            kind=kind,
            estimated_tokens=estimated,
            included_tokens=included,
            percentage_of_window=share,
            strategy_applied=strategy,
        )

    def _static_text_section_usage(self, kind: ContextSectionKind, text: str, warnings: list[str]) -> ContextSectionUsage:
        tokens = self._tokenizer.count_text(text)
        budget = self._section_budget(kind)
        strategy: str | None = None
        if budget.max_tokens is not None and tokens > budget.max_tokens:
            if budget.overflow == "error":
                raise ValueError(f"Section '{kind}' is over budget ({tokens} > {budget.max_tokens}) and overflow is 'error'.")
            strategy = "over_budget"
            warnings.append(
                f"Section '{kind}' exceeds its budget ({tokens} > {budget.max_tokens}); provider boundary content was reported but not transformed."
            )
        return self._section_usage(kind, tokens, tokens, strategy)

    def _classify_messages(self, messages: list[LLMMessage]) -> dict[ContextSectionKind, int]:
        bins: dict[ContextSectionKind, int] = {}
        for msg in messages:
            block_total = 0
            for block in msg.content:
                kind = _classify_block(msg.role, block)
                block_tokens = self._tokenizer.count_block(block)
                block_total += block_tokens
                bins[kind] = bins.get(kind, 0) + block_tokens
            overhead = max(self._tokenizer.count_messages([msg]) - block_total, 0)
            if overhead:
                bins["user_messages"] = bins.get("user_messages", 0) + overhead
        return bins

    # -- section trim --------------------------------------------------------

    def _apply_section_trim(
        self,
        messages: list[LLMMessage],
        kind: ContextSectionKind,
        budget: ContextSectionBudget,
    ) -> tuple[list[LLMMessage], int, str | None]:
        if budget.max_tokens is None:
            section_total = sum(self._tokenizer.count_block(b) for m in messages for b in m.content if _classify_block(m.role, b) == kind)
            return messages, section_total, None

        section_total = sum(self._tokenizer.count_block(b) for m in messages for b in m.content if _classify_block(m.role, b) == kind)

        if section_total <= budget.max_tokens:
            return messages, section_total, None

        if budget.overflow == "error":
            raise ValueError(f"Section '{kind}' is over budget ({section_total} > {budget.max_tokens}) and overflow is 'error'.")
        if budget.overflow == "drop":
            return self._drop_section(messages, kind), 0, "dropped"
        # trim / summarize / offload (offload of files handled elsewhere) -> truncate.
        return self._trim_section(messages, kind, budget.max_tokens), budget.max_tokens, budget.overflow

    def _drop_section(self, messages: list[LLMMessage], kind: ContextSectionKind) -> list[LLMMessage]:
        new_messages: list[LLMMessage] = []
        for msg in messages:
            kept = [b for b in msg.content if _classify_block(msg.role, b) != kind]
            if kept:
                new_messages.append(LLMMessage(role=msg.role, content=kept))
        return new_messages

    def _trim_section(
        self,
        messages: list[LLMMessage],
        kind: ContextSectionKind,
        budget: int,
    ) -> list[LLMMessage]:
        # Walk newest-first: keep recent content verbatim, truncate older ones.
        block_index: list[tuple[int, int, ContentBlock]] = []
        for mi, msg in enumerate(messages):
            for bi, block in enumerate(msg.content):
                if _classify_block(msg.role, block) == kind:
                    block_index.append((mi, bi, block))

        remaining = budget
        kept: dict[tuple[int, int], ContentBlock] = {}
        for mi, bi, block in reversed(block_index):
            tokens = self._tokenizer.count_block(block)
            if remaining <= 0:
                kept[(mi, bi)] = self._stub_block(block)
                continue
            if tokens <= remaining:
                kept[(mi, bi)] = block
                remaining -= tokens
                continue
            kept[(mi, bi)] = self._truncate_block(block, remaining)
            remaining = 0

        new_messages: list[LLMMessage] = []
        for mi, msg in enumerate(messages):
            new_blocks: list[ContentBlock] = []
            for bi, block in enumerate(msg.content):
                if _classify_block(msg.role, block) == kind:
                    new_blocks.append(kept.get((mi, bi), self._stub_block(block)))
                else:
                    new_blocks.append(block)
            new_messages.append(LLMMessage(role=msg.role, content=new_blocks))
        return new_messages

    def _truncate_block(self, block: ContentBlock, budget: int) -> ContentBlock:
        if isinstance(block, TextBlock):
            return TextBlock(text=_truncate_text(block.text, budget, self._tokenizer))
        if isinstance(block, ToolResultBlock):
            return ToolResultBlock(
                tool_use_id=block.tool_use_id,
                content=_truncate_text(block.content, budget, self._tokenizer),
                is_error=block.is_error,
            )
        return block

    def _stub_block(self, block: ContentBlock) -> ContentBlock:
        if isinstance(block, TextBlock):
            return TextBlock(text=_TRIM_MARKER.strip())
        if isinstance(block, ToolResultBlock):
            return ToolResultBlock(
                tool_use_id=block.tool_use_id,
                content=_TRIM_MARKER.strip(),
                is_error=block.is_error,
            )
        return block

    # -- first-class non-message sections -----------------------------------

    def _collect_context_section_inputs(
        self,
        *,
        context_sections: list[ContextSectionInput] | tuple[ContextSectionInput, ...] | None = None,
        file_contexts: Mapping[str, str] | None = None,
        memory_context: str | None = None,
        task_state: Mapping[str, object] | None = None,
        checkpoint_state: Mapping[str, object] | None = None,
        verification_failures: list[str] | tuple[str, ...] | None = None,
        artifact_handles: list[ContextArtifact] | tuple[ContextArtifact, ...] | None = None,
    ) -> list[ContextSectionInput]:
        inputs = list(context_sections or ())

        for label, content in (file_contexts or {}).items():
            inputs.append(ContextSectionInput(kind="files", label=label, content=content))

        if memory_context:
            inputs.append(ContextSectionInput(kind="memory_rag", label="retrieved_context", content=memory_context))

        combined_task_state: dict[str, object] = dict(self._policy.preserved_task_state)
        if task_state:
            combined_task_state.update(task_state)
        if combined_task_state:
            inputs.append(ContextSectionInput(kind="task_state", label="task_state", content=_format_mapping(combined_task_state)))

        if checkpoint_state:
            inputs.append(ContextSectionInput(kind="task_state", label="checkpoint_state", content=_format_mapping(checkpoint_state)))

        combined_failures = [*self._policy.preserved_verification_failures, *(verification_failures or ())]
        if combined_failures:
            inputs.append(ContextSectionInput(kind="verification", label="open_failures", content=_format_sequence(combined_failures)))

        if artifact_handles:
            artifact_text = "\n".join(render_placeholder(artifact) for artifact in artifact_handles)
            inputs.append(ContextSectionInput(kind="offloaded_artifacts", label="artifact_handles", content=artifact_text))

        return inputs

    def _prepare_context_section_messages(
        self,
        inputs: list[ContextSectionInput],
    ) -> tuple[list[LLMMessage], list[ContextSectionUsage], list[ContextSource], list[ContextArtifact]]:
        if not inputs:
            return [], [], [], []

        grouped: dict[ContextSectionKind, list[str]] = {}
        labels: dict[ContextSectionKind, list[str]] = {}
        preserved: dict[ContextSectionKind, bool] = {}
        for item in inputs:
            if item.kind in ("reserved_response", "system_instructions", "tool_definitions", "user_messages", "tool_results"):
                continue
            rendered = _render_context_input(item)
            if not rendered:
                continue
            grouped.setdefault(item.kind, []).append(rendered)
            labels.setdefault(item.kind, []).append(item.label)
            preserved[item.kind] = preserved.get(item.kind, False) or item.preserved

        messages: list[LLMMessage] = []
        usages: list[ContextSectionUsage] = []
        sources: list[ContextSource] = []
        offloaded: list[ContextArtifact] = []

        for kind, payloads in grouped.items():
            rendered = "\n\n".join(payloads)
            original_message = LLMMessage(role="user", content=[TextBlock(text=rendered)])
            estimated = self._tokenizer.count_messages([original_message])
            budget = self._section_budget(kind)
            included_text = rendered
            strategy: str | None = None

            if budget.max_tokens is not None and estimated > budget.max_tokens:
                if budget.overflow == "error":
                    raise ValueError(f"Section '{kind}' is over budget ({estimated} > {budget.max_tokens}) and overflow is 'error'.")
                if budget.overflow == "drop":
                    included_text = ""
                    strategy = "dropped"
                elif budget.overflow == "offload":
                    artifact = offload_text(rendered, self._policy.artifact_dir, label=f"context_{kind}")
                    offloaded.append(artifact)
                    self._artifacts.append(artifact)
                    included_text = render_placeholder(artifact)
                    strategy = "offload"
                else:
                    included_text = _truncate_text(rendered, budget.max_tokens, self._tokenizer)
                    strategy = budget.overflow

            included = 0
            if included_text:
                included_message = LLMMessage(role="user", content=[TextBlock(text=included_text)])
                included = self._tokenizer.count_messages([included_message])
                messages.append(included_message)

            usages.append(self._section_usage(kind, estimated, included, strategy))
            source_kind = cast(ContextSourceKind, _SECTION_SOURCE_KIND.get(kind, "external_memory"))
            sources.append(
                ContextSource(
                    kind=source_kind,
                    label=", ".join(labels.get(kind, ())) or kind,
                    estimated_tokens=estimated,
                    preserved=preserved.get(kind, False) or budget.priority in ("required", "high"),
                )
            )

        return messages, usages, sources, offloaded

    # -- macro pressure stages ----------------------------------------------

    def _offload_oversized_blocks(
        self,
        message: LLMMessage,
        threshold_tokens: int,
    ) -> tuple[LLMMessage, list[ContextArtifact]]:
        new_blocks: list[ContentBlock] = []
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

    def _mask_aged_tool_results(self, messages: list[LLMMessage]) -> tuple[list[LLMMessage], int]:
        """Replace tool results outside the protected recency window with short
        restorable pointers. Deterministic and cheap — this stage runs before
        offloading or any summarization, and the runner's own conversation
        retains the full content, so masking here is never destructive.
        """
        keep_n = max(self._policy.keep_recent_messages, 2)
        if len(messages) <= keep_n:
            return list(messages), 0
        masked_count = 0
        transformed: list[LLMMessage] = []
        for msg in messages[:-keep_n]:
            new_blocks: list[ContentBlock] = []
            changed = False
            for block in msg.content:
                if isinstance(block, ToolResultBlock) and len(block.content) > _MASK_MIN_CHARS:
                    pointer = f"[masked tool result: {len(block.content)} chars — superseded by later turns]"
                    new_blocks.append(ToolResultBlock(tool_use_id=block.tool_use_id, content=pointer, is_error=block.is_error))
                    masked_count += 1
                    changed = True
                else:
                    new_blocks.append(block)
            transformed.append(LLMMessage(role=msg.role, content=new_blocks) if changed else msg)
        transformed.extend(messages[-keep_n:])
        return transformed, masked_count

    def _archive_history(self, messages: list[LLMMessage]) -> str:
        """Archive the full pre-compaction history — always, before any
        summarization, so nothing a summary drops is ever unrecoverable."""
        target_dir = Path(self._policy.artifact_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"history-{int(time.time() * 1000)}.json"
        payload = {"messages": [m.model_dump() for m in messages]}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)

    def _compact_history(
        self,
        messages: list[LLMMessage],
        *,
        archive_path: str | None = None,
    ) -> tuple[list[LLMMessage], str]:
        keep_n = max(self._policy.keep_recent_messages, 2)
        if len(messages) <= keep_n + 1:
            return list(messages), ""
        head = messages[:-keep_n]
        tail = messages[-keep_n:]

        body = self._summarize_head(head)

        preserved_lines: list[str] = []
        if self._policy.preserved_task_state:
            preserved_lines.append("PRESERVED TASK STATE:")
            for key, value in self._policy.preserved_task_state.items():
                preserved_lines.append(f"  {key}: {value}")
        if self._policy.preserved_verification_failures:
            preserved_lines.append("OPEN VERIFICATION FAILURES:")
            for failure in self._policy.preserved_verification_failures:
                preserved_lines.append(f"  - {failure}")

        recovery_lines: list[str] = []
        if archive_path:
            recovery_lines.append(f"FULL HISTORY ARCHIVED AT: {archive_path}")
        if self._artifacts:
            recovery_lines.append("ARTIFACT INDEX (offloaded content, recoverable by path):")
            recovery_lines.extend(f"  - {a.artifact_id}: {a.path}" for a in self._artifacts[-20:])

        summary = _COMPACTION_PROMPT + "\n" + "\n".join([*preserved_lines, *recovery_lines, body])
        compact_msg = LLMMessage(role="user", content=[TextBlock(text=summary)])
        return [compact_msg, *tail], summary

    def _summarize_head(self, head: list[LLMMessage]) -> str:
        """Summarize compacted-away history: the configured summarizer when one
        is wired in (e.g. an LLM call), with the deterministic extractive path
        as both the default and the failure fallback."""
        if self._summarizer is not None:
            try:
                text = self._summarizer(head)
                if text and text.strip():
                    return text.strip()
            except Exception:  # noqa: BLE001 - degrade to deterministic, never corrupt
                pass
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
        return "\n".join(bullets[: self._policy.summary_target_tokens // 8])

    @staticmethod
    def _is_compaction_message(message: LLMMessage) -> bool:
        return any(isinstance(block, TextBlock) and _COMPACTION_PROMPT in block.text for block in message.content)

    def _write_handoff(self, messages: list[LLMMessage], pressure: ContextPressure) -> str:
        """Write a five-layer handoff artifact: typed state, narrative, decisions,
        priority queue, warnings — plus the compacted message set for rebuild.

        The state layer is machine-written and machine-read; the other layers
        orient a fresh context after a reset. Layers not derivable
        deterministically stay empty rather than being fabricated.
        """
        target_dir = Path(self._policy.artifact_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"handoff-{int(time.time() * 1000)}.json"
        payload = {
            "version": 2,
            "pressure": pressure,
            "policy": self._policy.model_dump(),
            "state": {
                "task_state": dict(self._policy.preserved_task_state),
                "verification_failures": list(self._policy.preserved_verification_failures),
            },
            "narrative": (
                f"Context handoff at pressure '{pressure}': history compacted to "
                f"{len(messages)} messages; task state and open verification failures preserved."
            ),
            "decisions": [],
            "next_steps": [],
            "warnings": [],
            "messages": [m.model_dump() for m in messages],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)

    def _fit_history_to_budget(self, messages: list[LLMMessage], budget: int) -> tuple[list[LLMMessage], str | None]:
        if budget < 0:
            return [], "dropped"
        if self._tokenizer.count_messages(messages) <= budget:
            return messages, None

        compacted, summary = self._compact_history(messages)
        if summary and self._tokenizer.count_messages(compacted) <= budget:
            return compacted, "compact"
        working = compacted if summary else list(messages)

        preserve_first = bool(working and self._is_compaction_message(working[0]))

        while len(working) > 1 and self._tokenizer.count_messages(working) > budget:
            if preserve_first:
                working = [working[0], *working[2:]]
            else:
                working = working[1:]
        if self._tokenizer.count_messages(working) <= budget:
            return working, "drop_oldest"

        if not working:
            return [], "dropped"

        newest = working[-1]
        kept_blocks: list[ContentBlock] = []
        remaining = budget
        for block in reversed(newest.content):
            block_tokens = self._tokenizer.count_block(block)
            if block_tokens <= remaining:
                kept_blocks.insert(0, block)
                remaining -= block_tokens
                continue
            if remaining > 0:
                kept_blocks.insert(0, self._truncate_block(block, remaining))
                remaining = 0
        if not kept_blocks:
            return [], "dropped"
        return [LLMMessage(role=newest.role, content=kept_blocks)], "trim"

    # -- public assemble + reconcile ----------------------------------------

    def assemble(
        self,
        messages: list[LLMMessage],
        *,
        system_prompt: str | None = None,
        tool_definitions_text: str | None = None,
        context_sections: list[ContextSectionInput] | tuple[ContextSectionInput, ...] | None = None,
        file_contexts: Mapping[str, str] | None = None,
        memory_context: str | None = None,
        task_state: Mapping[str, object] | None = None,
        checkpoint_state: Mapping[str, object] | None = None,
        verification_failures: list[str] | tuple[str, ...] | None = None,
        artifact_handles: list[ContextArtifact] | tuple[ContextArtifact, ...] | None = None,
    ) -> tuple[list[LLMMessage], ContextManifest]:
        """Return possibly transformed messages plus a manifest describing the decision."""
        max_tokens = self._effective_window
        warnings: list[str] = []
        if self._profile_warning:
            warnings.append(self._profile_warning)

        extra_inputs = self._collect_context_section_inputs(
            context_sections=context_sections,
            file_contexts=file_contexts,
            memory_context=memory_context,
            task_state=task_state,
            checkpoint_state=checkpoint_state,
            verification_failures=verification_failures,
            artifact_handles=artifact_handles,
        )

        reserved = self._policy.reserved_response_tokens
        estimated = self._tokenizer.count_messages(messages)
        # Pressure runs on calibrated estimates: provider-actual counts feed a
        # running correction factor (see `note_actual`) so compaction never
        # fires late because local token math undercounts.
        calibrated = int(estimated * self._calibration)
        ratio = calibrated / max_tokens if max_tokens and max_tokens > 0 else 0.0
        pressure: ContextPressure = _classify_pressure(ratio, self._policy) if self._policy.enabled else "normal"

        sources: list[ContextSource] = [
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
        archive_path: str | None = None
        new_messages = list(messages)

        if pressure in ("mask", "offload", "compact", "handoff"):
            new_messages, masked_count = self._mask_aged_tool_results(new_messages)
            if masked_count:
                sources.append(
                    ContextSource(
                        kind="working_memory",
                        label=f"masked_tool_results[{masked_count}]",
                        estimated_tokens=0,
                    )
                )

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
            # Archive-first discipline: the untouched history goes to disk
            # before any summarization, so compaction is never lossy.
            archive_path = self._archive_history(messages)
            new_messages, compaction_summary = self._compact_history(new_messages, archive_path=archive_path)
            if compaction_summary:
                sources.append(
                    ContextSource(
                        kind="task_state",
                        label="compaction_summary",
                        estimated_tokens=int(len(compaction_summary) * _TOKENS_PER_CHAR),
                    )
                )
                # Invariants re-enter as a fresh, maximally recent message so
                # rules never depend on surviving compression.
                if not _has_recent_invariants(new_messages):
                    new_messages.append(build_invariant_message(self._policy, notice=f"{_INVARIANT_MARKER} History above was compacted."))

        if pressure == "handoff":
            handoff_path = self._write_handoff(new_messages, pressure)
            sources.append(
                ContextSource(
                    kind="task_state",
                    label="handoff_file",
                    estimated_tokens=0,
                )
            )

        # Per-section trim — applied after macro stages so we honor strict
        # section caps even when the global window has plenty of headroom.
        section_strategies: dict[ContextSectionKind, str] = {}
        if self._policy.sections:
            for kind, budget in self._policy.sections.items():
                if kind in ("reserved_response", "system_instructions", "tool_definitions"):
                    continue
                new_messages, _included, strategy = self._apply_section_trim(new_messages, kind, budget)
                if strategy:
                    section_strategies[kind] = strategy

        history_messages = new_messages
        section_messages, explicit_section_usages, explicit_sources, explicit_offloads = self._prepare_context_section_messages(extra_inputs)
        if explicit_sources:
            sources.extend(explicit_sources)
        if explicit_offloads:
            offloaded.extend(explicit_offloads)

        if max_tokens and max_tokens > 0:
            fixed_tokens = reserved + self._tokenizer.count_messages(section_messages)
            if system_prompt:
                fixed_tokens += self._tokenizer.count_text(system_prompt)
            if tool_definitions_text:
                fixed_tokens += self._tokenizer.count_text(tool_definitions_text)
            history_budget = max_tokens - fixed_tokens
            history_messages, fit_strategy = self._fit_history_to_budget(history_messages, history_budget)
            if fit_strategy:
                section_strategies["user_messages"] = fit_strategy
                if history_budget < 0:
                    warnings.append("First-class context sections plus reserved tokens exceed the model context window; chat history was dropped.")

        final_messages = [*section_messages, *history_messages]

        # Build per-section usage report.
        section_bins = self._classify_messages(history_messages)
        section_usages: list[ContextSectionUsage] = []

        section_usages.append(self._section_usage("reserved_response", reserved, reserved, strategy="reserved" if reserved else None))

        if system_prompt:
            section_usages.append(self._static_text_section_usage("system_instructions", system_prompt, warnings))

        if tool_definitions_text:
            section_usages.append(self._static_text_section_usage("tool_definitions", tool_definitions_text, warnings))

        section_usages.extend(explicit_section_usages)

        for kind, tokens in section_bins.items():
            section_usages.append(self._section_usage(kind, tokens, tokens, section_strategies.get(kind)))

        final_estimated = self._tokenizer.count_messages(final_messages)
        used_total = final_estimated + reserved
        if system_prompt:
            used_total += self._tokenizer.count_text(system_prompt)
        if tool_definitions_text:
            used_total += self._tokenizer.count_text(tool_definitions_text)
        available = (max_tokens - used_total) if (max_tokens and max_tokens > 0) else None

        usage_report = ContextUsageReport(
            max_context_tokens=max_tokens,
            reserved_response_tokens=reserved,
            used_tokens=used_total,
            available_tokens=available,
            sections=tuple(section_usages),
            counting_confidence=self._tokenizer.confidence,
            profile=self._profile,
        )

        manifest = ContextManifest(
            pressure=pressure,
            estimated_tokens=final_estimated,
            max_tokens=max_tokens or 0,
            sources=sources,
            offloaded=offloaded,
            compaction_summary=compaction_summary,
            handoff_path=handoff_path,
            archive_path=archive_path,
            preserved_task_state=dict(self._policy.preserved_task_state),
            preserved_verification_failures=tuple(self._policy.preserved_verification_failures),
            provider=self._provider,
            usage_report=usage_report,
            actual_input_tokens=None,
            warnings=tuple(warnings),
        )
        return final_messages, manifest

    def note_actual(self, manifest: ContextManifest) -> None:
        """Feed a reconciled manifest's provider-actual token count back into
        the calibration factor used by pressure classification."""
        report = manifest.usage_report
        actual = manifest.actual_input_tokens
        if report is None or not actual:
            return
        estimated = report.used_tokens - report.reserved_response_tokens
        if estimated <= 0:
            return
        factor = actual / estimated
        # Exponential moving average, clamped so one outlier can't whipsaw the ladder.
        self._calibration = max(0.5, min(3.0, 0.7 * self._calibration + 0.3 * factor))

    @property
    def calibration(self) -> float:
        return self._calibration

    @staticmethod
    def reconcile(manifest: ContextManifest, usage: TokenUsage) -> ContextManifest:
        """Return a copy of `manifest` updated with provider-actual input tokens."""
        actual_input = usage.input_tokens
        report = manifest.usage_report
        if report is None:
            return manifest.model_copy(update={"actual_input_tokens": actual_input})
        new_report = report.model_copy(update={"counting_confidence": "provider"})
        return manifest.model_copy(
            update={
                "actual_input_tokens": actual_input,
                "usage_report": new_report,
            }
        )


def _has_recent_invariants(messages: list[LLMMessage]) -> bool:
    for msg in messages[-3:]:
        for block in msg.content:
            if isinstance(block, TextBlock) and _INVARIANT_MARKER in block.text:
                return True
    return False


def rebuild_from_handoff(path: str | Path) -> list[LLMMessage]:
    """Restore a list of messages from a handoff artifact written by ContextManager."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_messages = payload.get("messages") or []
    return [LLMMessage.model_validate(item) for item in raw_messages]


def build_invariant_message(policy: ContextPolicy, *, notice: str) -> LLMMessage:
    """Build the invariant re-injection message appended after a compaction or
    reset boundary: rules and constraints must not depend on surviving
    compression, so they re-enter as a fresh, maximally recent user message.
    """
    lines = [notice]
    if policy.preserved_task_state:
        lines.append("TASK STATE (authoritative):")
        lines.extend(f"  {key}: {value}" for key, value in policy.preserved_task_state.items())
    if policy.preserved_verification_failures:
        lines.append("OPEN VERIFICATION FAILURES:")
        lines.extend(f"  - {failure}" for failure in policy.preserved_verification_failures)
    lines.append("Continue the task from this state. Do not re-do work already completed above.")
    return LLMMessage(role="user", content=[TextBlock(text="\n".join(lines))])
