"""Context engineering primitives — model profiles, tokenizers, reporting.

The adaptive context lifecycle is a section-aware engine with model-aware
ceilings, real trim/offload/compact/handoff, and developer-visible token
usage reports.

Public API is intentionally small — most consumers go through `ContextManager`
in `anycode.core.context_manager` and the policy-driven runner integration.
"""

from __future__ import annotations

from anycode.context.profiles import (
    BUILT_IN_PROFILES,
    PROVIDER_DEFAULT_PROFILES,
    UNBOUNDED_PROFILE,
    resolve_profile,
)
from anycode.context.reporting import format_usage_report, render_usage_report_table
from anycode.context.tokenizer import (
    DEFAULT_TOKENIZER,
    HeuristicTokenizer,
    Tokenizer,
    count_messages,
    count_text,
    select_tokenizer,
)

__all__ = [
    "BUILT_IN_PROFILES",
    "PROVIDER_DEFAULT_PROFILES",
    "UNBOUNDED_PROFILE",
    "DEFAULT_TOKENIZER",
    "HeuristicTokenizer",
    "Tokenizer",
    "count_messages",
    "count_text",
    "format_usage_report",
    "render_usage_report_table",
    "resolve_profile",
    "select_tokenizer",
]
