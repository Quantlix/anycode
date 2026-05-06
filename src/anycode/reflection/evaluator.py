"""Parse critic JSON output into a CriticResult."""

from __future__ import annotations

import json
import re

from anycode.types import CriticResult

_JSON_FENCE = re.compile(r"\{[\s\S]*\}")


def parse_critic_json(text: str, *, threshold: float = 0.7) -> CriticResult:
    """Best-effort parse of a critic's JSON response. Falls back to a neutral score."""
    match = _JSON_FENCE.search(text)
    if not match:
        return CriticResult(score=0.5, passed=False, feedback=f"Could not parse critic output: {text[:200]}", suggestions=[])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return CriticResult(score=0.5, passed=False, feedback=f"Invalid JSON in critic output: {text[:200]}", suggestions=[])

    score = float(data.get("score", 0.0))
    passed = bool(data.get("passed", score >= threshold))
    feedback = str(data.get("feedback", ""))
    raw_suggestions = data.get("suggestions") or []
    suggestions = [str(s) for s in raw_suggestions if isinstance(s, str)]

    return CriticResult(score=max(0.0, min(1.0, score)), passed=passed, feedback=feedback, suggestions=suggestions)
