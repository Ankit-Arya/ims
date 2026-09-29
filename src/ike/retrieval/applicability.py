from __future__ import annotations

import re

from ike.retrieval.query_plan import QueryPlan
from ike.retrieval.types import Candidate


def _canon(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def applicability_score(plan: QueryPlan, candidate: Candidate) -> float:
    """Conservative metadata applicability score.

    Unknown metadata never excludes evidence. Explicit mismatches are modestly penalized;
    matches are boosted before the expensive reranker.
    """
    profile = candidate.document_profile or {}
    score = 0.0
    if plan.line:
        values = [str(x) for x in profile.get("line_codes", []) if x]
        if values:
            score += 0.18 if any(_canon(v) == _canon(plan.line) for v in values) else -0.08
    if plan.rolling_stock:
        primary_values = [str(x) for x in profile.get("primary_rolling_stock", []) if x]
        fallback_values = [str(x) for x in profile.get("rolling_stock", []) if x]
        values = primary_values or fallback_values
        if values:
            # Primary applicability is authoritative when available. A manual may mention
            # other stock families for comparison, compatibility or common equipment; those
            # secondary mentions must not make it an exact-family match.
            score += 0.48 if any(_canon(v) == _canon(plan.rolling_stock) for v in values) else -0.48

    # Index/contents documents are useful navigation sources but should not outrank the
    # governing procedure for a substantive operational question. Keep them neutral for
    # enumeration/overview requests where the index itself may be the desired evidence.
    title = (candidate.document_title or "").casefold()
    local_sections = [str(x).strip().casefold() for x in (candidate.section_path or []) if str(x).strip()]
    index_like = bool(
        re.search(r"\b(?:index|table\s+of\s+contents|contents)\b", title)
        or (local_sections and local_sections[-1] in {"index", "table of contents", "contents"})
    )
    enumeration_request = "enumeration" in (plan.facets or []) or plan.answer_type == "structured_list"
    if index_like and not enumeration_request and plan.intent in {"procedure", "troubleshooting", "information"}:
        score -= 0.28
    return score
