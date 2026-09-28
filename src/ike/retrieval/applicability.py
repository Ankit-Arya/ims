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
        values = [str(x) for x in profile.get("rolling_stock", []) if x]
        if values:
            score += 0.38 if any(_canon(v) == _canon(plan.rolling_stock) for v in values) else -0.32
    return score
