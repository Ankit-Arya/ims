from __future__ import annotations

from dataclasses import dataclass

from ike.core.config import get_settings
from ike.retrieval.query_plan import QueryPlan
from ike.workflows.evidence_planning import EvidencePlan


@dataclass(frozen=True, slots=True)
class RetrievalEffort:
    level: str  # fast | focused | research
    reason: str


_RESEARCH_CONTRACTS = {
    "all_supported_variants",
    "enumerate_set",
    "compare_variants",
    "calculation_inputs",
}
_FOCUSED_CONTRACTS = {
    "ordered_procedure",
    "conditional_rule",
    "relationship_proof",
    "authority_proof",
    "scalar_with_condition",
    # A single bounded goal can ask for all requested entities/rows without requiring
    # corpus-wide Research. True multi-entity/comparison strategies still route Research
    # via the strategy and goal-count checks below.
    "all_requested_entities",
}


def classify_retrieval_effort(
    question: str,
    query_plan: QueryPlan,
    evidence_plan: EvidencePlan,
    *,
    requested_mode: str = "auto",
) -> RetrievalEffort:
    """Map an information need to a bounded retrieval effort.

    The classifier intentionally depends on the evidence plan rather than domain-specific
    terms.  It mirrors mature retrieval systems that skip expensive planning for simple
    lookups and reserve multi-query/coverage work for genuinely complex requests.
    """

    if requested_mode == "research":
        return RetrievalEffort("research", "Research was explicitly selected.")

    if not get_settings().adaptive_retrieval_effort_enabled:
        return RetrievalEffort(
            "research" if evidence_plan.needs_research else "focused",
            "Adaptive retrieval effort is disabled; using the compatibility route.",
        )

    required = evidence_plan.required_goals
    contracts = {goal.coverage_contract for goal in required}
    kinds = {goal.kind for goal in required}
    if (
        len(required) >= 5
        or contracts & _RESEARCH_CONTRACTS
        or query_plan.coverage_sensitive
        or evidence_plan.strategy in {"comparison", "enumeration", "overview", "multi_entity"}
    ):
        return RetrievalEffort(
            "research",
            "The request requires multi-goal, comparative, enumerative, or corpus-coverage retrieval.",
        )

    # ``needs_research`` from semantic planning must never be silently collapsed back to
    # the fast/direct path. If the plan does not require true corpus-wide decomposition,
    # treat it as Focused so one bounded broad/recovery pass can satisfy the missing fact.
    if evidence_plan.needs_research:
        return RetrievalEffort(
            "focused",
            "The evidence planner requested broader retrieval, but the request does not require full research fan-out.",
        )

    if (
        contracts & _FOCUSED_CONTRACTS
        or kinds & {"procedure", "relationship", "condition", "exception", "consequence", "applicability", "scenario"}
        or evidence_plan.needs_verification
    ):
        return RetrievalEffort(
            "focused",
            "The request needs a bounded procedural/conditional/relational evidence path.",
        )

    return RetrievalEffort(
        "fast",
        "The request is a bounded lookup/fact/definition and does not require corpus-wide coverage.",
    )
