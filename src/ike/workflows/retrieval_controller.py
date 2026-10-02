from __future__ import annotations

from dataclasses import asdict, dataclass

from ike.retrieval.evidence_shape import evidence_shape_score
from ike.retrieval.types import Evidence
from ike.workflows.evidence_planning import EvidencePlan, GoalSatisfaction


@dataclass(slots=True)
class RecoveryDiagnosis:
    goal_id: str
    goal_kind: str
    status: str
    failure_reason: str
    recommended_strategies: list[str]
    evidence_ids: list[str]
    max_shape_score: float
    semantic_rerank_recommended: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _goal_evidence(goal_id: str, evidence_ids: list[str], evidence: list[Evidence]) -> list[Evidence]:
    by_id = {item.evidence_id.upper(): item for item in evidence}
    selected: list[Evidence] = []
    seen = set()

    for evidence_id in evidence_ids:
        item = by_id.get(str(evidence_id).upper())
        if item is not None and item.candidate.chunk_id not in seen:
            selected.append(item)
            seen.add(item.candidate.chunk_id)

    # The semantic auditor may intentionally return no evidence IDs for a missing goal even
    # though retrieval attributed weak candidates to that goal. Inspect those too so the
    # controller can distinguish "nothing found" from "mentions found but wrong evidence shape".
    for item in evidence:
        if f"goal:{goal_id}" not in item.candidate.sources:
            continue
        if item.candidate.chunk_id in seen:
            continue
        selected.append(item)
        seen.add(item.candidate.chunk_id)
    return selected


def diagnose_recovery(
    plan: EvidencePlan,
    satisfaction: GoalSatisfaction,
    evidence: list[Evidence],
    *,
    quality_issues: list[str] | None = None,
) -> list[RecoveryDiagnosis]:
    """Explain *why* each unresolved goal failed before choosing the next retrieval action.

    The diagnoses are retrieval-control hints only. They never become answer evidence and
    contain no organisation/domain-specific vocabulary.
    """

    issues = [str(value) for value in (quality_issues or []) if str(value).strip()]
    goals = {goal.id: goal for goal in plan.goals}
    diagnoses: list[RecoveryDiagnosis] = []

    for status in satisfaction.statuses:
        if status.status not in {"missing", "partial", "contradicted"}:
            continue
        goal = goals.get(status.goal_id)
        if goal is None:
            continue

        attributed = _goal_evidence(goal.id, list(status.evidence_ids or []), evidence)
        shape_scores = [
            evidence_shape_score(goal.kind, goal.question, item.candidate)
            for item in attributed
        ]
        max_shape = max(shape_scores, default=0.0)
        kind = (goal.kind or "fact").casefold()

        if not attributed:
            reason = "no_supporting_evidence"
        elif kind in {"enumeration", "overview", "responsibility"} and max_shape < 0.35:
            reason = "mention_without_set_bearing_evidence"
        elif kind in {"attribute", "calculation"} and max_shape < 0.28:
            reason = "related_evidence_without_value_bearing_structure"
        elif kind == "procedure" and max_shape < 0.28:
            reason = "related_evidence_without_procedural_structure"
        elif status.status == "contradicted":
            reason = "conflicting_evidence"
        else:
            reason = "partial_support"

        strategies: list[str] = []
        if reason == "mention_without_set_bearing_evidence":
            strategies.extend([
                "structured_evidence_search",
                "corpus_section_rediscovery",
                "semantic_query_repair",
                "goal_local_rerank",
            ])
        elif reason == "related_evidence_without_value_bearing_structure":
            strategies.extend([
                "table_or_value_search",
                "corpus_section_navigation",
                "semantic_query_repair",
                "goal_local_rerank",
            ])
        elif reason == "related_evidence_without_procedural_structure":
            strategies.extend([
                "same_section_expansion",
                "procedural_section_search",
                "semantic_query_repair",
                "goal_local_rerank",
            ])
        elif reason == "conflicting_evidence":
            strategies.extend([
                "source_authority_check",
                "independent_goal_rerank",
                "semantic_query_repair",
            ])
        else:
            strategies.extend([
                "corpus_section_navigation",
                "semantic_query_repair",
                "goal_local_rerank",
            ])

        if issues:
            strategies.append("quality_gate_recovery")

        diagnoses.append(
            RecoveryDiagnosis(
                goal_id=goal.id,
                goal_kind=goal.kind,
                status=status.status,
                failure_reason=reason,
                recommended_strategies=list(dict.fromkeys(strategies)),
                evidence_ids=[item.evidence_id for item in attributed],
                max_shape_score=round(max_shape, 4),
                semantic_rerank_recommended=("goal_local_rerank" in strategies),
            )
        )

    return diagnoses


def balanced_recovery_queries(
    goal_ids: list[str],
    expansions: dict[str, list[str]],
    *,
    minimum_total_budget: int,
) -> list[str]:
    """Round-robin recovery probes so one missing goal cannot starve its siblings."""

    ordered_goals = list(dict.fromkeys(goal_ids))
    budget = max(max(0, int(minimum_total_budget)), len(ordered_goals))
    per_goal = {
        goal_id: [
            " ".join(str(query or "").split())
            for query in expansions.get(goal_id, [])
            if str(query or "").strip()
        ]
        for goal_id in ordered_goals
    }
    selected: list[str] = []
    seen: set[str] = set()
    max_depth = max((len(items) for items in per_goal.values()), default=0)
    for depth in range(max_depth):
        for goal_id in ordered_goals:
            items = per_goal.get(goal_id, [])
            if depth >= len(items):
                continue
            query = items[depth]
            key = query.casefold()
            if not query or key in seen:
                continue
            seen.add(key)
            selected.append(query)
            if len(selected) >= budget:
                return selected
    return selected


def diagnoses_prompt_block(diagnoses: list[RecoveryDiagnosis]) -> str:
    if not diagnoses:
        return "No specific retrieval-gap diagnosis was produced."
    lines = [
        "The following diagnoses explain why the previous retrieval was insufficient.",
        "They are search-control hints only, not answer evidence:",
    ]
    for item in diagnoses:
        lines.append(
            f"- {item.goal_id} ({item.goal_kind}, {item.status}): "
            f"{item.failure_reason}; strategies={','.join(item.recommended_strategies)}; "
            f"evidence_shape={item.max_shape_score:.2f}"
        )
    return "\n".join(lines)
