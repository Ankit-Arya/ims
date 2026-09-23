from __future__ import annotations

from collections import defaultdict

from ike.core.config import get_settings
from ike.retrieval.types import Evidence
from ike.workflows.evidence_planning import EvidencePlan, GoalSatisfaction


def _goal_ids(item: Evidence) -> list[str]:
    return [
        source.split(":", 1)[1]
        for source in item.candidate.sources
        if source.startswith("goal:") and source.count(":") == 1
    ]


def _budget(effort: str) -> int:
    settings = get_settings()
    if effort == "fast":
        return settings.fast_draft_evidence_k
    if effort == "focused":
        return settings.focused_draft_evidence_k
    return settings.research_draft_evidence_k


def assemble_draft_context(
    evidence: list[Evidence],
    *,
    effort: str,
    plan: EvidencePlan | None,
    satisfaction: GoalSatisfaction | None,
) -> tuple[list[Evidence], dict]:
    """Create a diverse, goal-balanced generation context from the full evidence ledger."""

    settings = get_settings()
    limit = min(_budget(effort), max(1, settings.compositional_max_evidence_k))
    if len(evidence) <= limit:
        selected = list(evidence)
    else:
        by_id = {item.evidence_id.upper(): item for item in evidence}
        selected: list[Evidence] = []
        seen = set()
        per_document: defaultdict[object, int] = defaultdict(int)

        def add(item: Evidence, *, override_doc_cap: bool = False) -> None:
            if item.candidate.chunk_id in seen or len(selected) >= limit:
                return
            if (
                not override_doc_cap
                and per_document[item.candidate.document_id] >= settings.draft_evidence_per_document_max
            ):
                return
            selected.append(item)
            seen.add(item.candidate.chunk_id)
            per_document[item.candidate.document_id] += 1

        # Semantic-audit support is the highest-priority input to drafting.
        if satisfaction is not None:
            for status in satisfaction.statuses:
                for evidence_id in status.evidence_ids:
                    item = by_id.get(str(evidence_id).upper())
                    if item is not None:
                        add(item, override_doc_cap=True)

        # Guarantee a minimum number of evidence units per required goal before filling by score.
        if plan is not None:
            for goal in plan.required_goals:
                goal_items = [item for item in evidence if goal.id in _goal_ids(item)]
                goal_items.sort(key=lambda item: item.candidate.final_retrieval_score, reverse=True)
                for item in goal_items[: settings.draft_evidence_per_goal_min]:
                    add(item, override_doc_cap=True)

        ranked = sorted(
            evidence,
            key=lambda item: (
                item.candidate.final_retrieval_score,
                len(item.candidate.goal_rerank_scores),
                1 if "exact" in item.candidate.sources else 0,
            ),
            reverse=True,
        )
        for item in ranked:
            add(item)
        # If diversity caps left room, fill the remainder rather than losing useful context.
        if len(selected) < limit:
            for item in ranked:
                add(item, override_doc_cap=True)

    # Preserve ledger evidence IDs so generated citations continue to map to the full
    # retrieval trace and query-log evidence list.
    draft = list(selected)
    char_count = sum(len(item.prompt_block()) for item in draft)
    goal_map = {
        goal.id: [item.evidence_id for item in draft if goal.id in _goal_ids(item)]
        for goal in (plan.goals if plan is not None else [])
    }
    trace = {
        "effort": effort,
        "draft_context_count": len(draft),
        "draft_context_chars": char_count,
        "draft_context_token_estimate": max(1, char_count // 4),
        "draft_context_evidence_ids": [item.evidence_id for item in draft],
        "draft_context_chunk_ids": [str(item.candidate.chunk_id) for item in draft],
        "draft_context_document_ids": list(dict.fromkeys(str(item.candidate.document_id) for item in draft)),
        "draft_context_goal_map": goal_map,
    }
    return draft, trace
