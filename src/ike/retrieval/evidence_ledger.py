from __future__ import annotations

from dataclasses import dataclass, field
import re
from uuid import UUID

from ike.retrieval.types import Evidence
from ike.workflows.evidence_planning import EvidencePlan, GoalSatisfaction


@dataclass(slots=True)
class EvidenceLedgerResult:
    evidence: list[Evidence]
    protected_chunk_ids: list[UUID] = field(default_factory=list)
    added_chunk_ids: list[UUID] = field(default_factory=list)
    dropped_chunk_ids: list[UUID] = field(default_factory=list)

    def trace(self) -> dict:
        return {
            "protected_chunk_ids": [str(item) for item in self.protected_chunk_ids],
            "added_chunk_ids": [str(item) for item in self.added_chunk_ids],
            "dropped_chunk_ids": [str(item) for item in self.dropped_chunk_ids],
            "evidence_count": len(self.evidence),
        }


def _goal_ids(item: Evidence) -> set[str]:
    return {
        source.split(":", 1)[1]
        for source in item.candidate.sources
        if source.startswith("goal:") and source.count(":") == 1
    }


def _next_evidence_number(evidence: list[Evidence]) -> int:
    highest = 0
    for item in evidence:
        match = re.fullmatch(r"E(\d+)", str(item.evidence_id).upper())
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def monotonic_merge(
    primary: list[Evidence],
    recovered: list[Evidence],
    *,
    limit: int,
    plan: EvidencePlan | None = None,
    satisfaction: GoalSatisfaction | None = None,
) -> EvidenceLedgerResult:
    """Merge staged retrieval into a monotonic *support* ledger.

    Evidence explicitly cited by an earlier supported/partial semantic audit is immutable
    support and therefore cannot be evicted. Recovery for unresolved goals is admitted next,
    before unused primary candidates, so a full earlier top-k cannot block a successful
    repair. Primary evidence IDs remain stable; newly admitted recovery evidence receives
    fresh IDs above the primary ledger's current maximum.
    """

    del plan  # Reserved for future claim/cardinality-aware replacement policies.

    protected_ids: set[str] = set()
    unresolved: set[str] = set()
    if satisfaction is not None:
        for status in satisfaction.statuses:
            if status.status in {"supported", "partial"}:
                protected_ids.update(str(value).upper() for value in status.evidence_ids)
        unresolved.update(satisfaction.missing_goal_ids)
        unresolved.update(satisfaction.partial_goal_ids)

    selected: list[Evidence] = []
    seen: set[UUID] = set()
    protected_chunks: list[UUID] = []
    added_chunks: list[UUID] = []
    next_number = _next_evidence_number(primary)

    def append_primary(item: Evidence, *, protected: bool = False) -> bool:
        if item.candidate.chunk_id in seen or len(selected) >= limit:
            return False
        selected.append(item)
        seen.add(item.candidate.chunk_id)
        if protected:
            protected_chunks.append(item.candidate.chunk_id)
        return True

    def append_recovered(item: Evidence) -> bool:
        nonlocal next_number
        if item.candidate.chunk_id in seen or len(selected) >= limit:
            return False
        admitted = Evidence(evidence_id=f"E{next_number}", candidate=item.candidate)
        next_number += 1
        selected.append(admitted)
        seen.add(admitted.candidate.chunk_id)
        added_chunks.append(admitted.candidate.chunk_id)
        return True

    # 1) Never discard documentary support explicitly relied on by the previous audit.
    for item in primary:
        if item.evidence_id.upper() in protected_ids:
            append_primary(item, protected=True)

    # 2) Admit recovery evidence for unresolved goals *before* low-value residual primary
    # candidates. This is what allows retrieval repair to improve a full prior evidence set.
    targeted: list[Evidence] = []
    other: list[Evidence] = []
    for item in recovered:
        if unresolved and _goal_ids(item) & unresolved:
            targeted.append(item)
        else:
            other.append(item)
    targeted.sort(key=lambda item: item.candidate.rerank_score, reverse=True)
    other.sort(key=lambda item: item.candidate.rerank_score, reverse=True)
    for item in targeted:
        append_recovered(item)

    # 3) Preserve as much earlier context as capacity permits. Supported evidence is already
    # guaranteed above; unprotected candidates may yield space to a successful recovery.
    for item in primary:
        append_primary(item)

    # 4) Optional new context fills only remaining capacity.
    for item in other:
        append_recovered(item)

    all_ids = {item.candidate.chunk_id for item in [*primary, *recovered]}
    selected_ids = {item.candidate.chunk_id for item in selected}
    return EvidenceLedgerResult(
        evidence=selected,
        protected_chunk_ids=protected_chunks,
        added_chunk_ids=added_chunks,
        dropped_chunk_ids=list(all_ids - selected_ids),
    )

