from __future__ import annotations

from collections.abc import Iterable

from ike.retrieval.types import Candidate


def same_logical_section(anchor: Candidate, other: Candidate) -> bool:
    if anchor.document_id != other.document_id:
        return False
    a = [x.strip().casefold() for x in (anchor.section_path or []) if x.strip()]
    b = [x.strip().casefold() for x in (other.section_path or []) if x.strip()]
    if not a or not b:
        return abs(anchor.ordinal - other.ordinal) <= 1
    # The deepest known heading is the best boundary signal. A prefix match tolerates
    # a nested subsection while preventing drift into unrelated sibling procedures.
    shorter = min(len(a), len(b))
    common = 0
    for idx in range(shorter):
        if a[idx] != b[idx]:
            break
        common += 1
    return common >= max(1, min(len(a), len(b)) - 1)


def order_section_candidates(anchor: Candidate, candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(
        (item for item in candidates if same_logical_section(anchor, item)),
        key=lambda item: item.ordinal,
    )
