from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[str, list[UUID], float]],
    *,
    k: int = 60,
) -> dict[UUID, tuple[float, set[str]]]:
    scores: dict[UUID, float] = defaultdict(float)
    sources: dict[UUID, set[str]] = defaultdict(set)
    for source, ids, weight in ranked_lists:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] += weight / (k + rank)
            sources[chunk_id].add(source)
    return {chunk_id: (score, sources[chunk_id]) for chunk_id, score in scores.items()}
