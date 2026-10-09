from __future__ import annotations

import re

from ike.retrieval.types import Candidate


def _fingerprint(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]{3,}", text or "")
    }


def deduplicate_candidates(candidates: list[Candidate], *, threshold: float = 0.88) -> list[Candidate]:
    """Remove near-duplicate evidence while preserving scenario/document differences."""
    kept: list[Candidate] = []
    fingerprints: list[set[str]] = []
    for candidate in candidates:
        fp = _fingerprint(candidate.contextual_text or candidate.text)
        duplicate = False
        for existing, existing_fp in zip(kept, fingerprints, strict=True):
            if candidate.document_id != existing.document_id:
                continue
            # Never collapse different sections; repetition across a manual can be meaningful.
            if (candidate.section_path or []) != (existing.section_path or []):
                continue
            # Distinct table chunks often represent continuation rows of one large table.
            # Similar headers/labels must not cause later rows to disappear from evidence.
            if candidate.content_kind == "table" or existing.content_kind == "table":
                continue
            union = fp | existing_fp
            similarity = len(fp & existing_fp) / max(1, len(union))
            if similarity >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            fingerprints.append(fp)
    return kept
