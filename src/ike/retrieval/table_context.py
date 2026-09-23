from __future__ import annotations

import re
from typing import Any

from ike.retrieval.types import Candidate

_TABLE_KEYS = ("caption", "header", "heading", "title", "column", "table", "label")

_QUERY_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on", "at",
    "what", "which", "who", "when", "how", "is", "are", "was", "were", "be",
    "list", "show", "give", "provide", "all", "every", "its", "it", "s",
}


def _walk(value: Any, key_hint: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).casefold()
            if any(token in key_text for token in _TABLE_KEYS):
                found.extend(_walk(child, key_text))
            elif isinstance(child, (dict, list)):
                found.extend(_walk(child, key_text))
    elif isinstance(value, list):
        for child in value[:40]:
            found.extend(_walk(child, key_hint))
    elif isinstance(value, str) and any(token in key_hint for token in _TABLE_KEYS):
        cleaned = re.sub(r"\s+", " ", value).strip()
        if 1 < len(cleaned) <= 300:
            found.append(cleaned)
    return found


def table_metadata_context(candidate: Candidate, *, max_items: int = 12) -> list[str]:
    """Extract already-stored table title/header/caption context from Docling metadata."""
    if candidate.content_kind != "table":
        return []
    values = []
    values.extend(candidate.section_path or [])
    values.extend(_walk(candidate.source_metadata or {}))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key and key not in seen:
            unique.append(value)
            seen.add(key)
        if len(unique) >= max_items:
            break
    return unique


def retrieval_text(candidate: Candidate) -> str:
    """Text used by the reranker; table rows retain structural context.

    0.5.0 ingestion already embeds HybridChunker.contextualize(...) with
    repeat_table_header=True and stores Docling chunk metadata, so this improves ranking
    and generation without re-OCR/re-chunk/re-embedding.
    """
    contextual = (candidate.contextual_text or candidate.text or "").strip()
    if candidate.content_kind != "table":
        return contextual
    metadata = table_metadata_context(candidate)
    if not metadata:
        return contextual
    prefix = "Table context: " + " | ".join(metadata)
    if prefix.casefold() in contextual.casefold():
        return contextual
    return f"{prefix}\n{contextual}"


def table_query_affinity(query: str, candidate: Candidate) -> float:
    """Small pre-rerank boost for table rows when compact structured facts are likely.

    The boost is deliberately small: it prevents terse rows from being systematically
    dominated by verbose prose, but never makes a table relevant by itself.
    """
    if candidate.content_kind != "table":
        return 0.0
    query_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query)
        if len(token) > 2 and token.casefold() not in _QUERY_STOP_WORDS
    }
    if not query_terms:
        return 0.0
    haystack = retrieval_text(candidate).casefold()
    hits = sum(1 for token in query_terms if token in haystack)
    if not hits:
        return 0.0
    coverage = hits / max(1, len(query_terms))
    # terse table rows get slightly more protection because paragraphs otherwise dominate.
    brevity = 1.0 if len(candidate.text or "") < 500 else 0.5
    return min(0.18, 0.08 * coverage + 0.06 * brevity)
