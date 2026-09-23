from __future__ import annotations

import re


def normalize_term(value: str) -> str:
    """Normalize corpus vocabulary for fuzzy matching, not factual rewriting."""

    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n:;,-").casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def fuzzy_query_tokens(query: str, *, limit: int) -> list[str]:
    """Return ordinary-language tokens eligible for corpus-aware typo expansion.

    Uppercase and identifier-like tokens are deliberately excluded. Technical identifiers
    are handled by exact/identifier-variant retrieval and should not be mutated by generic
    spelling logic.
    """

    values: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9/_-]{2,}", query):
        if raw.isupper() or any(ch.isdigit() for ch in raw) or "/" in raw or "_" in raw:
            continue
        token = normalize_term(raw)
        if len(token) < 4 or token in seen:
            continue
        seen.add(token)
        values.append(token)
        if len(values) >= max(1, limit):
            break
    return values
