"""Helpers for corpus-generic role/responsibility coverage retrieval.

The helpers deliberately avoid organisation-specific role dictionaries. Short role
identifiers such as ``SC`` are resolved from document language (for example,
``Responsibilities of Station Controller`` or ``Station Controller (SC)``), then the
retrieval engine performs deterministic section discovery for the resolved role names.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from ike.retrieval.types import Candidate

_INITIALISM_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
}
_ROLE_CUE = r"(?:responsibilit(?:y|ies)|dut(?:y|ies)|functions?|obligations?|role(?:s)?)"
_ROLE_PHRASE = r"[A-Za-z][A-Za-z0-9/&+,'’.-]*(?:\s+[A-Za-z][A-Za-z0-9/&+,'’.-]*){0,7}"


def is_short_role_identifier(value: str) -> bool:
    cleaned = value.strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_./-]{1,11}", cleaned)) and (
        cleaned.isupper() or len(cleaned) <= 4
    )


def phrase_initialism(phrase: str) -> str:
    words = re.findall(r"[A-Za-z]+", phrase)
    kept = [word for word in words if word.lower() not in _INITIALISM_STOP_WORDS]
    return "".join(word[0].upper() for word in kept)


def _clean_role_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:;,.()-–—")
    # Avoid swallowing sentence tails after a role phrase.
    value = re.split(
        r"\b(?:shall|must|may|will|who|which|where|when|while|during|under|with|without)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" :;,.()-–—")
    # Corpus text often yields grammatical wrappers such as 'the Station Controller'
    # or 'and Station Controller'. Remove only leading function words; never remove
    # domain words from the role itself.
    value = re.sub(r"^(?:(?:the|a|an|and|or)\s+)+", "", value, flags=re.IGNORECASE)
    return value[:120]


def extract_role_aliases(identifier: str, texts: Iterable[str]) -> list[str]:
    """Extract full role names whose initialism matches ``identifier``.

    Sources include conventional acronym definitions and structural role headings. The
    returned aliases are evidence-derived candidates only; they are not a knowledge-base
    lookup and no organisation-specific expansion is hard-coded.
    """

    term = identifier.strip()
    if not is_short_role_identifier(term):
        return []
    target = re.sub(r"[^A-Za-z0-9]", "", term).upper()
    escaped = re.escape(term)
    counter: Counter[str] = Counter()

    patterns: tuple[tuple[re.Pattern[str], int], ...] = (
        (
            re.compile(rf"(?P<role>{_ROLE_PHRASE})\s*\(\s*{escaped}\s*\)", re.IGNORECASE),
            8,
        ),
        (
            re.compile(rf"\b{escaped}\b\s*\(\s*(?P<role>[^)\n]{{3,120}})\)", re.IGNORECASE),
            6,
        ),
        (
            re.compile(
                rf"\b{escaped}\b\s*(?:=|:|[-–—])\s*(?P<role>{_ROLE_PHRASE})",
                re.IGNORECASE,
            ),
            7,
        ),
        (
            re.compile(rf"\b{_ROLE_CUE}\s+of\s+(?P<role>{_ROLE_PHRASE})", re.IGNORECASE),
            10,
        ),
        (
            re.compile(rf"\bevery\s+(?P<role>{_ROLE_PHRASE})\s+shall\b", re.IGNORECASE),
            9,
        ),
    )

    for text_value in texts:
        if not text_value:
            continue
        for pattern, weight in patterns:
            for match in pattern.finditer(text_value):
                role = _clean_role_phrase(match.group("role"))
                if not role or role.lower() == term.lower():
                    continue
                if phrase_initialism(role) == target:
                    counter[role] += weight

    # Collapse case/spacing variants while preserving the strongest surface form.
    grouped: dict[str, tuple[str, int]] = {}
    for role, score in counter.items():
        key = re.sub(r"\s+", " ", role).casefold()
        previous = grouped.get(key)
        if previous is None or score > previous[1]:
            grouped[key] = (role, score)
        elif score == previous[1] and len(role) < len(previous[0]):
            grouped[key] = (role, score)

    return [role for role, _score in sorted(grouped.values(), key=lambda item: (-item[1], len(item[0]), item[0].lower()))]


def role_candidate_score(aliases: Iterable[str], candidate: Candidate) -> int:
    """Score how directly a chunk assigns responsibilities to a role."""

    text_value = candidate.contextual_text or candidate.text
    section = " / ".join(candidate.section_path or [])
    combined = f"{section}\n{text_value}"
    score = 0

    for alias in aliases:
        escaped = re.escape(alias.strip())
        if not escaped:
            continue
        if re.search(rf"\bresponsibilit(?:y|ies)\s+of\s+{escaped}\b", combined, re.IGNORECASE):
            score = max(score, 30)
        if re.search(rf"\bdut(?:y|ies)\s+of\s+{escaped}\b", combined, re.IGNORECASE):
            score = max(score, 28)
        if re.search(rf"\bfunctions?\s+of\s+{escaped}\b", combined, re.IGNORECASE):
            score = max(score, 26)
        if re.search(rf"\bevery\s+{escaped}\s+shall\b", combined, re.IGNORECASE):
            score = max(score, 25)
        if re.search(rf"\b{escaped}\s+shall\b", combined, re.IGNORECASE):
            score = max(score, 22)
        if re.search(rf"\b{escaped}\s+must\b", combined, re.IGNORECASE):
            score = max(score, 20)
        if re.search(rf"\b{escaped}\b.{0,80}\bresponsib", combined, re.IGNORECASE | re.DOTALL):
            score = max(score, 18)
        if re.search(rf"\b{escaped}\b", section, re.IGNORECASE):
            score += 4
        if re.search(rf"\b{escaped}\b", text_value, re.IGNORECASE):
            score += 2

    if re.search(_ROLE_CUE, section, re.IGNORECASE):
        score += 4
    return score


def role_section_key(candidate: Candidate) -> tuple[str, str]:
    """Return a stable section-diversity key for role coverage selection."""

    section = " / ".join(candidate.section_path or [])
    normalized = re.sub(r"\s+", " ", section).strip().casefold()
    if not normalized:
        # Chunks without headings still need local diversity. A small ordinal bucket avoids
        # treating every neighboring chunk as a separate responsibility section.
        normalized = f"ordinal-bucket:{candidate.ordinal // 3}"
    return str(candidate.document_id), normalized
