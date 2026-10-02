from __future__ import annotations

import re

from ike.retrieval.table_context import retrieval_text
from ike.retrieval.types import Candidate

_QUERY_STOP = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
    "were", "be", "list", "show", "give", "provide", "all", "every", "requested",
    "identify", "establish", "set", "details", "documented",
}
_ENUMERATION_SECTION_RE = re.compile(
    r"\b(?:list|directory|index|register|inventory|catalog(?:ue)?|schedule|"
    r"summary|details|locations?)\b",
    re.IGNORECASE,
)
_VALUE_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:%|percent|rs\.?|inr|km/?h|kmph|km|m|"
    r"hours?|hrs?|minutes?|mins?|days?|months?|years?)\b|₹\s*\d)",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(?:step|procedure|shall|must|should|required|submit|attach|follow|"
    r"ensure|before|after|approval|form|invoice|receipt)\b",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"\b(?:means|refers\s+to|is\s+defined\s+as|shall\s+mean|definition)\b",
    re.IGNORECASE,
)
_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d{1,3}[.)]|[A-Za-z][.)])\s+",
    re.MULTILINE,
)


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_]{1,}", query or ""):
        folded = token.casefold().strip("_-/")
        if len(folded) < 3 or folded in _QUERY_STOP:
            continue
        terms.add(folded)
        # Cheap morphology bridge so "depots" can match "depot" and vice versa.
        if len(folded) > 4 and folded.endswith("s") and not folded.endswith(("ss", "is", "us")):
            terms.add(folded[:-1])
        elif len(folded) > 4:
            terms.add(f"{folded}s")
    return terms


def _concept_stem_sequence(query: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_]{1,}", query or ""):
        folded = token.casefold().strip("_-/")
        if len(folded) < 3 or folded in _QUERY_STOP:
            continue
        if len(folded) > 4 and folded.endswith("s") and not folded.endswith(("ss", "is", "us")):
            folded = folded[:-1]
        if folded not in seen:
            seen.add(folded)
            result.append(folded)
    return result


def _anchor_stems(query: str) -> set[str]:
    stems = _concept_stem_sequence(query)
    # In ordinary noun phrases the final token is frequently a generic head noun
    # (station, control, rule, charge, etc.). Require the preceding modifier(s) too so
    # "interchange stations" cannot be satisfied by an arbitrary table of stations.
    return set(stems[:-1] if len(stems) > 1 else stems)


def _query_overlap(query: str, candidate: Candidate) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    haystack = (
        retrieval_text(candidate)
        + "\n"
        + " > ".join(candidate.section_path or [])
        + "\n"
        + (candidate.document_title or "")
    ).casefold()
    anchors = _anchor_stems(query)
    if anchors and not all(anchor in haystack for anchor in anchors):
        return 0.0
    hits = sum(1 for term in terms if term and term in haystack)
    return min(1.0, hits / max(1, min(len(terms), 5)))


def _concept_stems(query: str) -> set[str]:
    stems: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_]{1,}", query or ""):
        folded = token.casefold().strip("_-/")
        if len(folded) < 3 or folded in _QUERY_STOP:
            continue
        if len(folded) > 4 and folded.endswith("s") and not folded.endswith(("ss", "is", "us")):
            folded = folded[:-1]
        stems.add(folded)
    return stems


def _field_label_strength(query: str, text: str) -> float:
    """Detect repeated record fields tied to the requested concept.

    Docling often linearizes a table as ``1, Location of Depot = ...`` or
    ``2, Depot Name = ...``. Repetition of a concept-bearing field is much stronger
    enumeration evidence than a table that merely mentions the concept in narrative cells.
    """

    stems = _concept_stems(query)
    if not stems:
        return 0.0
    labels = re.findall(
        r"(?:^|[.;]\s*|\d{1,3}[,.]\s*)([A-Za-z][A-Za-z0-9 /_-]{1,55}?)\s*=",
        text,
        flags=re.IGNORECASE,
    )
    related = 0
    for label in labels:
        label_tokens = {
            token.casefold().rstrip("s")
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", label)
        }
        if any(stem.rstrip("s") in label_tokens for stem in stems):
            related += 1
    if related >= 3:
        return 1.0
    if related == 2:
        return 0.65
    if related == 1:
        return 0.3
    return 0.0


def _short_matching_heading(query: str, candidate: Candidate) -> bool:
    if not candidate.section_path:
        return False
    local = str(candidate.section_path[-1] or "").casefold()
    tokens = re.findall(r"[a-z0-9]+", local)
    if not tokens or len(tokens) > 8:
        return False
    stems = _concept_stems(query)
    normalized_tokens = {token.rstrip("s") for token in tokens}
    return bool(stems) and all(stem.rstrip("s") in normalized_tokens for stem in stems)


def enumeration_shape_score(query: str, candidate: Candidate) -> float:
    """Estimate whether a candidate can actually carry a requested set/list.

    This is deliberately corpus/domain neutral. It rewards structural evidence such as
    tables, list-like rows and list/directory-style sections only when the candidate also
    overlaps the requested concept. A paragraph that merely mentions a term should remain
    weaker than a table or section that can enumerate members.
    """

    overlap = _query_overlap(query, candidate)
    if overlap <= 0.0:
        return 0.0

    text = retrieval_text(candidate)
    section = " > ".join(candidate.section_path or [])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    field_strength = _field_label_strength(query, text)
    short_heading = _short_matching_heading(query, candidate)
    score = 0.08 * overlap

    # A table is useful structure, but table-ness alone must never imply that it enumerates
    # the requested concept. Member-bearing fields/headings supply the stronger signal.
    if candidate.content_kind == "table":
        score += 0.18
    if _ENUMERATION_SECTION_RE.search(section):
        score += 0.16
    if field_strength:
        score += 0.38 * field_strength
    if short_heading:
        score += 0.22
    if len(lines) >= 5:
        score += 0.06
    if len(_LIST_LINE_RE.findall(text)) >= 3:
        score += 0.12
    if text.count("|") >= 4 or text.count("\t") >= 4:
        score += 0.08

    return min(1.0, score)


def value_shape_score(query: str, candidate: Candidate) -> float:
    """Estimate whether a candidate can support a rate/limit/amount style answer."""

    overlap = _query_overlap(query, candidate)
    if overlap <= 0.0:
        return 0.0
    text = retrieval_text(candidate)
    score = 0.12 * overlap
    if candidate.content_kind == "table":
        score += 0.28
    if _VALUE_RE.search(text):
        score += 0.38
    if re.search(r"\b(?:rate|limit|ceiling|amount|charge|entitlement|grade|scale)\b", text, re.IGNORECASE):
        score += 0.12
    return min(1.0, score)


def procedure_shape_score(query: str, candidate: Candidate) -> float:
    """Estimate whether a candidate carries procedural/actionable structure."""

    overlap = _query_overlap(query, candidate)
    if overlap <= 0.0:
        return 0.0
    text = retrieval_text(candidate)
    score = 0.12 * overlap
    if len(_LIST_LINE_RE.findall(text)) >= 2:
        score += 0.28
    procedure_hits = len(_PROCEDURE_RE.findall(text))
    if procedure_hits:
        score += min(0.42, 0.08 * procedure_hits)
    return min(1.0, score)


def definition_shape_score(query: str, candidate: Candidate) -> float:
    overlap = _query_overlap(query, candidate)
    if overlap <= 0.0:
        return 0.0
    score = 0.15 * overlap
    if _DEFINITION_RE.search(retrieval_text(candidate)):
        score += 0.60
    if candidate.section_path and any(
        token in (candidate.section_path[-1] or "").casefold()
        for token in _query_terms(query)
    ):
        score += 0.15
    return min(1.0, score)


def evidence_shape_score(goal_kind: str, query: str, candidate: Candidate) -> float:
    kind = (goal_kind or "").casefold()
    if kind in {"enumeration", "overview", "responsibility"}:
        return enumeration_shape_score(query, candidate)
    if kind in {"attribute", "calculation"}:
        return value_shape_score(query, candidate)
    if kind == "procedure":
        return procedure_shape_score(query, candidate)
    if kind == "definition":
        return definition_shape_score(query, candidate)
    return 0.0
