"""Pure helpers for conservative acronym/identifier definition retrieval."""

import re

from ike.retrieval.types import Candidate


def definition_score(term: str, candidate: Candidate) -> int:
    """Score source text for definition-like patterns without creating an answer.

    The score is used only to protect likely source definitions from candidate-budget
    crowding. All final meanings must still come from quoted/cited document evidence.
    """

    term_tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9]+", term or "")]
    escaped = r"\s+".join(term_tokens) if term_tokens else re.escape(term)
    text_value = candidate.contextual_text or candidate.text
    score = 0
    if re.search(rf'\b{escaped}\b[\'\"\u2019\u201d)]*\s+(?:shall\s+)?means?\b', text_value, re.IGNORECASE):
        score += 16
    if re.search(rf'\b{escaped}\b[\'\"\u2019\u201d)]*\s+(?:is|are)\s+(?:hereby\s+)?defined\s+as\b', text_value, re.IGNORECASE):
        score += 16
    if re.search(rf'\b{escaped}\b[\'\"\u2019\u201d)]*\s+(?:refers?\s+to|is\s+the\s+person|is\s+an?\s+)\b', text_value, re.IGNORECASE):
        score += 9
    if "definition" in " ".join(candidate.section_path).casefold() and re.search(rf"\b{escaped}\b", text_value, re.IGNORECASE):
        score += 6
    if re.search(rf"\b{escaped}\b(?:\s*,\s*\d+)?\s*(?:=|:|[-–—])\s*[A-Za-z]", text_value, re.IGNORECASE):
        score += 8
    if re.search(rf"[A-Za-z][A-Za-z0-9 /&+,'-]{{2,100}}\(\s*{escaped}\s*\)", text_value, re.IGNORECASE):
        score += 8
    if re.search(rf"\b{escaped}\b\s*\([^)]{{3,100}}\)", text_value, re.IGNORECASE):
        score += 5
    section = " ".join(candidate.section_path).lower()
    lowered = text_value.lower()
    if "abbreviation" in section or "abbreviation" in lowered:
        score += 4
    if "nomenclature" in section or "nomenclature" in lowered or "glossary" in section:
        score += 2
    if candidate.content_kind == "table":
        score += 1
    return score
