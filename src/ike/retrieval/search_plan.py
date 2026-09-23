from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Search-engine query construction is deliberately domain-neutral.  These words are
# grammatical/request scaffolding, not corpus concepts.  Removing them from a relaxed
# lexical probe improves recall without inventing facts or domain terminology.
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "is", "it", "its", "may", "me", "of", "on", "or", "our", "please", "shall", "should",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}


def word_tokens(value: str) -> list[str]:
    """Return Unicode-aware word tokens without depending on a language model/tokenizer."""

    result: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        token = "".join(current).strip("./-_")
        if token:
            result.append(token)
        current.clear()

    for char in str(value or ""):
        category = unicodedata.category(char)
        if category[:1] in {"L", "M", "N"} or char == "_":
            current.append(char)
            continue
        if char in "./-" and current:
            current.append(char)
            continue
        flush()
    flush()
    return result


def content_tokens(value: str, *, max_terms: int = 12) -> list[str]:
    """Extract query concepts for recall-oriented lexical/section probes.

    The original dense query remains untouched.  This representation only strips common
    request grammar and very short noise tokens.  It never adds a domain term, answer,
    acronym expansion, threshold, place, or document title.
    """

    result: list[str] = []
    seen: set[str] = set()
    for token in word_tokens(value):
        key = token.casefold()
        if key in _QUERY_STOPWORDS:
            continue
        if len(key) <= 1 and not key.isdigit():
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
        if len(result) >= max_terms:
            break
    return result


def relaxed_websearch_expression(value: str, *, max_terms: int = 10) -> str:
    """Build a PostgreSQL websearch OR expression from user/planner concepts.

    Ordinary ``websearch_to_tsquery`` prose behaves mostly like an AND query.  That is
    precise but brittle for paraphrases.  IMS 0.8 adds a separate *relaxed* lane rather than
    weakening the existing lexical lane.  ts_rank still rewards passages matching several
    concepts, while one missing surface word no longer produces an empty lexical result.
    """

    terms = content_tokens(value, max_terms=max_terms)
    if not terms:
        return ""
    escaped = [term.replace('"', "") for term in terms if term.replace('"', "")]
    return " OR ".join(f'"{term}"' for term in escaped)


def token_overlap(query: str, candidate: str) -> float:
    q = {item.casefold() for item in content_tokens(query, max_terms=20)}
    c = {item.casefold() for item in content_tokens(candidate, max_terms=80)}
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def materially_novel_query(candidate: str, previous: Iterable[str], *, threshold: float = 0.82) -> bool:
    """Reject recovery probes that are only another near-duplicate paraphrase.

    We use a conservative token-Jaccard check.  This is not a relevance model; it simply
    prevents recovery from spending another retrieval pass on the same lexical formulation.
    """

    candidate_tokens = {item.casefold() for item in content_tokens(candidate, max_terms=24)}
    if not candidate_tokens:
        return False
    for prior in previous:
        prior_tokens = {item.casefold() for item in content_tokens(prior, max_terms=24)}
        if not prior_tokens:
            continue
        score = len(candidate_tokens & prior_tokens) / len(candidate_tokens | prior_tokens)
        if score >= threshold:
            return False
    return True


def table_retrieval_relevant(*, goal_kinds: Iterable[str], facets: Iterable[str], question: str) -> bool:
    """Decide whether a table-specific lane is structurally justified.

    Table retrieval is valuable for lists, matrices and numeric/attribute facts, but using a
    high-weight table lane for every responsibility/relationship question creates systematic
    pollution from layout/contact tables.  Dense/lexical retrieval can still return a table;
    this function only gates the *extra table-specific boost*.
    """

    kinds = {str(value).casefold() for value in goal_kinds}
    facet_set = {str(value).casefold() for value in facets}
    if kinds & {"enumeration", "attribute", "comparison", "calculation"}:
        return True
    if facet_set & {"enumeration", "location", "contact", "constraint"}:
        return True
    return bool(
        re.search(
            r"\b(?:table|matrix|schedule|timetable|list|directory|chart|limit|limits|duration|"
            r"frequency|contact|location|locations|repayment|amount|number|numbers|"
            r"max(?:imum)?|min(?:imum)?|highest|lowest|rated|nominal|capacity|dimension|dimensions|"
            r"length|width|height|weight|rate|ratio|percentage|percent|quantity|value|values|"
            r"specification|specifications|parameter|parameters)\b",
            question,
            re.IGNORECASE,
        )
    )
