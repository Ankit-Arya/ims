from __future__ import annotations

import re
from typing import Any

from ike.core.config import get_settings
from ike.retrieval.search_plan import materially_novel_query
from ike.workflows.evidence_planning import EvidencePlan, GoalSatisfaction, standalone_identifiers

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:[%°]|[A-Za-z]+)?(?![A-Za-z0-9])")
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9_./-]{1,15})(?![A-Za-z0-9])")
_IDENTIFIER_STOP = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "DO", "FOR", "FROM", "HOW", "I",
    "IF", "IN", "IS", "IT", "NO", "NOT", "OF", "ON", "OR", "THE", "THEN", "TO", "WHAT",
    "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WITH", "YES",
}


def _safe_hypothesis(value: str, original: str, trusted_corpus_text: str = "") -> bool:
    """Allow conceptual search hypotheses while blocking invented identifiers/numbers.

    A retrieval hypothesis is never evidence.  Ordinary words such as a possible role title
    or section concept are allowed because they can improve recall and are subsequently
    validated against the corpus.  New numeric thresholds and technical identifiers remain
    blocked because they can anchor search to a fabricated fact.
    """

    original_numbers = {item.casefold() for item in _NUMBER_RE.findall(original)}
    generated_numbers = {item.casefold() for item in _NUMBER_RE.findall(value)}
    if generated_numbers - original_numbers:
        return False

    original_identifiers = {item.casefold() for item in standalone_identifiers(original)}
    generated_identifiers = {
        match.group(1).casefold()
        for match in _IDENTIFIER_RE.finditer(value)
        if match.group(1).upper() not in _IDENTIFIER_STOP
    }
    new_identifiers = generated_identifiers - original_identifiers
    trusted = trusted_corpus_text.casefold()
    return all(identifier in trusted for identifier in new_identifiers)


def _preserves_goal_constraints(value: str, goal) -> bool:
    """Prevent recovery from silently substituting a different condition/relation."""
    query_tokens = {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", value)}
    stop = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are", "with"}
    for qualifier in goal.qualifiers or []:
        required = {
            token.casefold() for token in re.findall(r"[A-Za-z0-9]+", qualifier)
            if token.casefold() not in stop and len(token) > 2
        }
        if required:
            overlap = len(required & query_tokens)
            minimum = 1 if len(required) == 1 else 2
            if overlap < min(minimum, len(required)):
                return False
    # Preserve at least one entity anchor when the goal has explicit entities.
    entity_tokens = {
        token.casefold()
        for entity in goal.entity_terms or []
        for token in re.findall(r"[A-Za-z0-9]+", entity)
        if token.casefold() not in stop and len(token) > 1
    }
    if entity_tokens and not (entity_tokens & query_tokens):
        return False
    return True


def repair_system_prompt(max_queries_per_goal: int) -> str:
    return (
        "You are the retrieval-repair controller for an evidence-grounded internal-document system. "
        "Do NOT answer the user and do NOT state any guessed fact as true. The previous retrieval pass "
        "did not establish one or more evidence requirements. Generate materially different SEARCH-ONLY "
        "hypotheses for those unresolved goals. You MAY hypothesize likely section-heading vocabulary, "
        "role/function terminology, governing concepts, procedural nouns, synonyms and alternate ways a "
        "document could express the requested relation. Those hypotheses are probes only and can never be "
        "used as answer evidence. Prefer terminology visible in the supplied corpus snippets when useful. "
        "Do not merely change tense, pluralization or word order. Do not invent numeric values, acronym "
        "expansions, technical identifiers, places, outcomes or policy conclusions. For ambiguous verbs, "
        "change the retrieval relation so the requested subject-predicate-object meaning is explicit. "
        f"Return at most {max_queries_per_goal} new probes per unresolved goal."
    )


def repair_user_prompt(
    question: str,
    plan: EvidencePlan,
    satisfaction: GoalSatisfaction,
    attempted_queries: list[str],
    evidence_text: str,
) -> str:
    unresolved = set(satisfaction.missing_goal_ids + satisfaction.partial_goal_ids)
    goal_text = "\n\n".join(goal.prompt_block() for goal in plan.goals if goal.id in unresolved)
    attempted = "\n".join(f"- {item}" for item in attempted_queries[-30:]) or "- none"
    return (
        f"User question:\n{question}\n\nUnresolved evidence requirements:\n{goal_text}\n\n"
        f"Queries already attempted (do not repeat near-duplicates):\n{attempted}\n\n"
        f"Best corpus snippets/headings seen so far:\n{evidence_text or 'No useful snippet was retained.'}\n\n"
        "Return JSON {\"goals\": [{\"goal_id\": \"g1\", \"queries\": [\"new probe\"], "
        "\"strategy_note\": \"short explanation of how this differs\"}]}. Return only unresolved goals."
    )


def repair_queries_from_payload(
    payload: Any,
    *,
    question: str,
    plan: EvidencePlan,
    satisfaction: GoalSatisfaction,
    attempted_queries: list[str],
    max_queries_per_goal: int,
    trusted_corpus_text: str = "",
) -> dict[str, list[str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("goals"), list):
        return {}
    unresolved = set(satisfaction.missing_goal_ids + satisfaction.partial_goal_ids)
    valid_goals = {goal.id for goal in plan.goals if goal.id in unresolved}
    goals_by_id = {goal.id: goal for goal in plan.goals}
    result: dict[str, list[str]] = {}
    all_previous = list(attempted_queries)

    for row in payload["goals"]:
        if not isinstance(row, dict):
            continue
        goal_id = str(row.get("goal_id") or "").strip().lower()
        if goal_id not in valid_goals:
            continue
        queries: list[str] = []
        for raw in row.get("queries") or []:
            value = re.sub(r"\s+", " ", str(raw or "").strip())[:240]
            if not value or not _safe_hypothesis(value, question, trusted_corpus_text):
                continue
            if get_settings().semantic_drift_guard_enabled and not _preserves_goal_constraints(
                value, goals_by_id[goal_id]
            ):
                continue
            if not materially_novel_query(value, [*all_previous, *queries]):
                continue
            queries.append(value)
            if len(queries) >= max_queries_per_goal:
                break
        if queries:
            result[goal_id] = queries
            all_previous.extend(queries)
    return result
