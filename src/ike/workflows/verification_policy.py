from __future__ import annotations

import re

from ike.retrieval.query_plan import QueryPlan
from ike.retrieval.normalization import technical_identifier_variants
from ike.retrieval.types import Evidence

_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?\s*(?:km/?h|kmph|m/s|sec(?:ond)?s?|min(?:ute)?s?|hours?|V|kV|A|bar|mm|cm|m)\b", re.IGNORECASE)
_SAFETY = re.compile(r"\b(?:emergency|evacuat|fire|brake|isolate|traction|signal|protection|safety|accident|rescue)\b", re.IGNORECASE)


def verification_risk(question: str, evidence: list[Evidence], trace: dict) -> tuple[int, list[str]]:
    """Observable risk score; decisions, not hidden reasoning, are persisted in traces."""
    score = 0
    reasons: list[str] = []
    quality_issues = list(trace.get("retrieval_quality_issues") or [])
    if quality_issues:
        score += 3
        reasons.extend(f"retrieval_quality:{item}" for item in quality_issues)
    if not trace.get("coverage_complete", True):
        score += 2
        reasons.append("coverage_incomplete")
    if trace.get("coverage_sensitive"):
        score += 1
        reasons.append("coverage_sensitive")
    if trace.get("entity_attribute_coverage_sensitive"):
        score += 1
        reasons.append("entity_attribute_coverage")
    facets = trace.get("query_facets") or []
    if len(facets) >= 2:
        score += 1
        reasons.append("multi_facet_question")
    entity_terms = trace.get("entity_terms") or []
    if len(entity_terms) >= 2:
        score += 1
        reasons.append("multi_entity_set_question")
    if _SAFETY.search(question):
        score += 2
        reasons.append("operational_or_safety_topic")
    if _NUMERIC.search(question) or any(_NUMERIC.search(item.candidate.text or "") for item in evidence[:8]):
        score += 1
        reasons.append("numeric_or_limit_evidence")
    revisions = {item.candidate.revision for item in evidence if item.candidate.revision}
    if len(revisions) > 1:
        score += 1
        reasons.append("multiple_revisions")
    if trace.get("role_coverage_sensitive"):
        score += 1
        reasons.append("broad_role_synthesis")
    if trace.get("compositional"):
        score += 2
        reasons.append("compositional_evidence_plan")
    if trace.get("goal_complete") is False:
        score += 2
        reasons.append("required_goal_incomplete")
    goal_satisfaction = trace.get("goal_satisfaction") or {}
    if goal_satisfaction.get("contradicted_goal_ids"):
        score += 1
        reasons.append("contradicted_premise_or_goal")
    evidence_plan = trace.get("evidence_plan") or {}
    if evidence_plan.get("strategy") in {"relationship", "comparison", "conditional", "multi_hop", "claim_check"}:
        score += 1
        reasons.append("relation_or_reasoning_sensitive")
    return score, reasons

_SCALAR_REQUEST = re.compile(
    r"\b(?:speeds?|rates?|limits?|maximum|max\.?|minimum|min\.?|distance|duration|pressure|temperature|voltage|current|capacity|frequency)\b",
    re.IGNORECASE,
)
_COMMON_QUERY_WORDS = {
    "what", "which", "where", "when", "who", "why", "how", "please", "give", "provide", "tell",
    "the", "this", "that", "these", "those", "with", "from", "into", "about", "under", "over", "for",
    "and", "or", "but", "are", "is", "was", "were", "have", "has", "had", "does", "do", "can", "should",
    "would", "could", "line", "instruction", "instructions", "procedure", "procedures", "required", "action",
}

def _quality_text(evidence: list[Evidence], *, limit: int = 12) -> str:
    parts: list[str] = []
    for item in evidence[:limit]:
        c = item.candidate
        parts.extend([c.document_title or "", c.filename or "", " ".join(c.section_path or []), c.contextual_text or c.text or ""])
    return "\n".join(parts)


def _query_anchors(question: str) -> list[str]:
    anchors: list[str] = []
    for token in re.findall(r"[A-Za-z]{4,}", question.casefold()):
        if token in _COMMON_QUERY_WORDS or token in anchors:
            continue
        anchors.append(token)
    return anchors[:8]


def retrieval_quality_issues(
    question: str,
    evidence: list[Evidence],
    trace: dict,
    query_plan: QueryPlan,
) -> list[str]:
    """Cheap pre-answer checks for retrieval drift/incompleteness.

    These checks do not decide factual correctness. They only force one bounded recovery
    pass when observable query constraints are absent from the selected evidence.
    """
    if not evidence:
        return ["no_evidence"]
    issues: list[str] = []
    text = _quality_text(evidence)
    lowered = text.casefold()
    compact = re.sub(r"[^a-z0-9]", "", lowered)

    if query_plan.lookup_term:
        variants = technical_identifier_variants(query_plan.lookup_term, max_variants=8)
        compact_variants = {re.sub(r"[^a-z0-9]", "", value.casefold()) for value in variants}
        if compact_variants and not any(value and value in compact for value in compact_variants):
            issues.append("technical_identifier_not_represented")

    if query_plan.line:
        line_match = re.search(r"(\d{1,3}[a-z]?)", query_plan.line, re.IGNORECASE)
        if line_match:
            key = line_match.group(1).casefold()
            requested = (f"line {key}", f"line-{key}", f"line_{key}", f"line{key}")
            if not any(value in lowered for value in requested):
                issues.append("explicit_line_scope_not_represented")

    if "enumeration" in (query_plan.facets or []):
        if not trace.get("coverage_complete", False) or int(trace.get("coverage_documents_discovered", 0) or 0) == 0:
            issues.append("enumeration_coverage_incomplete")

    if _SCALAR_REQUEST.search(question) and not any(_NUMERIC.search(item.candidate.contextual_text or item.candidate.text or "") for item in evidence[:10]):
        issues.append("scalar_value_evidence_missing")

    rerank_details = trace.get("rerank_details") or {}
    if rerank_details.get("mode") in {"realtime_fusion_bypass", "qa_fusion_bypass"}:
        lexical_anchor = any(
            any(source in {"exact", "lexical", "lookup", "routed_lexical", "section", "governing"} for source in item.candidate.sources)
            for item in evidence[:6]
        )
        if not lexical_anchor:
            issues.append("fusion_only_semantic_unverified")

    anchors = _query_anchors(question)
    if query_plan.intent == "definition" and query_plan.lookup_term:
        # Source-scope words (for example a named rulebook in a scoped definition request)
        # identify the document family; they need not be repeated inside the definition
        # chunk itself.  The structural definition path already enforces source scope.
        lookup_words = {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", query_plan.lookup_term)
            if len(token) >= 3
        }
        anchors = [token for token in anchors if token.casefold() in lookup_words]
    if len(anchors) >= 2:
        matched = sum(1 for token in anchors if token in lowered)
        if matched / len(anchors) < 0.5:
            issues.append("query_anchor_drift")

    return list(dict.fromkeys(issues))

