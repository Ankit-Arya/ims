from __future__ import annotations

import re
from dataclasses import dataclass, field

from ike.retrieval.normalization import canonical_query_text, lexical_form_variants, technical_identifier_variants
from ike.workflows.routing import (
    coverage_search_query,
    extract_entity_term,
    extract_entity_terms,
    extract_lookup_term,
    extract_role_subject,
    extract_source_scope,
    is_coverage_question,
    is_entity_attribute_coverage_question,
    is_role_coverage_question,
    query_facets,
)

_LINE_RE = re.compile(r"\bline[\s_-]?(\d{1,3}[a-z]?)\b", re.IGNORECASE)
_STOCK_RE = re.compile(r"\b(?:rs|rolling\s*stock)[\s_-]?(\d{1,4}[a-z]?)\b", re.IGNORECASE)
_OPERATIONAL_PROBLEM_RE = re.compile(
    r"\b(?:wrong|incorrect|failure|failed|fault|not\s+working|not\s+closing|not\s+opening|"
    r"doesn['’]?t\s+work|does\s+not\s+work|not\s+available|stuck|unable|abnormal|blank|showing|open|opened|smoke|fire|alarm|"
    r"problem|issue|is\s+happening|happening)\b",
    re.IGNORECASE,
)
_OPERATIONAL_OBJECT_RE = re.compile(
    r"\b(?:announcements?|pa\s*/?\s*pis|pis|doors?|trains?|signals?|brak(?:e|ing)s?|pantographs?|"
    r"traction|vcb|target\s+speed|speed|pea|passengers?|coupling|wind|ato|uto|atp|tims|tcms)\b",
    re.IGNORECASE,
)
_EXPLICIT_VERIFICATION_RE = re.compile(
    r"\b(?:is\s+it\s+true|is\s+this\s+true|verify|confirm\s+whether|whether|"
    r"is\s+it\s+mentioned|is\s+there\s+any\s+record)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class QueryPlan:
    """Observable, corpus-generic interpretation of one user question."""

    original: str
    normalized: str
    intent: str = "information"
    topic: str | None = None
    content_terms: list[str] = field(default_factory=list)
    scope_terms: list[str] = field(default_factory=list)
    source_scope: str | None = None
    line: str | None = None
    rolling_stock: str | None = None
    system: str | None = None
    subsystem: str | None = None
    scenario: str | None = None
    answer_type: str = "quick_reference"
    lookup_term: str | None = None
    entity_term: str | None = None
    entity_terms: list[str] = field(default_factory=list)
    role_subject: str | None = None
    coverage_kind: str | None = None
    facets: list[str] = field(default_factory=list)
    attribute_terms: list[str] = field(default_factory=list)
    short_or_vague: bool = False
    semantic_queries: list[str] = field(default_factory=list)
    lexical_queries: list[str] = field(default_factory=list)
    exact_terms: list[str] = field(default_factory=list)
    visual_intent: bool = False

    @property
    def coverage_sensitive(self) -> bool:
        return self.coverage_kind is not None


def _looks_like_operational_problem(text: str) -> bool:
    if _EXPLICIT_VERIFICATION_RE.search(text):
        return False
    return bool(_OPERATIONAL_PROBLEM_RE.search(text) and _OPERATIONAL_OBJECT_RE.search(text))


def _troubleshooting_queries(text: str) -> list[str]:
    queries = [text]
    lowered = text.casefold()
    if "announcement" in lowered or re.search(r"\bpa\s*/?\s*pis\b|\bpis\b", lowered):
        queries.extend([
            "announcement troubleshooting PA PIS",
            "automatic announcement manual announcement",
            "Train Operator announcement procedure",
            "wrong station announcement correction manual announcement",
        ])
    if "door" in lowered:
        queries.extend(["door fault troubleshooting procedure", "door isolation operating procedure"])
    if "signal" in lowered:
        queries.extend(["signal fault troubleshooting procedure", "signal failure operating instruction"])
    return _unique(queries, limit=6)


def _operational_semantic_queries(text: str) -> list[str]:
    lowered = text.casefold()
    queries = [text]
    if "wind" in lowered:
        queries.extend([
            "high wind speed train movement procedure",
            "thunderstorm cyclone train operating instruction",
            "wind threshold train operation action",
        ])
    if "coupl" in lowered:
        queries.extend([
            "train coupling procedure",
            "rescue train coupling procedure",
            "coupling speed brake communication procedure",
        ])
    if "vcb" in lowered and any(term in lowered for term in ("open", "opened", "trip", "tripped", "showing")):
        queries.extend([
            "VCB tripping VCB gets opened indication",
            "VCB open trip troubleshooting reset close OCC procedure",
        ])
    if ("speed" in lowered or "target speed" in lowered) and any(term in lowered for term in ("not available", "unavailable", "not received", "missing")):
        queries.extend([
            "target speed not available not received ATP troubleshooting",
            "ATP information unavailable ATP MCB reset ROS RM OCC procedure",
        ])
    if any(term in lowered for term in ("stuck", "immobile", "immobilised", "immobilized", "unable to move")):
        queries.extend([
            "immobile train mid section procedure",
            "rescue of immobilised revenue train",
            "detrainment passengers immobile train technical failure",
        ])
    return _unique(queries, limit=5)


def _intent(text: str, role: bool, procedure: bool, facets: list[str]) -> str:
    lowered = text.casefold()
    if role:
        return "responsibilities"
    # A live abnormal condition remains troubleshooting even when phrased as "what to do".
    # Otherwise conditional wording ("in case of") turns simple fault reports into expensive
    # research plans before the operational recovery path can run.
    if _looks_like_operational_problem(text) or any(term in lowered for term in ("fail", "fault", "not move", "doesn't move", "does not move", "unavailable", "troubleshoot")):
        return "troubleshooting"
    if procedure or any(term in lowered for term in ("what should i do", "how do i", "how to", "procedure", "steps", "handle this fault", "action required")):
        return "procedure"
    if any(term in lowered for term in ("compare", "difference", "versus", " vs ")):
        return "comparison"
    if "definition" in facets:
        return "definition"
    if "location" in facets:
        return "location"
    return "information"


def _visual_intent(text: str) -> bool:
    return bool(re.search(r"\b(show|figure|fig\.?|diagram|layout|panel|where.*located|location)\b", text, re.IGNORECASE))


def _unique(values: list[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", value.strip())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def _attribute_terms(facets: list[str], question: str) -> list[str]:
    terms: list[str] = []
    lowered = question.casefold()
    if "location" in facets:
        terms.extend(["location", "located", "address", "site", "place", "room", "office", "station", "depot"])
    if "contact" in facets:
        terms.extend(["contact", "phone", "mobile", "extension", "telephone"])
    if "constraint" in facets:
        for term in ("limit", "repayment", "duration", "validity", "frequency", "maximum", "minimum"):
            if term in lowered or term in {"maximum", "minimum"}:
                terms.append(term)
    if "enumeration" in facets:
        # Structural hints only. ``table`` already has a dedicated ranking bonus and broad
        # words such as ``summary``/``overview`` caused 0.6.1 to over-rank unrelated annexures.
        terms.extend(["number", "total", "name", "general information", "directory"])
    return _unique(terms, limit=16)


def build_query_plan(question: str) -> QueryPlan:
    original = question.strip()
    normalized = canonical_query_text(original)
    lookup = extract_lookup_term(normalized)
    source_scope = extract_source_scope(normalized)
    entities = extract_entity_terms(normalized)
    entity = entities[0] if len(entities) == 1 else None
    if len(entities) > 1 and lookup:
        lookup = None
    if not entities and lookup:
        entities = [lookup]
        entity = lookup
    if not entities and not lookup:
        # Preserve explicit short acronyms through later planning/recovery. This prevents
        # a fault question from understanding BIC/VCB/ATP initially and then losing the
        # component identity during recovery.
        acronyms = [
            token for token in re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", normalized)
            if token not in {"DMRC", "LINE"}
        ]
        if acronyms:
            entities = _unique(acronyms, limit=4)
            entity = entities[0] if len(entities) == 1 else None
    facets = query_facets(normalized)
    if lookup and "definition" not in facets:
        facets.append("definition")
    role = is_role_coverage_question(normalized)
    procedure = is_coverage_question(normalized)
    entity_attribute = is_entity_attribute_coverage_question(normalized)
    intent = _intent(normalized, role, procedure, facets)

    coverage_parts: list[str] = []
    if procedure:
        coverage_parts.append("procedure")
    if role:
        coverage_parts.append("role")
    if entity_attribute:
        coverage_parts.append("entity_attribute")
    coverage_kind = "+".join(coverage_parts) or None

    tokens = re.findall(r"[A-Za-z0-9_./-]+", normalized)
    line_match = _LINE_RE.search(normalized)
    stock_match = _STOCK_RE.search(normalized)
    line = f"Line {line_match.group(1)}" if line_match else None
    stock = f"RS-{stock_match.group(1)}" if stock_match else None

    # Organisation words are scope metadata, not mandatory content words.
    scope_terms: list[str] = []
    content = normalized
    for org in ("DMRC", "Delhi Metro Rail Corporation"):
        if re.search(rf"\b{re.escape(org)}\b", content, re.IGNORECASE):
            scope_terms.append(org)
            content = re.sub(rf"\b{re.escape(org)}\b", " ", content, flags=re.IGNORECASE)
    content = re.sub(r"\s+", " ", content).strip(" ,.-")

    content_terms: list[str] = []
    if entities:
        content_terms.extend(entities)
    elif procedure:
        topic = coverage_search_query(content or normalized)
        if topic:
            content_terms.append(topic)
    elif content:
        content_terms.append(content)

    semantic_queries: list[str] = []
    semantic_queries.extend(_operational_semantic_queries(normalized))
    if intent == "troubleshooting":
        semantic_queries.extend(_troubleshooting_queries(normalized))
        for entity_value in entities[:3]:
            if stock:
                semantic_queries.extend([
                    f"{stock} {entity_value} fault troubleshooting procedure",
                    f"{stock} {entity_value} isolation restriction action",
                ])
            elif line:
                semantic_queries.append(f"{line} {entity_value} fault troubleshooting procedure")
            else:
                semantic_queries.append(f"{entity_value} fault troubleshooting procedure")
    # Identifier variants are useful only when the question actually contains a technical
    # identifier. Do not turn ordinary numbers such as "between 2 stations" into malformed
    # semantic queries that consume the Real-Time query budget.
    if re.search(r"\b(?:SC|SI|TI|RS|ATP|ATO|UTO|TCMS|TIMS|PA|PIS)[\s_-]*[A-Za-z0-9./-]+\b", normalized, re.IGNORECASE):
        semantic_queries.extend(technical_identifier_variants(normalized))
    if content and content.casefold() != normalized.casefold():
        semantic_queries.extend(technical_identifier_variants(content))
    for entity_value in entities:
        semantic_queries.append(entity_value)
        # Attribute formulations are semantic hypotheses only; entity-only forms remain
        # present so terse tables without explicit attribute words are still retrievable.
        if "location" in facets:
            semantic_queries.append(f"{entity_value} location")
        if "contact" in facets:
            semantic_queries.append(f"{entity_value} contact details")
        if "constraint" in facets:
            semantic_queries.append(f"{entity_value} limit duration")
    semantic_queries = _unique(semantic_queries, limit=8)

    lexical_queries: list[str] = []
    for term in content_terms:
        lexical_queries.extend(lexical_form_variants(term, max_variants=6))
        lexical_queries.extend(technical_identifier_variants(term, max_variants=6))
    if not lexical_queries and content:
        lexical_queries.extend(lexical_form_variants(content, max_variants=4))
    lexical_queries = _unique(lexical_queries, limit=8)

    exact_terms: list[str] = []
    if entities:
        for entity_value in entities:
            exact_terms.extend(lexical_form_variants(entity_value, max_variants=6))
            exact_terms.extend(technical_identifier_variants(entity_value, max_variants=8))
    elif lookup:
        exact_terms.extend(technical_identifier_variants(lookup, max_variants=8))
    if line:
        exact_terms.append(line)
    if stock:
        exact_terms.extend(technical_identifier_variants(stock))
    exact_terms = _unique(exact_terms, limit=8)

    return QueryPlan(
        original=original,
        normalized=normalized,
        intent=intent,
        topic=(entity or " / ".join(entities) or content or normalized),
        content_terms=content_terms,
        scope_terms=scope_terms,
        source_scope=source_scope,
        line=line,
        rolling_stock=stock,
        answer_type=("structured_list" if "enumeration" in facets else "quick_reference"),
        lookup_term=lookup,
        entity_term=entity,
        entity_terms=entities,
        role_subject=extract_role_subject(normalized) if role else None,
        coverage_kind=coverage_kind,
        facets=facets,
        attribute_terms=_attribute_terms(facets, normalized),
        short_or_vague=len(tokens) <= 4,
        semantic_queries=semantic_queries,
        lexical_queries=lexical_queries,
        exact_terms=exact_terms,
        visual_intent=_visual_intent(normalized),
    )
