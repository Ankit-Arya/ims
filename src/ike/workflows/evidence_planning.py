from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from ike.retrieval.query_plan import QueryPlan

_ALLOWED_GOAL_KINDS = {
    "definition",
    "attribute",
    "enumeration",
    "procedure",
    "responsibility",
    "comparison",
    "relationship",
    "condition",
    "exception",
    "consequence",
    "applicability",
    "claim_check",
    "calculation",
    "temporal",
    "fact",
    "scenario",
    "overview",
}
_ALLOWED_COVERAGE_CONTRACTS = {
    "single_fact",
    "primary_definition",
    "all_supported_variants",
    "enumerate_set",
    "ordered_procedure",
    "all_requested_entities",
    "relationship_proof",
    "conditional_rule",
    "calculation_inputs",
    "compare_variants",
    "authority_proof",
    "scalar_with_condition",
}

_ALLOWED_STRATEGIES = {
    "single",
    "multi_lookup",
    "multi_entity",
    "comparison",
    "relationship",
    "conditional",
    "multi_hop",
    "claim_check",
    "enumeration",
    "procedure",
    "overview",
}


class SemanticEvidenceGoalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    question: str
    search_queries: list[str] = Field(default_factory=list, max_length=6)
    entity_terms: list[str] = Field(default_factory=list, max_length=12)
    required: bool = True
    depends_on: list[str] = Field(default_factory=list, max_length=6)
    qualifiers: list[str] = Field(default_factory=list, max_length=8)
    relation: str | None = None
    coverage_contract: str = "single_fact"


class SemanticEvidencePlanPayload(BaseModel):
    """Strict transport schema for LLM evidence planning.

    Domain-specific validity is still enforced by ``evidence_plan_from_payload``; structured
    output removes malformed-shape failures without granting the model factual authority.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "single"
    entities: list[str] = Field(default_factory=list, max_length=12)
    needs_research: bool = False
    needs_verification: bool = False
    goals: list[SemanticEvidenceGoalPayload] = Field(default_factory=list, max_length=12)


class GoalAuditRowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str
    status: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    reason: str = ""
    recovery_queries: list[str] = Field(default_factory=list, max_length=3)


class GoalAuditPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goals: list[GoalAuditRowPayload] = Field(default_factory=list, max_length=12)


_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9_./-]{1,15})(?![A-Za-z0-9])")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:[%°]|[A-Za-z]+)?(?![A-Za-z0-9])")
_RELATION_RE = re.compile(
    r"\b(?:relation(?:ship)?|relat(?:e|es|ed|ing)|interact(?:ion|s|ed|ing)?|connect(?:ed|s|ing)?\s+to|affect|effect\s+on|impact|cause|causes|caused|"
    r"lead(?:s|ing)?\s+to|depend(?:s|ence|ency)?|linked?|associated|between|override|supersede|"
    r"conflict|versus|\bvs\b|compare|comparison|difference|distinguish)\b",
    re.IGNORECASE,
)
_SCALAR_LOOKUP_RE = re.compile(
    r"\b(?:max(?:imum)?|min(?:imum)?|highest|lowest|how\s+many|how\s+much|rate|ratio|"
    r"capacity|dimension|dimensions|length|width|height|weight|value|amount|limit)\b",
    re.IGNORECASE,
)

_CONDITIONAL_RE = re.compile(
    r"\b(?:if|when|whenever|while|after|before|unless|except|provided\s+that|in\s+case\s+of|"
    r"assuming|suppose|what\s+if|then|otherwise)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(?:why|because|therefore|result(?:s|ed)?\s+in|consequence|consequences|due\s+to|"
    r"caus(?:e|es|ed|ing|al))\b",
    re.IGNORECASE,
)
_CLAIM_RE = re.compile(
    r"\b(?:is\s+it\s+(?:true|correct|right|wrong)|is\s+this\s+(?:true|correct|right|wrong)|"
    r"am\s+i\s+(?:right|wrong)|allowed\s+or\s+not|permitted\s+or\s+not|"
    r"correct\?|right\?|wrong\?|verify|confirm|whether)\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)
_BROAD_RE = re.compile(
    r"\b(?:everything\s+about|all\s+about|complete\s+(?:overview|details?)|comprehensive|"
    r"full\s+(?:overview|details?)|tell\s+me\s+everything)\b",
    re.IGNORECASE,
)
_CALC_RE = re.compile(
    r"\b(?:calculate|calculation|compute|total|sum|percentage|percent|how\s+much|how\s+long|"
    r"difference\s+between|remaining|balance)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(?:latest|current|currently|effective|valid\s+now|as\s+of|previous|old|older|newer|"
    r"revision|supersed(?:e|ed|es)|historical)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|except|unless|cannot|can't|doesn't|isn't|aren't)\b", re.IGNORECASE)
_MULTI_CLAUSE_RE = re.compile(
    r"(?:[;]|\band\s+then\b|\bthen\b|\bbut\b|\bhowever\b|\btherefore\b|"
    r"\band\s+(?:who|what|which|where|when|how)\b)",
    re.IGNORECASE,
)
_CONFIRMATION_SUFFIX_RE = re.compile(
    r"(?:[,;:]\s*|\s+)(?:correct|right|wrong|true|isn['’]?t\s+it|aren['’]?t\s+they)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_CONFLICT_RE = re.compile(
    r"\b(?:conflict(?:ing|s|ed)?|contradict(?:ion|ions|ory|s|ed)?|inconsisten(?:t|cy|cies)|"
    r"disagree(?:ment|ments|s|d)?|difference\s+between|different\s+(?:rules?|versions?|values?|limits?|instructions?)|"
    r"which\s+(?:one\s+)?applies|which\s+(?:one\s+)?prevails|takes?\s+precedence|"
    r"governing\s+version|authoritative\s+version)\b",
    re.IGNORECASE,
)

_POLICY_ENTITLEMENT_RE = re.compile(
    r"\b(?:entitl(?:e|ed|ement|ements)|eligib(?:le|ility)|allowance(?:s)?|reimburse(?:ment|ments|d|able)?|"
    r"claim(?:s|ing|ed)?|expenses?|benefits?|admissib(?:le|ility)|compensation|fare|lodging|hotel|"
    r"conveyance|refreshment|overtime|subsid(?:y|ies)|grants?|meal\s+(?:voucher|allowance)s?|"
    r"compensatory\s+(?:rest|leave)|daily\s+allowance|travel(?:ling)?\s+allowance|"
    r"pay\s+(?:scale|level)|designation)\b",
    re.IGNORECASE,
)
_POLICY_CONTEXT_RE = re.compile(
    r"\b(?:official|duty|tour|outstation|meeting|visit|journey|travel|posting|transfer|headquarters?|"
    r"holiday|sunday|weekend|overtime|extra\s+hours?|working\s+hours?|worked|shift|"
    r"another\s+(?:city|state|station)|away\s+from)\b",
    re.IGNORECASE,
)
_POLICY_TRAVEL_CONTEXT_RE = re.compile(
    r"\b(?:official|duty|tour|outstation|meeting|visit|journey|travel|headquarters?|"
    r"another\s+(?:city|state|station)|away\s+from)\b",
    re.IGNORECASE,
)
_POLICY_SUPPORT_RE = re.compile(
    r"\b(?:documents?|bills?|receipts?|proof|voucher(?:s)?|submit|submission|process|procedure|"
    r"how\s+to|what\s+.*\s+need)\b",
    re.IGNORECASE,
)
_CORPUS_NEGATIVE_RE = re.compile(
    r"\b(?:which|what|list|show|find)\s+(?:documents?|files?|manuals?|sources?)\b.*"
    r"\b(?:do\s+not|don['’]?t|does\s+not|doesn['’]?t|without|never)\b.*"
    r"\b(?:mention|contain|include|state|provide|refer)\w*\b|"
    r"\b(?:documents?|files?|manuals?|sources?)\b.*\bwithout\b.*\b(?:mention|reference|entry)\w*\b",
    re.IGNORECASE,
)
_SOURCE_SCOPE_RE = re.compile(
    r"\b(?:according\s+to|in|from|within|only\s+from)\s+(?:the\s+)?"
    r"(?:selected|this|that|specified|provided|uploaded)\s+(?:document|file|manual|source)\b",
    re.IGNORECASE,
)

_IDENTIFIER_STOP = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "DO", "FOR", "FROM", "HOW", "I",
    "IF", "IN", "IS", "IT", "NO", "NOT", "OF", "ON", "OR", "THE", "THEN", "TO", "WHAT",
    "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WITH", "YES",
}
_GENERIC_QUERY_WORDS = {
    "definition", "meaning", "function", "functions", "purpose", "location", "locations", "contact",
    "contacts", "relationship", "relation", "interaction", "compare", "comparison", "difference", "rule",
    "rules", "requirement", "requirements", "procedure", "procedures", "condition", "conditions", "exception",
    "exceptions", "consequence", "consequences", "applicability", "eligibility", "entitlement",
    "policy", "policies", "process", "processes", "instruction", "instructions", "responsibility", "duties",
    "overview", "details", "evidence", "document", "documents", "manual", "manuals", "current", "latest",
}


@dataclass(slots=True)
class EvidenceGoal:
    id: str
    kind: str
    question: str
    search_queries: list[str] = field(default_factory=list)
    entity_terms: list[str] = field(default_factory=list)
    required: bool = True
    depends_on: list[str] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)
    relation: str | None = None
    coverage_contract: str = "single_fact"

    def __post_init__(self) -> None:
        if self.coverage_contract == "single_fact":
            self.coverage_contract = _default_coverage_contract(self.kind)
        if self.coverage_contract not in _ALLOWED_COVERAGE_CONTRACTS:
            self.coverage_contract = _default_coverage_contract(self.kind)

    def prompt_block(self) -> str:
        entities = ", ".join(self.entity_terms) or "none explicitly extracted"
        dependencies = ", ".join(self.depends_on) or "none"
        qualifiers = "; ".join(self.qualifiers) or "none"
        return (
            f"{self.id}: kind={self.kind}; required={'yes' if self.required else 'no'}\n"
            f"Evidence question: {self.question}\n"
            f"Entity terms: {entities}\n"
            f"Relation: {self.relation or 'none'}\n"
            f"Dependencies: {dependencies}\n"
            f"Coverage contract: {self.coverage_contract}\n"
            f"Qualifiers/conditions: {qualifiers}"
        )


@dataclass(slots=True)
class EvidencePlan:
    original: str
    strategy: str = "single"
    entities: list[str] = field(default_factory=list)
    goals: list[EvidenceGoal] = field(default_factory=list)
    requires_decomposition: bool = False
    needs_research: bool = False
    needs_verification: bool = False
    planner_source: str = "deterministic"
    warnings: list[str] = field(default_factory=list)

    @property
    def required_goals(self) -> list[EvidenceGoal]:
        return [goal for goal in self.goals if goal.required]

    def prompt_block(self) -> str:
        goals = "\n\n".join(goal.prompt_block() for goal in self.goals) or "No explicit goals."
        return (
            f"Strategy: {self.strategy}\n"
            f"Planner source: {self.planner_source}\n"
            f"Requires decomposition: {'yes' if self.requires_decomposition else 'no'}\n"
            f"Entities: {', '.join(self.entities) or 'none explicitly extracted'}\n\n"
            f"Evidence goals:\n{goals}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GoalStatus:
    goal_id: str
    status: str  # supported | partial | missing | contradicted
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    recovery_queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoalSatisfaction:
    complete: bool
    statuses: list[GoalStatus] = field(default_factory=list)
    missing_goal_ids: list[str] = field(default_factory=list)
    partial_goal_ids: list[str] = field(default_factory=list)
    contradicted_goal_ids: list[str] = field(default_factory=list)
    audit_source: str = "deterministic"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self) -> str:
        lines: list[str] = []
        for status in self.statuses:
            ids = ", ".join(status.evidence_ids) or "none"
            lines.append(
                f"{status.goal_id}: {status.status}; evidence={ids}; reason={status.reason or 'not supplied'}"
            )
        return "\n".join(lines) or "No goal audit available."



def _default_coverage_contract(kind: str) -> str:
    kind = str(kind or "fact").casefold()
    if kind == "definition":
        # Conservative default for arbitrary definition goals. The deterministic planner
        # explicitly opts ordinary bounded term lookups into primary_definition.
        return "all_supported_variants"
    if kind in {"enumeration", "responsibility", "overview"}:
        return "enumerate_set"
    if kind == "procedure":
        return "ordered_procedure"
    if kind == "comparison":
        return "compare_variants"
    if kind == "relationship":
        return "relationship_proof"
    if kind == "authority":
        return "authority_proof"
    if kind in {"scalar", "attribute"}:
        return "scalar_with_condition"
    if kind in {"condition", "exception", "consequence", "applicability", "claim_check", "scenario"}:
        return "conditional_rule"
    if kind == "calculation":
        return "calculation_inputs"
    return "single_fact"

def _unique(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip()).strip(" ,;:.!?")
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


def _word_tokens(value: str) -> list[str]:
    """Unicode-aware word tokens for planner structure/sanitization.

    Python ``\\w`` splits some scripts at combining marks, so tokenize from Unicode
    categories instead. Technical identifier detection remains deliberately ASCII/
    uppercase, while ordinary user language can use any Unicode letters/marks/numbers.
    """

    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        token = "".join(current).strip("./-")
        if token:
            tokens.append(token)
        current.clear()

    for char in value:
        category = unicodedata.category(char)
        if category[:1] in {"L", "M", "N"} or char == "_":
            current.append(char)
            continue
        if char in "./-" and current:
            current.append(char)
            continue
        flush()
    flush()
    return tokens


def _content_tokens(value: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did",
        "for", "from", "how", "i", "if", "in", "is", "it", "of", "on", "or", "the",
        "then", "to", "what", "when", "where", "which", "who", "why", "with", "would",
        "could", "should", "can", "tell", "me", "find", "establish", "documentary", "evidence",
        "relevant", "this", "part", "users", "user", "request", "they", "them", "their", "there",
        "was", "were", "has", "have", "had", "been", "being", "both", "same", "also", "too",
        "came", "left", "sent", "provided", "provide", "provides", "needed", "need", "needs",
    }
    return {
        token.casefold()
        for token in _word_tokens(value)
        if token.casefold() not in stop and token.casefold() not in _GENERIC_QUERY_WORDS
    }


def _ordered_content_tokens(value: str) -> list[str]:
    allowed = _content_tokens(value)
    ordered: list[str] = []
    seen: set[str] = set()
    for token in _word_tokens(value):
        key = token.casefold()
        if key not in allowed or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _normalize_identifier_plurals(value: str) -> str:
    normalized = value
    for raw in _word_tokens(value):
        if len(raw) > 2 and raw.endswith("s") and raw[:-1].isupper():
            normalized = normalized.replace(raw, raw[:-1])
    return normalized


def standalone_identifiers(question: str) -> list[str]:
    """Return explicit technical-looking identifiers in user text without guessing expansions."""

    values: list[str] = []
    for raw in _word_tokens(question):
        if len(raw) > 2 and raw.endswith("s") and raw[:-1].isupper():
            token = raw[:-1]
            if token.upper() not in _IDENTIFIER_STOP and token not in values:
                values.append(token)
    for match in _IDENTIFIER_RE.finditer(question):
        token = match.group(1)
        pieces = (
            [part for part in token.split("/") if re.fullmatch(r"[A-Z][A-Z0-9_-]{1,15}", part)]
            if "/" in token
            else [token]
        )
        if not pieces:
            pieces = [token]
        for piece in pieces:
            if piece.upper() in _IDENTIFIER_STOP:
                continue
            if piece not in values:
                values.append(piece)
    return values[:12]


def _facet_goal_kind(facet: str) -> str:
    return {
        "definition": "definition",
        "enumeration": "enumeration",
        "location": "attribute",
        "contact": "attribute",
        "constraint": "attribute",
    }.get(facet, "fact")


def _goal_query(entity: str, facet: str) -> str:
    if facet == "definition":
        return entity
    if facet == "enumeration":
        return entity
    if facet == "location":
        return f"{entity} location"
    if facet == "contact":
        return f"{entity} contact"
    if facet == "constraint":
        return f"{entity} limits requirements"
    return entity


def _clause_fragments(question: str, *, limit: int = 6) -> list[str]:
    """Conservative fallback decomposition for long conditional text.

    This is deliberately grammatical. It does not know any domain noun, rule, station,
    benefit, equipment type, or expected answer.
    """

    text = re.sub(r"\s+", " ", question.strip())
    parts = re.split(
        r"\s*(?:;|\band\s+then\b|\bthen\b|\bbut\b|\bhowever\b|\btherefore\b|\bso\s+that\b)\s*",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = _unique((part for part in parts if len(part.strip()) >= 3), limit=limit)
    return cleaned if len(cleaned) >= 2 else [text]





def _multipart_units(question: str, *, limit: int = 8) -> list[str]:
    """Split explicit sibling asks while preserving setup/context clauses."""
    text = re.sub(r"\s+", " ", question.strip())

    # "Separately" is an explicit independence marker and must split even when the
    # following clause does not start with a wh-word.
    text = re.sub(r"\s*,?\s+and\s+separately\s+", " ; separately ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,?\s+separately\s+", " ; separately ", text, flags=re.IGNORECASE)

    parts = re.split(
        r"\s*;\s*|"
        r"\s*,\s*(?=(?:and\s+)?(?:what|who|which|where|when|how|can|should|does|do|is|are)\b)|"
        r"\s+and\s+(?=(?:what|who|which|where|when|how|can|should|does|do|is|are)\b)",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = _unique((part.strip(" ,.?") for part in parts if len(part.strip()) >= 4), limit=limit)
    return cleaned if len(cleaned) >= 2 else [text]


def _setup_context(unit: str) -> str | None:
    """Return an explicit setup/source/scenario clause that later asks should inherit."""
    text = re.sub(r"\s+", " ", unit.strip(" ,.?"))
    patterns = (
        r"^(?:if|when|whenever|while)\s+(.+)$",
        r"^(?:for|during|under|in\s+case\s+of|in\s+the\s+event\s+of)\s+(.+)$",
        r"^(?:as\s+per|according\s+to|from|under)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            context = match.group(1).strip(" ,.?")
            if context:
                return context[:140]
    return None


def _shared_context_anchor(question: str, units: list[str]) -> str | None:
    """Infer shared context from explicit setup plus conservative dependency cues."""
    if len(units) < 2 or re.search(r"\bseparately\b", question, re.IGNORECASE):
        return None

    first = units[0]
    setup = _setup_context(first)
    if setup:
        return setup

    match = re.search(
        r"\b(?:during|under|while|when|in\s+case\s+of|in\s+the\s+event\s+of)\s+"
        r"([^,;?.]{2,90})",
        first,
        re.IGNORECASE,
    )
    if match:
        anchor = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        anchor = re.split(
            r"\b(?:what|who|which|where|when|how|can|should)\b",
            anchor,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if anchor:
            return anchor[:120]

    # Later pronoun/generic follow-ups usually inherit the first ask's topic.
    dependent = 0
    for unit in units[1:]:
        lowered = unit.casefold().strip()
        lowered = re.sub(r"^(?:and|also)\s+", "", lowered)
        if re.match(
            r"^(?:who|when|can\b|should\b|what\s+(?:action|actions|checks|precautions|"
            r"restriction|restrictions|speed|records|refund|reissue|preconditions|steps|duties))",
            lowered,
        ):
            dependent += 1
        elif re.search(r"\b(?:it|they|this|that|the\s+train|the\s+staff|the\s+operator)\b", lowered):
            dependent += 1

    if dependent == len(units) - 1 and dependent:
        tokens = [
            token for token in _word_tokens(first)
            if token.casefold() not in _GENERIC_QUERY_WORDS
            and token.casefold() not in {
                "what","who","when","where","which","how","are","is","does","do","should",
                "can","the","a","an","of","for","to","on","or","and",
            }
        ]
        anchor = " ".join(tokens[:10]).strip()
        return anchor[:140] or None

    return None


def _multipart_goal_kind(fragment: str) -> str:
    lowered = fragment.casefold()
    if re.search(r"\bconditions?\b|\bwhen\b|\bunder\s+what\b", lowered):
        return "condition"
    if re.search(r"\bwho\b|\bresponsible\b|\bdut(?:y|ies)\b", lowered):
        return "responsibility"
    if re.search(
        r"\bactions?\b|\bwhat\s+(?:shall|should|must|does?|precautions?|checks?)\b|"
        r"\bprocedure\b|\bsteps?\b|\bprecautions?\b|\bchecks?\b",
        lowered,
    ):
        return "procedure"
    if re.search(r"\bspeed\b|\blimit\b|\brestrictions?\b|\bvalue\b|\brate\b", lowered):
        return "attribute"
    return "fact"


def _build_multipart_plan(question: str, query_plan: QueryPlan) -> EvidencePlan | None:
    units = _multipart_units(question)
    if len(units) < 2:
        return None

    explicit_independent = bool(re.search(r"\bseparately\b", question, re.IGNORECASE))
    anchor = None if explicit_independent else _shared_context_anchor(question, units)
    setup = _setup_context(units[0]) if anchor else None

    goal_units = list(units)
    if setup and len(units) > 1:
        # The first unit is a setup/constraint, not a standalone answer request.
        goal_units = units[1:]

    goals: list[EvidenceGoal] = []

    # A counted technical condition can only be mapped to a threshold if retrieval also
    # establishes what one counted item represents and the relevant formation/denominator.
    # Add this once as a prerequisite instead of polluting every requested outcome goal.
    # The probes remain generic and user-grounded; they do not guess the component's unit.
    setup_identifiers = _unique(standalone_identifiers(setup or ""))
    numeric_mapping_goal_id: str | None = None
    if (
        anchor
        and setup
        and _NUMBER_RE.search(setup)
        and query_plan.rolling_stock
        and len(setup_identifiers) >= 2
    ):
        combined_setup_entities = " ".join(setup_identifiers)
        counted_identifier = next(
            (
                item for item in setup_identifiers
                if re.sub(r"[^a-z0-9]", "", item.casefold())
                != re.sub(r"[^a-z0-9]", "", query_plan.rolling_stock.casefold())
            ),
            setup_identifiers[0],
        )
        numeric_mapping_goal_id = "g1"
        goals.append(
            EvidenceGoal(
                id=numeric_mapping_goal_id,
                kind="calculation",
                question=(
                    "Establish the documentary mapping and denominator needed to classify "
                    f"the counted technical items in {setup} against any count or percentage "
                    "threshold used by the applicable rule."
                ),
                search_queries=_unique(
                    [
                        f"{counted_identifier} {query_plan.rolling_stock} quantity per car",
                        f"{query_plan.rolling_stock} train formation cars",
                        combined_setup_entities,
                    ],
                    limit=3,
                ),
                entity_terms=setup_identifiers,
                required=True,
                qualifiers=[
                    f"shared_context:{anchor}",
                    f"inherited_context:{setup}",
                    "numeric_threshold_mapping",
                ],
                coverage_contract="calculation_inputs",
            )
        )

    first_outcome_index = 2 if numeric_mapping_goal_id else 1
    for index, raw_unit in enumerate(goal_units, start=first_outcome_index):
        unit = re.sub(r"^separately\s+", "", raw_unit, flags=re.IGNORECASE).strip()
        kind = _multipart_goal_kind(unit)
        target = unit
        if anchor and anchor.casefold() not in unit.casefold():
            target = f"{unit} in the shared context of {anchor}"
        goal_entities = _unique(standalone_identifiers(target))
        combined_entities = " ".join(goal_entities)
        compact_goal = " ".join(_ordered_content_tokens(target)[:10])
        coverage = (
            "ordered_procedure" if kind == "procedure"
            else "enumerate_set" if kind == "responsibility"
            else "conditional_rule" if kind == "condition"
            else "scalar_with_condition" if kind == "attribute"
            else "single_fact"
        )
        qualifiers = [f"shared_context:{anchor}"] if anchor else ["independent_subquery"]
        if setup:
            qualifiers.append(f"inherited_context:{setup}")
        goals.append(
            EvidenceGoal(
                id=f"g{index}",
                kind=kind,
                question=target,
                # Keep atomic goal retrieval isolated. The parent question is handled
                # separately by the retrieval engine when appropriate; attaching it to
                # every goal causes one broad formulation to be attributed to all goals.
                search_queries=_unique([_normalize_identifier_plurals(target), compact_goal, combined_entities], limit=3),
                # Keep entities local to the atomic ask. Global entity inheritance
                # makes identifiers from one independent subquery leak into every goal.
                entity_terms=goal_entities,
                required=True,
                depends_on=[numeric_mapping_goal_id] if numeric_mapping_goal_id else [],
                qualifiers=qualifiers,
                coverage_contract=coverage,
            )
        )

    if not goals:
        return None

    return EvidencePlan(
        original=question,
        strategy="multi_hop" if anchor else "multi_lookup",
        entities=_unique(query_plan.entity_terms),
        goals=goals,
        requires_decomposition=True,
        needs_research=True,
        needs_verification=True,
        warnings=[
            f"multipart_shared_context:{anchor}" if anchor else "multipart_independent_subqueries"
        ],
    )


def _build_policy_entitlement_plan(question: str, query_plan: QueryPlan) -> EvidencePlan | None:
    """Plan institutional eligibility/benefit/expense questions by information need, not domain."""

    if not _POLICY_ENTITLEMENT_RE.search(question):
        return None

    tokens = _word_tokens(question)
    explicit_enumeration = bool(
        "enumeration" in (query_plan.facets or [])
        or re.search(r"\b(?:what\s+all|all\s+(?:types?|kinds?|claims?|allowances?|benefits?)|types?|list)\b", question, re.IGNORECASE)
    )
    has_context = bool(_POLICY_CONTEXT_RE.search(question) or _CONDITIONAL_RE.search(question))
    travel_context = bool(_POLICY_TRAVEL_CONTEXT_RE.search(question))
    has_amount = bool(
        _SCALAR_LOOKUP_RE.search(question)
        or _CALC_RE.search(question)
        or re.search(r"\b(?:pay\s+(?:scales?|levels?)|designations?|grades?|amounts?|rates?|limits?)\b", question, re.IGNORECASE)
    )
    has_support = bool(
        query_plan.intent == "procedure"
        or _POLICY_SUPPORT_RE.search(question)
        or re.search(r"\b(?:need\s+to\s+claim|claim\s+bills?|file\s+a\s+claim)\b", question, re.IGNORECASE)
    )
    has_calculation = bool(
        _CALC_RE.search(question)
        or (
            re.search(r"\btotal\b", question, re.IGNORECASE)
            and bool(_NUMBER_RE.search(question))
        )
    )

    # Do not turn a short factual mention of an allowance/benefit into a large plan.
    if not any((explicit_enumeration, has_context, has_amount, has_support, has_calculation)) and len(tokens) < 10:
        return None

    core_tokens = [
        token for token in _ordered_content_tokens(question)
        if token not in {"different", "all", "types", "type", "amount", "amounts"}
    ][:14]
    core = " ".join(core_tokens) or question

    # Build a compact, stable policy probe from the user's own policy nouns and explicit
    # conditions. Long conversational requests often contain locations, dates and filler
    # that dilute lexical retrieval; this probe keeps the institutional concept while the
    # original wording remains available as a separate retrieval hypothesis.
    lowered = question.casefold()
    policy_heads: list[str] = []
    for pattern, canonical in (
        (r"\ballowances?\b", "allowance"),
        (r"\breimburse(?:ment|ments|d|able)?\b", "reimbursement"),
        (r"\bclaims?\b|\bclaiming\b", "claim"),
        (r"\bbenefits?\b", "benefit"),
        (r"\bcompensation\b", "compensation"),
        (r"\bconveyance\b", "conveyance"),
        (r"\brefreshment\b", "refreshment"),
        (r"\bcompensatory\s+(?:rest|leave)\b", "compensatory rest"),
        (r"\bovertime\b", "overtime"),
        (r"\bgrants?\b", "grant"),
        (r"\bsubsid(?:y|ies)\b", "subsidy"),
        (r"\bhotel\b|\blodging\b", "hotel"),
        (r"\bdaily\s+allowance\b", "daily allowance"),
        (r"\btravel(?:ling)?\b|\bjourney\b|\btour\b", "travel"),
        (r"\btransfer(?:red|ring)?\b", "transfer"),
        (r"\bctg\b", "CTG"),
    ):
        if re.search(pattern, lowered, re.IGNORECASE) and canonical not in policy_heads:
            policy_heads.append(canonical)
    condition_heads = [
        token
        for token in _ordered_content_tokens(question)
        if token in {
            "official", "tour", "outstation", "meeting", "transfer", "retire", "retirement",
            "holiday", "sunday", "weekend", "overtime", "shift", "worked",
            "hotel", "travel", "taxi", "receipt", "receipts", "manager", "designation", "grade",
            "request",
        }
    ]
    if re.search(r"\bown\s+request\b|\bself[- ]request\b", lowered):
        condition_heads.append("own request")
    compact_policy = " ".join(_unique([*policy_heads, *condition_heads], limit=8)).strip()
    entities = _unique(query_plan.entity_terms)
    goals: list[EvidenceGoal] = []

    def add_goal(kind: str, goal_question: str, probes: list[str], contract: str, *, depends_on: list[str] | None = None) -> None:
        goals.append(
            EvidenceGoal(
                id=f"g{len(goals) + 1}",
                kind=kind,
                question=goal_question,
                search_queries=_unique(probes, limit=3),
                entity_terms=entities,
                required=True,
                depends_on=list(depends_on or []),
                coverage_contract=contract,
            )
        )

    if has_context:
        add_goal(
            "applicability",
            "Establish the eligibility and applicability conditions for the requested entitlement or reimbursement in the user's stated context.",
            [
                (
                    "official tour travel entitlement applicability"
                    if travel_context
                    else f"{compact_policy} eligibility applicability"
                    if compact_policy
                    else f"{core} policy eligibility"
                ),
                f"{core} eligibility applicability",
                f"{core} official duty entitlement",
            ],
            "conditional_rule",
        )

    if explicit_enumeration:
        add_goal(
            "enumeration",
            "Identify the documented categories that satisfy the user's requested set; do not substitute similarly named but differently scoped benefits.",
            [
                (
                    "official tour travel lodging daily allowance local conveyance"
                    if travel_context
                    else f"{compact_policy} categories rates pay scale"
                    if compact_policy and has_amount
                    else f"{compact_policy} categories entitlement"
                    if compact_policy
                    else f"{core} benefit reimbursement categories"
                ),
                f"{core} categories types",
                f"{core} entitlement reimbursement",
            ],
            "enumerate_set",
            depends_on=["g1"] if goals else [],
        )

    if has_amount:
        add_goal(
            "attribute",
            "Establish the applicable rates, limits, amount basis, grade/designation distinctions, and conditions requested by the user.",
            [
                (
                    "official tour daily allowance hotel lodging rates designation pay scale"
                    if travel_context
                    else f"{compact_policy} rates limits pay scale designation"
                    if compact_policy
                    else f"{core} rates ceilings grade"
                ),
                f"{core} rates limits entitlement",
                f"{core} pay scale designation grade",
            ],
            "all_requested_entities",
            depends_on=[goal.id for goal in goals if goal.kind in {"applicability", "enumeration"}],
        )

    if has_support:
        add_goal(
            "procedure",
            "Establish the documented claim/submission process and any required bills, receipts, approvals, or supporting proof.",
            [
                (
                    "official tour claim bills receipts approval supporting documents"
                    if travel_context
                    else f"{compact_policy} claim receipts approval supporting documents"
                    if compact_policy
                    else f"{core} submission supporting documents"
                ),
                f"{core} claim process documents",
                f"{core} bills receipts approval",
            ],
            "ordered_procedure",
            depends_on=[goal.id for goal in goals if goal.kind == "applicability"],
        )

    if has_calculation:
        add_goal(
            "calculation",
            "Establish every documentary input and rule needed to calculate the requested total; distinguish policy limits from the user's actual expenses.",
            [
                (
                    "official tour daily allowance hotel duration fare calculation"
                    if travel_context
                    else f"{compact_policy} rates duration calculation"
                    if compact_policy
                    else f"{core} calculation inputs rates duration"
                ),
                f"{core} calculation total rate",
                f"{core} daily allowance duration amount",
            ],
            "calculation_inputs",
            depends_on=[goal.id for goal in goals if goal.kind in {"applicability", "attribute"}],
        )

    if not goals:
        return None

    complex_shape = len(goals) >= 3 or (has_calculation and len(tokens) >= 14)
    strategy = (
        "multi_hop" if has_calculation or has_support
        else "enumeration" if explicit_enumeration
        else "conditional" if has_context
        else "single"
    )
    warnings = ["policy_entitlement_plan"]
    if complex_shape:
        warnings.append("semantic_repair_recommended")

    return EvidencePlan(
        original=question,
        strategy=strategy,
        entities=entities,
        goals=goals,
        requires_decomposition=len(goals) > 1,
        needs_research=len(goals) > 1 or query_plan.coverage_sensitive,
        needs_verification=True,
        warnings=warnings,
    )


def build_deterministic_evidence_plan(question: str, query_plan: QueryPlan) -> EvidencePlan:
    """Build a safe evidence-requirement plan without domain-specific assumptions.

    The deterministic path handles common structural forms and acts as a non-LLM fallback.
    It intentionally does not expand acronyms, infer policies, or guess relations.
    """

    original = re.sub(r"\s+", " ", (query_plan.normalized or question).strip())

    policy_plan = _build_policy_entitlement_plan(original, query_plan)
    if policy_plan is not None:
        return policy_plan

    # Attribute/limit questions with explicit applicability constraints must remain one
    # constrained evidence goal. Do this before acronym-definition shortcuts.
    if _SCALAR_LOOKUP_RE.search(original) and re.search(r"\b(?:speed|limit|rate|value|amount|maximum|minimum|max|min)\b", original, re.IGNORECASE):
        qualifiers = []
        for label, value in (
            ("source", query_plan.source_scope),
            ("line", query_plan.line),
            ("rolling_stock", query_plan.rolling_stock),
            ("system", query_plan.system),
            ("subsystem", query_plan.subsystem),
            ("scenario", query_plan.scenario),
        ):
            if value:
                qualifiers.append(f"{label}:{value}")
        return EvidencePlan(
            original=original,
            strategy="conditional" if qualifiers else "single",
            entities=_unique(query_plan.entity_terms),
            goals=[EvidenceGoal(
                id="g1",
                kind="attribute",
                question=original,
                search_queries=_unique([original, *query_plan.semantic_queries, *query_plan.lexical_queries], limit=5),
                entity_terms=_unique(query_plan.entity_terms),
                required=True,
                qualifiers=qualifiers,
                coverage_contract="scalar_with_condition",
            )],
            requires_decomposition=False,
            needs_research=bool(qualifiers),
            needs_verification=True,
            warnings=["constrained_attribute_query"] if qualifiers else [],
        )

    identifiers = standalone_identifiers(original)
    if query_plan.source_scope:
        scope_compact = re.sub(r"[^a-z0-9]", "", query_plan.source_scope.casefold())
        identifiers = [i for i in identifiers if re.sub(r"[^a-z0-9]", "", i.casefold()) not in scope_compact]
    relation_requested = bool(_RELATION_RE.search(original))
    conditional = bool(_CONDITIONAL_RE.search(original))
    causal = bool(_CAUSAL_RE.search(original))
    claim_check = bool(
        _CLAIM_RE.search(original)
        or _YES_NO_RE.search(original)
        or _CONFIRMATION_SUFFIX_RE.search(original)
    )
    conflict = bool(_CONFLICT_RE.search(original))
    corpus_negative = bool(_CORPUS_NEGATIVE_RE.search(original))
    source_scoped = bool(_SOURCE_SCOPE_RE.search(original))
    broad = bool(_BROAD_RE.search(original))
    calculation = bool(_CALC_RE.search(original))
    temporal = bool(_TEMPORAL_RE.search(original))
    multi_clause = bool(_MULTI_CLAUSE_RE.search(original))

    entities = _unique(query_plan.entity_terms or ([] if query_plan.entity_term is None else [query_plan.entity_term]))
    facets = list(query_plan.facets or [])
    # Short juxtaposed identifiers are separate information units even when the legacy
    # query parser sees the phrase as one topic. This covers arbitrary A/B/C-style inputs.
    if len(identifiers) >= 2:
        entities = identifiers

    goals: list[EvidenceGoal] = []

    # Explicit sibling sub-questions must be interpreted before legacy facet shortcuts.
    # This distinguishes one shared scenario with several requested facets from genuinely
    # independent questions, and prevents entity extraction (for example TO/SC) from
    # hijacking the whole request into unrelated authority goals.
    multipart_plan = _build_multipart_plan(original, query_plan)
    if multipart_plan is not None:
        return multipart_plan

    # Keep explicit requested aspects independent. This prevents a compound question
    # from being collapsed into one generic definition/fact goal.
    if "authority" in facets:
        aspect_entities = entities or ([query_plan.lookup_term] if query_plan.lookup_term else [])
        if aspect_entities:
            for entity in aspect_entities[:4]:
                if "definition" in facets:
                    goals.append(
                        EvidenceGoal(
                            id=f"g{len(goals)+1}",
                            kind="definition",
                            question=f"Establish the documentary definition of {entity}.",
                            search_queries=[entity, f"{entity} definition"],
                            entity_terms=[entity],
                            required=True,
                            qualifiers=["definition"],
                            coverage_contract="primary_definition",
                        )
                    )
                goals.append(
                    EvidenceGoal(
                        id=f"g{len(goals)+1}",
                        kind="authority",
                        question=f"Establish who imposes, authorizes, approves or is responsible for {entity}.",
                        search_queries=[
                            f"{entity} imposed by",
                            f"{entity} imposing authority",
                            f"{entity} approval authority",
                            entity,
                        ],
                        entity_terms=[entity],
                        required=True,
                        qualifiers=["authority"],
                        coverage_contract="authority_proof",
                    )
                )
            return EvidencePlan(
                original=original,
                strategy="multi_hop",
                entities=aspect_entities,
                goals=goals,
                requires_decomposition=len(goals) > 1,
                needs_research=False,
                needs_verification=True,
                warnings=["deterministic_multi_aspect"],
            )

    # A negative inventory is categorically different from ordinary retrieval: absence
    # from top-k evidence never proves absence from a corpus. Mark it explicitly so the
    # answer/verification stages fail safely unless a future dedicated inventory operation
    # supplies exhaustive evidence. This is generic across document types and domains.
    if corpus_negative:
        return EvidencePlan(
            original=original,
            strategy="overview",
            entities=entities,
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="claim_check",
                    question=(
                        "Establish whether the requested corpus-wide negative inventory can be "
                        "proven from exhaustive accessible-document evidence; do not infer non-mention from top-k retrieval."
                    ),
                    search_queries=[original],
                    entity_terms=entities,
                    required=True,
                    qualifiers=["corpus_negative_requires_exhaustive_inventory"],
                    relation="corpus_absence",
                )
            ],
            requires_decomposition=True,
            needs_research=True,
            needs_verification=True,
            warnings=["corpus_negative_requires_exhaustive_inventory"],
        )

    # Operational abnormalities take precedence over generic conditional/causal syntax.
    # A phrase such as "what shall TO do if announcement is wrong" is a troubleshooting
    # request, not a claim-check or abstract conditional-analysis request.
    if query_plan.intent == "troubleshooting":
        topic = query_plan.content_terms[0] if query_plan.content_terms else original
        search_queries = _unique([*query_plan.semantic_queries, topic, original], limit=6)
        return EvidencePlan(
            original=original,
            strategy="procedure",
            entities=_unique(query_plan.entity_terms),
            goals=[EvidenceGoal(
                id="g1",
                kind="procedure",
                question=(
                    "Establish the applicable troubleshooting/corrective procedure, manual or override controls, "
                    f"reporting steps, and role-specific actions for: {topic}"
                ),
                search_queries=search_queries,
                entity_terms=_unique(query_plan.entity_terms),
                required=True,
                qualifiers=["operational_abnormality"],
                coverage_contract="ordered_procedure",
            )],
            requires_decomposition=False,
            needs_research=True,
            needs_verification=True,
            warnings=["operational_abnormality_treated_as_troubleshooting"],
        )

    if query_plan.intent == "responsibilities" and identifiers:
        role_goals = []
        for index, identifier in enumerate(identifiers[:8], start=1):
            role_goals.append(
                EvidenceGoal(
                    id=f"g{index}",
                    kind="responsibility",
                    question=f"Establish the duties/responsibilities of {identifier} in the requested scenario.",
                    search_queries=[f"{identifier} duties responsibilities", original],
                    entity_terms=[identifier],
                    required=True,
                    coverage_contract="enumerate_set",
                )
            )
        return EvidencePlan(
            original=original,
            strategy="overview",
            entities=identifiers[:8],
            goals=role_goals,
            requires_decomposition=len(role_goals) > 1,
            needs_research=True,
            needs_verification=True,
        )

    if len(identifiers) >= 2:
        # Establish each identifier independently. Definitions are required for a bare
        # multi-identifier lookup and supporting (optional) for an explicit relation/comparison.
        identifier_scenario = conditional or causal or claim_check or conflict or multi_clause or calculation or temporal
        token_count = len(_word_tokens(original))
        definition_requested = bool(
            query_plan.intent == "definition"
            or "definition" in facets
            or query_plan.lookup_term
            or (token_count <= 4 and not identifier_scenario and not relation_requested)
        )
        definitions_required = definition_requested and not relation_requested and not identifier_scenario
        for index, identifier in enumerate(identifiers, start=1):
            goal_kind = "definition" if definition_requested else "fact"
            goals.append(
                EvidenceGoal(
                    id=f"g{index}",
                    kind=goal_kind,
                    question=(
                        f"Establish the documentary meaning or function of {identifier}."
                        if definition_requested
                        else f"Establish the requested documentary facts for {identifier}: {original}"
                    ),
                    search_queries=(
                        [identifier]
                        if definition_requested
                        else [f"{identifier} {original}", identifier]
                    ),
                    entity_terms=[identifier],
                    required=(definitions_required if definition_requested else True),
                    coverage_contract=("all_supported_variants" if definition_requested else "single_fact"),
                )
            )
        if relation_requested:
            relation_queries: list[str] = []
            for left_index, left in enumerate(identifiers):
                for right in identifiers[left_index + 1 :]:
                    relation_queries.append(f"{left} {right}")
            relation_queries.append(" ".join(identifiers))
            goals.append(
                EvidenceGoal(
                    id=f"g{len(goals) + 1}",
                    kind="comparison" if re.search(r"\b(?:compare|comparison|difference|versus|\bvs\b)\b", original, re.IGNORECASE) else "relationship",
                    question=f"Establish only relationships among {', '.join(identifiers)} that are explicitly supported by the documents.",
                    search_queries=_unique(relation_queries, limit=8),
                    entity_terms=identifiers,
                    required=True,
                    relation="user_requested_relationship",
                    depends_on=[goal.id for goal in goals],
                )
            )
        if identifier_scenario:
            scenario_kind = (
                "claim_check" if claim_check else
                "relationship" if conflict else
                "temporal" if temporal else
                "calculation" if calculation else
                "condition" if conditional else
                "consequence" if causal else
                "scenario"
            )
            for fragment in _clause_fragments(original, limit=4):
                goals.append(
                    EvidenceGoal(
                        id=f"g{len(goals) + 1}",
                        kind=scenario_kind,
                        question=f"Establish the documentary rule/fact needed for this part of the request: {fragment}",
                        search_queries=[fragment, " ".join(identifiers)],
                        entity_terms=identifiers,
                        required=True,
                        depends_on=[goal.id for goal in goals if goal.kind == "definition"],
                        relation="conditional_or_contextual_use",
                    )
                )
        strategy = (
            "claim_check" if claim_check else
            "conditional" if conditional else
            "relationship" if conflict else
            "multi_hop" if identifier_scenario else
            "comparison" if any(goal.kind == "comparison" for goal in goals) else
            "relationship" if relation_requested else
            "multi_lookup"
        )
        return EvidencePlan(
            original=original,
            strategy=strategy,
            entities=entities,
            goals=goals,
            requires_decomposition=True,
            needs_research=True,
            needs_verification=True,
        )

    # If a sentence contains one explicit technical identifier, preserve an independent
    # exact/lookup path for that literal token even when a legacy phrase extractor wrapped
    # it in descriptive words. The original phrase is retained as a secondary query.
    structural_complex = bool(
        relation_requested
        or conditional
        or causal
        or claim_check
        or conflict
        or multi_clause
        or calculation
        or temporal
        or broad
    )
    if structural_complex and len(identifiers) == 1:
        # Preserve the literal technical identifier instead of a legacy extractor's larger
        # conversational phrase (for example ``CODE and why ...``). The full wording still
        # appears in the structural goal and semantic planner prompt.
        entities = [identifiers[0]]

    # Structurally complex ordinary-language questions must be planned before legacy
    # entity/facet shortcuts. Otherwise a phrase like "different limits ... which applies"
    # can be reduced to an attribute lookup and lose the conflict/precedence obligation.
    # Seed obvious facets/clauses from user wording, then let the semantic planner refine.
    if structural_complex:
        goal_id = 1
        for entity in entities:
            for facet in facets:
                goals.append(
                    EvidenceGoal(
                        id=f"g{goal_id}",
                        kind=_facet_goal_kind(facet),
                        question=(
                            f"Establish the {facet} requested for {entity}."
                            if facet != "enumeration"
                            else f"Identify the requested set/list of {entity}."
                        ),
                        search_queries=[_goal_query(entity, facet), entity],
                        entity_terms=[entity],
                        required=True,
                        qualifiers=[facet],
                        coverage_contract=("primary_definition" if facet == "definition" else "single_fact"),
                    )
                )
                goal_id += 1

        structural_kind = (
            "claim_check" if claim_check else
            "comparison" if query_plan.intent == "comparison" else
            "relationship" if relation_requested or conflict else
            "calculation" if calculation else
            "temporal" if temporal else
            "overview" if broad else
            "condition" if conditional else
            "consequence" if causal else
            "scenario"
        )
        for fragment in _clause_fragments(original):
            fragment_tokens = _content_tokens(fragment)
            # Avoid a duplicate whole-clause goal when a single explicit facet seed already
            # covers the same short wording; relation/claim/condition goals are always kept.
            if structural_kind in {"relationship", "comparison", "claim_check", "condition", "consequence", "calculation", "temporal"} or not goals:
                goals.append(
                    EvidenceGoal(
                        id=f"g{goal_id}",
                        kind=structural_kind if len(_clause_fragments(original)) == 1 else (
                            "scenario" if fragment != _clause_fragments(original)[-1] else structural_kind
                        ),
                        question=f"Find documentary evidence relevant to this part of the user's request: {fragment}",
                        search_queries=[fragment],
                        entity_terms=entities,
                        required=True,
                        depends_on=([f"g{goal_id - 1}"] if goal_id > 1 and fragment_tokens else []),
                        qualifiers=(
                            ["conflict_or_precedence"] if conflict else
                            ["user_premise_requires_verification"] if claim_check else
                            []
                        ),
                        relation=(
                            "conflict_or_precedence" if conflict else
                            "user_requested_relationship" if relation_requested else
                            "conditional_or_contextual_use" if conditional or causal else
                            None
                        ),
                    )
                )
                goal_id += 1

        strategy = (
            "claim_check" if claim_check else
            "comparison" if query_plan.intent == "comparison" else
            "relationship" if relation_requested or conflict else
            "overview" if broad else
            "conditional" if conditional else
            "multi_hop" if causal or multi_clause else
            "single"
        )
        return EvidencePlan(
            original=original,
            strategy=strategy,
            entities=entities,
            goals=goals or [
                EvidenceGoal(
                    id="g1",
                    kind=structural_kind,
                    question=original,
                    search_queries=[original],
                    entity_terms=entities,
                    required=True,
                )
            ],
            requires_decomposition=True,
            needs_research=True,
            needs_verification=True,
        )

    if len(identifiers) == 1 and (query_plan.lookup_term or query_plan.intent == "definition"):
        identifier = identifiers[0]
        contextual = query_plan.lookup_term or (entities[0] if entities else original)
        return EvidencePlan(
            original=original,
            strategy="single",
            entities=[identifier],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="definition",
                    question=f"Establish the documentary meaning or function of {identifier} in the user's stated context.",
                    search_queries=_unique([identifier, contextual, original], limit=3),
                    entity_terms=[identifier],
                    required=True,
                    coverage_contract="primary_definition",
                )
            ],
            requires_decomposition=False,
            needs_research=False,
            needs_verification=False,
        )

    if entities and facets:
        goal_id = 1
        for entity in entities:
            for facet in facets:
                goals.append(
                    EvidenceGoal(
                        id=f"g{goal_id}",
                        kind=_facet_goal_kind(facet),
                        question=(
                            f"Establish the {facet} requested for {entity}."
                            if facet != "enumeration"
                            else f"Identify the requested set/list of {entity}."
                        ),
                        search_queries=[_goal_query(entity, facet), entity],
                        entity_terms=[entity],
                        required=True,
                        qualifiers=[facet],
                        coverage_contract=("primary_definition" if facet == "definition" else "single_fact"),
                    )
                )
                goal_id += 1
        return EvidencePlan(
            original=original,
            strategy="enumeration" if "enumeration" in facets else ("multi_entity" if len(goals) > 1 else "single"),
            entities=entities,
            goals=goals,
            requires_decomposition=len(goals) > 1,
            needs_research=(len(goals) > 1 or query_plan.coverage_sensitive),
            needs_verification=(len(goals) > 1 or query_plan.coverage_sensitive),
        )

    if "enumeration" in facets and not entities:
        return EvidencePlan(
            original=original,
            strategy="enumeration",
            entities=[],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="enumeration",
                    question=f"Identify the requested documentary set/list: {original}",
                    search_queries=_unique([original, *query_plan.semantic_queries, *query_plan.lexical_queries], limit=5),
                    entity_terms=[],
                    required=True,
                    coverage_contract="enumerate_set",
                )
            ],
            requires_decomposition=False,
            needs_research=True,
            needs_verification=True,
        )

    if "constraint" in facets and not entities and query_plan.intent == "information":
        return EvidencePlan(
            original=original,
            strategy="single",
            entities=[],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="attribute",
                    question=f"Establish the requested numeric/time/rate/limit fact: {original}",
                    search_queries=_unique([original, *query_plan.semantic_queries, *query_plan.lexical_queries], limit=4),
                    entity_terms=[],
                    required=True,
                    coverage_contract="single_fact",
                )
            ],
            requires_decomposition=False,
            needs_research=False,
            needs_verification=False,
        )

    if query_plan.lookup_term:
        term = query_plan.lookup_term
        return EvidencePlan(
            original=original,
            strategy="single",
            entities=[term],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="definition",
                    question=f"Establish the documentary meaning or function of {term}.",
                    search_queries=[term],
                    entity_terms=[term],
                    required=True,
                    coverage_contract="primary_definition",
                )
            ],
            requires_decomposition=False,
            needs_research=False,
            needs_verification=False,
        )

    if query_plan.role_subject:
        subject = query_plan.role_subject
        return EvidencePlan(
            original=original,
            strategy="overview",
            entities=[subject],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="responsibility",
                    question=f"Establish the duties/responsibilities of {subject} across applicable evidence.",
                    search_queries=[subject, f"{subject} duties responsibilities"],
                    entity_terms=[subject],
                )
            ],
            requires_decomposition=False,
            needs_research=True,
            needs_verification=True,
        )

    if "procedure" in (query_plan.coverage_kind or "") or query_plan.intent == "procedure":
        topic = query_plan.content_terms[0] if query_plan.content_terms else original
        return EvidencePlan(
            original=original,
            strategy="procedure",
            entities=_unique(query_plan.entity_terms),
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="procedure",
                    question=f"Establish the applicable procedure/requirements for: {topic}",
                    search_queries=[topic, original],
                    entity_terms=_unique(query_plan.entity_terms),
                )
            ],
            requires_decomposition=False,
            needs_research=query_plan.coverage_sensitive,
            needs_verification=query_plan.coverage_sensitive,
        )

    # A bare single token is normally a request to identify/explain that term, even when
    # the user typed it in lowercase. Treat it as a term goal without assuming it is an
    # acronym or inventing an expansion.
    bare_tokens = _word_tokens(original)
    if len(bare_tokens) == 1 and 2 <= len(bare_tokens[0]) <= 40:
        term = bare_tokens[0]
        return EvidencePlan(
            original=original,
            strategy="single",
            entities=[term],
            goals=[
                EvidenceGoal(
                    id="g1",
                    kind="definition",
                    question=f"Establish the documentary meaning, function or identified concept for {term}.",
                    search_queries=[term],
                    entity_terms=[term],
                    required=True,
                    coverage_contract="primary_definition",
                )
            ],
            requires_decomposition=False,
            needs_research=False,
            needs_verification=False,
        )

    # Final conservative fallback. It preserves the existing low-latency behavior for
    # ordinary one-part information questions while still giving downstream code one goal.
    return EvidencePlan(
        original=original,
        strategy="single",
        entities=entities,
        goals=[
            EvidenceGoal(
                id="g1",
                kind="fact",
                question=original,
                search_queries=[original],
                entity_terms=entities,
            )
        ],
        requires_decomposition=False,
        needs_research=query_plan.coverage_sensitive,
        needs_verification=query_plan.coverage_sensitive,
    )


def should_use_semantic_planner(question: str, query_plan: QueryPlan, plan: EvidencePlan) -> bool:
    """Use the LLM only when syntax alone cannot safely express the information need."""

    # Operational abnormalities already have a deterministic troubleshooting plan with
    # bounded domain vocabulary. Do not let generic words such as "if" or "when" reopen
    # semantic planning and turn the fault report back into a claim/condition analysis.
    if query_plan.intent == "troubleshooting":
        return False

    # A bounded definition lookup is a structural retrieval problem, including formulations
    # such as "definition of Incident as per MRGR".  Source scope is handled separately by
    # retrieval and must not trigger an LLM planning detour.
    if query_plan.intent == "definition" and query_plan.lookup_term:
        return False

    identifiers = standalone_identifiers(question)
    structurally_complex = bool(
        _CONDITIONAL_RE.search(question)
        or _CAUSAL_RE.search(question)
        or _CLAIM_RE.search(question)
        or _CONFIRMATION_SUFFIX_RE.search(question)
        or _YES_NO_RE.search(question)
        or _CONFLICT_RE.search(question)
        or _CORPUS_NEGATIVE_RE.search(question)
        or _MULTI_CLAUSE_RE.search(question)
    )
    if len(identifiers) >= 2 and not structurally_complex:
        # Bare multi-identifier lookup and short explicit comparison/relation requests have
        # a safer deterministic atomic plan. Rich conditional uses still go semantic.
        token_count = len(_word_tokens(question))
        if plan.strategy == "multi_lookup" or token_count <= 8:
            return False
    if plan.strategy in {"conditional", "multi_hop", "claim_check", "comparison", "relationship"}:
        return True
    if (
        _CONDITIONAL_RE.search(question)
        or _CAUSAL_RE.search(question)
        or _CLAIM_RE.search(question)
        or _CONFIRMATION_SUFFIX_RE.search(question)
        or _YES_NO_RE.search(question)
        or _CONFLICT_RE.search(question)
        or _CORPUS_NEGATIVE_RE.search(question)
        or _SOURCE_SCOPE_RE.search(question)
    ):
        return True
    if len(question) > 180 or len(_clause_fragments(question)) >= 3:
        return True
    token_count = len(_word_tokens(question))
    # Straight scalar/attribute lookups should not pay an LLM-planning tax merely because
    # the user used ordinary language. Hybrid + table/fact + corpus-vocabulary retrieval
    # can start immediately, and evidence-quality gates may escalate if needed.
    if _SCALAR_LOOKUP_RE.search(question) and plan.strategy == "single":
        return False

    # Explicit procedure questions can still use the semantic planner for richer
    # condition/role decomposition; troubleshooting returned on the fast path above.
    if query_plan.intent == "procedure":
        return token_count >= 2

    # Ambiguous free-form information requests with no resolved entity benefit from semantic
    # planning, especially across departments and languages. Scalar lookups returned above
    # remain on the cheap path. Until speculative fan-out is implemented, keep the threshold
    # low enough that terse institutional language (for example three conceptual terms) is
    # still understood rather than forcing employees to know official document vocabulary.
    if not query_plan.entity_terms and not query_plan.lookup_term and query_plan.intent == "information":
        has_non_ascii_letters = any(ch.isalpha() and ord(ch) > 127 for ch in question)
        return has_non_ascii_letters or token_count >= 3

    return False


def should_repair_with_semantic_planner(question: str, query_plan: QueryPlan, plan: EvidencePlan) -> bool:
    """Escalate only when the deterministic shape is too lossy for the user's request."""

    if query_plan.intent in {"troubleshooting", "definition"}:
        return False
    if "semantic_repair_recommended" in plan.warnings:
        return True

    token_count = len(_word_tokens(question))
    required = plan.required_goals
    if len(required) > 1:
        return False

    if _CALC_RE.search(question) and token_count >= 10:
        return True
    if len(_clause_fragments(question)) >= 3:
        return True
    if (
        "enumeration" in (query_plan.facets or [])
        and token_count >= 12
        and (_POLICY_ENTITLEMENT_RE.search(question) or _POLICY_CONTEXT_RE.search(question))
    ):
        return True
    if (
        query_plan.intent == "procedure"
        and token_count >= 16
        and (_POLICY_SUPPORT_RE.search(question) or _POLICY_CONTEXT_RE.search(question))
    ):
        return True
    return False


def semantic_planner_system_prompt(max_goals: int, max_queries_per_goal: int) -> str:
    return (
        "You are an evidence-planning component for an internal-document retrieval system. "
        "Do NOT answer the user's question and do NOT use your own domain knowledge to decide facts. "
        "Decompose the request into atomic evidence requirements that can be independently searched and verified. "
        "Use the fewest goals that fully cover the explicit asks: normally one goal per requested sub-question or required calculation input group. Do not create extra authority, approval, governance, conflict, or ownership goals unless the user asked for them or they are strictly necessary to validate an explicit condition. "
        "Preserve every explicit acronym, identifier, number, negation, exception, time/revision qualifier, and condition. "
        "Treat the user's premise as a claim to verify when it may be wrong; never assume it is true. "
        "For causal or multi-step scenarios, distinguish evidence for each event/condition, the requested outcome, and any relationship that must be proven. "
        "For comparisons, retrieve each side independently and add a comparison/relationship goal. "
        "For several terms, never require all terms to co-occur merely to establish each term individually. "
        "A relationship goal must seek evidence of the relationship itself; separate definitions do not prove a relationship. "
        "Search queries are SEARCH-ONLY hypotheses, never evidence. They may paraphrase the user and may hypothesize likely section-heading vocabulary, role/function terminology, governing concepts, procedural nouns, synonyms, and alternate expressions of the requested relation. Such hypotheses must never be treated as facts unless retrieved evidence supports them. "
        "When the query budget permits, make a goal's probes complementary rather than near-duplicates: include a literal/normalized probe, a relation-explicit probe that states the requested subject-predicate-object meaning, and a likely document-vocabulary/heading/role probe. Do not merely change tense, pluralization, punctuation, or word order. "
        "Do not invent numeric thresholds, acronym expansions, technical identifiers, places, or factual outcomes. "
        "The user may write in any language or mix languages. Preserve the original literal terms; for non-English prose you may additionally provide a concise English translation as a retrieval hypothesis when useful, but never translate or expand technical identifiers by guessing. "
        f"Return at most {max_goals} goals and at most {max_queries_per_goal} concise search queries per goal."
    )


def semantic_planner_user_prompt(question: str, deterministic: EvidencePlan) -> str:
    return (
        f"User question:\n{question}\n\n"
        f"Safe deterministic seed plan:\n{deterministic.prompt_block()}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "strategy": "single|multi_lookup|multi_entity|comparison|relationship|conditional|multi_hop|claim_check|enumeration|procedure|overview",\n'
        '  "entities": ["literal user-grounded entity/term", ...],\n'
        '  "needs_research": true|false,\n'
        '  "needs_verification": true|false,\n'
        '  "goals": [\n'
        "    {\n"
        '      "id": "g1",\n'
        '      "kind": "definition|attribute|enumeration|procedure|responsibility|comparison|relationship|condition|exception|consequence|applicability|claim_check|calculation|temporal|fact|scenario|overview",\n'
        '      "question": "what documentary fact must be established",\n'
        '      "search_queries": ["concise retrieval query", ...],\n'
        '      "entity_terms": ["literal terms from the user question", ...],\n'
        '      "required": true|false,\n'
        '      "depends_on": ["gX", ...],\n'
        '      "qualifiers": ["explicit user condition/negation/time qualifier", ...],\n'
        '      "relation": "short relation label or null",\n'
        '      "coverage_contract": "single_fact|primary_definition|all_supported_variants|enumerate_set|ordered_procedure|all_requested_entities|relationship_proof|conditional_rule|calculation_inputs|compare_variants"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Every required part of the user's request must appear in at least one required goal. "
        "Do not add a goal for an issue the user did not ask about unless it is necessary to validate a stated condition or relationship."
    )


def _original_token_sets(question: str) -> tuple[set[str], set[str], set[str]]:
    words = {token.casefold() for token in _word_tokens(question)}
    identifiers = {token.casefold() for token in standalone_identifiers(question)}
    numbers = {token.casefold() for token in _NUMBER_RE.findall(question)}
    return words, identifiers, numbers


def _anchored_entity(value: str, original_words: set[str]) -> bool:
    tokens = [token.casefold() for token in _word_tokens(value)]
    substantive = [token for token in tokens if token not in _GENERIC_QUERY_WORDS]
    if not substantive:
        return False
    return any(token in original_words for token in substantive)


def _safe_generated_query(value: str, original_identifiers: set[str], original_numbers: set[str]) -> bool:
    # New numbers and new acronym-like identifiers are high-risk planner inventions.
    generated_numbers = {token.casefold() for token in _NUMBER_RE.findall(value)}
    if generated_numbers - original_numbers:
        return False
    generated_identifiers = {
        match.group(1).casefold()
        for match in _IDENTIFIER_RE.finditer(value)
        if match.group(1).upper() not in _IDENTIFIER_STOP
    }
    return not (generated_identifiers - original_identifiers)


def evidence_plan_from_payload(
    question: str,
    payload: Any,
    fallback: EvidencePlan,
    *,
    max_goals: int = 8,
    max_queries_per_goal: int = 3,
) -> EvidencePlan:
    """Validate/sanitize an LLM evidence plan and merge safety-critical seed goals."""

    if not isinstance(payload, dict):
        fallback.warnings.append("semantic_planner_invalid_payload")
        return fallback

    original_words, original_identifiers, original_numbers = _original_token_sets(question)
    warnings: list[str] = []

    strategy = str(payload.get("strategy") or fallback.strategy).strip().casefold()
    if strategy not in _ALLOWED_STRATEGIES:
        strategy = fallback.strategy
        warnings.append("invalid_strategy_replaced")

    raw_entities = payload.get("entities") if isinstance(payload.get("entities"), list) else []
    entities = _unique(
        str(item) for item in raw_entities if _anchored_entity(str(item), original_words)
    )
    # Preserve explicit identifiers and deterministic entity extraction if the semantic
    # planner omitted them.
    entities = _unique([*entities, *fallback.entities], limit=12)

    raw_goals = payload.get("goals") if isinstance(payload.get("goals"), list) else []
    goals: list[EvidenceGoal] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_goals[:max_goals], start=1):
        if not isinstance(raw, dict):
            warnings.append("non_object_goal_dropped")
            continue
        goal_id = str(raw.get("id") or f"g{index}").strip().lower()
        if not re.fullmatch(r"g\d{1,2}", goal_id) or goal_id in seen_ids:
            goal_id = f"g{index}"
        seen_ids.add(goal_id)
        kind = str(raw.get("kind") or "fact").strip().casefold()
        if kind not in _ALLOWED_GOAL_KINDS:
            kind = "fact"
            warnings.append(f"{goal_id}:invalid_kind_replaced")

        raw_entity_terms = raw.get("entity_terms") if isinstance(raw.get("entity_terms"), list) else []
        entity_terms = _unique(
            str(item) for item in raw_entity_terms if _anchored_entity(str(item), original_words)
        )

        goal_question = re.sub(r"\s+", " ", str(raw.get("question") or "").strip())[:420]
        if not goal_question or not _safe_generated_query(goal_question, original_identifiers, original_numbers):
            goal_question = "Find documentary evidence for: " + (
                " / ".join(entity_terms) if entity_terms else question
            )
            warnings.append(f"{goal_id}:unsafe_question_replaced")

        raw_queries = raw.get("search_queries") if isinstance(raw.get("search_queries"), list) else []
        search_queries = _unique(
            (
                str(item)[:240]
                for item in raw_queries
                if str(item).strip()
                and _safe_generated_query(str(item), original_identifiers, original_numbers)
            ),
            limit=max_queries_per_goal,
        )
        if not search_queries:
            search_queries = _unique([*entity_terms, goal_question], limit=max_queries_per_goal)

        depends = _unique(
            (str(item).strip().lower() for item in (raw.get("depends_on") or []) if str(item).strip()),
            limit=6,
        )
        qualifiers = _unique(
            (str(item)[:240] for item in (raw.get("qualifiers") or []) if str(item).strip()),
            limit=6,
        )
        relation = str(raw.get("relation") or "").strip()[:120] or None
        coverage_contract = str(raw.get("coverage_contract") or _default_coverage_contract(kind)).strip().casefold()
        if coverage_contract not in _ALLOWED_COVERAGE_CONTRACTS:
            coverage_contract = _default_coverage_contract(kind)
            warnings.append(f"{goal_id}:invalid_coverage_contract_replaced")
        goals.append(
            EvidenceGoal(
                id=goal_id,
                kind=kind,
                question=goal_question,
                search_queries=search_queries,
                entity_terms=entity_terms,
                required=bool(raw.get("required", True)),
                depends_on=depends,
                qualifiers=qualifiers,
                relation=relation,
                coverage_contract=coverage_contract,
            )
        )

    if not goals:
        fallback.warnings.append("semantic_planner_no_valid_goals")
        return fallback

    if "policy_entitlement_plan" in fallback.warnings:
        # Semantic repair may paraphrase all probes back into the employee's wording.
        # Preserve one deterministic institutional-vocabulary bridge per matching goal kind.
        # These are search-only hypotheses and remain subject to evidence verification.
        seed_by_kind = {seed.kind: seed for seed in fallback.goals}
        for goal in goals:
            seed = seed_by_kind.get(goal.kind)
            if seed and seed.search_queries:
                # Deterministic policy plans place the stable institutional-vocabulary
                # bridge first. Semantic repair may improve decomposition, but it must not
                # demote that bridge behind conversational paraphrases or routed retrieval
                # will again search only the user's surface wording.
                bridge = seed.search_queries[0]
                goal.search_queries = _unique(
                    [bridge, *goal.search_queries[:2]],
                    limit=max_queries_per_goal,
                )

    # Ensure every explicit short identifier from the deterministic plan retains an
    # independently searchable goal. This prevents one planner mistake from recreating the
    # all-terms-at-once failure class for multi-identifier requests.
    covered_identifier_terms = {
        term.casefold()
        for goal in goals
        for term in goal.entity_terms
        if term.casefold() in original_identifiers
    }
    next_id = 1
    used_ids = {goal.id for goal in goals}
    for seed in fallback.goals:
        seed_identifiers = {term.casefold() for term in seed.entity_terms} & original_identifiers
        if seed_identifiers and not seed_identifiers.issubset(covered_identifier_terms):
            while f"g{next_id}" in used_ids:
                next_id += 1
            goals.append(
                EvidenceGoal(
                    id=f"g{next_id}",
                    kind=seed.kind,
                    question=seed.question,
                    search_queries=seed.search_queries,
                    entity_terms=seed.entity_terms,
                    required=seed.required,
                    depends_on=[],
                    qualifiers=seed.qualifiers,
                    relation=seed.relation,
                    coverage_contract=seed.coverage_contract,
                )
            )
            used_ids.add(f"g{next_id}")
            covered_identifier_terms.update(seed_identifiers)
            warnings.append("preserved_missing_identifier_goal")
            if len(goals) >= max_goals:
                break

    # Preserve deterministic required clauses/facets that the semantic planner appears to
    # have dropped. This is intentionally lexical and domain-agnostic: it protects pieces
    # of the user's own wording, not any expected answer. It is especially important for
    # long event chains where an LLM planner can accidentally skip the middle condition.
    semantic_text_by_goal = {
        goal.id: " ".join([goal.question, *goal.search_queries, *goal.entity_terms])
        for goal in goals
    }
    for seed in fallback.goals:
        if not seed.required or len(goals) >= max_goals:
            continue
        seed_source = " ".join(seed.search_queries or [seed.question])
        seed_tokens = _content_tokens(seed_source)
        if not seed_tokens:
            continue
        threshold = 1 if len(seed_tokens) == 1 else 2
        covered = any(
            len(seed_tokens & _content_tokens(candidate_text)) >= threshold
            for candidate_text in semantic_text_by_goal.values()
        )
        if covered:
            continue
        while f"g{next_id}" in used_ids:
            next_id += 1
        preserved = EvidenceGoal(
            id=f"g{next_id}",
            kind=seed.kind,
            question=seed.question,
            search_queries=seed.search_queries,
            entity_terms=seed.entity_terms,
            required=True,
            depends_on=[],
            qualifiers=seed.qualifiers,
            relation=seed.relation,
            coverage_contract=seed.coverage_contract,
        )
        goals.append(preserved)
        used_ids.add(preserved.id)
        semantic_text_by_goal[preserved.id] = " ".join(
            [preserved.question, *preserved.search_queries, *preserved.entity_terms]
        )
        warnings.append("preserved_omitted_user_clause_goal")

    # Normalize dependency references after any sanitization and avoid self-dependencies.
    valid_ids = {goal.id for goal in goals}
    for goal in goals:
        goal.depends_on = [item for item in goal.depends_on if item in valid_ids and item != goal.id]

    required_count = sum(1 for goal in goals if goal.required)
    requires_decomposition = required_count > 1 or strategy in {
        "multi_lookup", "multi_entity", "comparison", "relationship", "conditional", "multi_hop", "claim_check"
    }
    return EvidencePlan(
        original=question,
        strategy=strategy,
        entities=entities,
        goals=goals[:max_goals],
        requires_decomposition=requires_decomposition,
        needs_research=bool(payload.get("needs_research", True)) or requires_decomposition,
        needs_verification=bool(payload.get("needs_verification", True)) or requires_decomposition,
        planner_source="semantic",
        warnings=[*fallback.warnings, *warnings],
    )


def plan_query_specs(plan: EvidencePlan, *, goal_ids: set[str] | None = None, max_queries: int = 12) -> list[dict[str, Any]]:
    """Return de-duplicated per-goal retrieval queries while preserving goal attribution."""

    specs_by_key: dict[str, dict[str, Any]] = {}
    for goal in plan.goals:
        if goal_ids is not None and goal.id not in goal_ids:
            continue
        probes = list(goal.search_queries)
        if "policy_entitlement_plan" in plan.warnings and len(probes) >= 2:
            # The deterministic policy planner already places its stable heading-vocabulary
            # bridge first. Preserve that order so routed search sees institutional terms
            # before conversational paraphrases.
            probes = list(probes)
        elif len(probes) >= 3:
            # Under tight query budgets preserve one literal probe plus the most divergent
            # vocabulary/heading hypothesis before spending budget on a near-paraphrase.
            probes = [probes[0], probes[-1], *probes[1:-1]]
        entity_probes = [] if "policy_entitlement_plan" in plan.warnings else list(goal.entity_terms)
        queries = _unique([*probes, *entity_probes, goal.question], limit=4)
        for query in queries:
            key = query.casefold()
            spec = specs_by_key.get(key)
            if spec is None:
                spec = {
                    "text": query,
                    "goal_ids": [],
                    "goal_kinds": [],
                    "weight": 1.0,
                }
                specs_by_key[key] = spec
            if goal.id not in spec["goal_ids"]:
                spec["goal_ids"].append(goal.id)
            if goal.kind not in spec["goal_kinds"]:
                spec["goal_kinds"].append(goal.kind)
            # Relationship/comparison/claim-check queries deserve slightly more weight
            # because merely finding independent entities cannot satisfy these goals.
            if goal.kind in {"relationship", "comparison", "claim_check", "condition", "consequence"}:
                spec["weight"] = 1.12

    # Round-robin through goals rather than taking all formulations for g1 first.
    ordered: list[dict[str, Any]] = []
    goal_order = [goal.id for goal in plan.goals if goal_ids is None or goal.id in goal_ids]
    remaining = list(specs_by_key.values())
    while remaining and len(ordered) < max_queries:
        made_progress = False
        for goal_id in goal_order:
            for index, spec in enumerate(remaining):
                if goal_id in spec["goal_ids"]:
                    ordered.append(spec)
                    remaining.pop(index)
                    made_progress = True
                    break
            if len(ordered) >= max_queries:
                break
        if not made_progress:
            break
    return ordered[:max_queries]


def lookup_terms_by_goal(plan: EvidencePlan, *, goal_ids: set[str] | None = None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for goal in plan.goals:
        if goal_ids is not None and goal.id not in goal_ids:
            continue
        if goal.kind != "definition":
            continue
        terms = _unique(goal.entity_terms, limit=3)
        if terms:
            result[goal.id] = terms
    return result


def satisfaction_from_trace(plan: EvidencePlan, trace: dict[str, Any]) -> GoalSatisfaction:
    stats = trace.get("goal_stats") if isinstance(trace, dict) else None
    stats = stats if isinstance(stats, dict) else {}
    statuses: list[GoalStatus] = []
    missing: list[str] = []
    partial: list[str] = []
    for goal in plan.goals:
        goal_stat = stats.get(goal.id) if isinstance(stats.get(goal.id), dict) else {}
        evidence_count = int(goal_stat.get("evidence_count", 0) or 0)
        lexical_like = int(goal_stat.get("lexical_like_evidence_count", 0) or 0)
        max_rerank = float(goal_stat.get("max_rerank_score", 0.0) or 0.0)
        if evidence_count <= 0:
            status = "missing"
            reason = "No retrieved evidence was attributed to this evidence goal."
            if goal.required:
                missing.append(goal.id)
        elif (
            goal.coverage_contract in {"single_fact", "primary_definition"}
            and goal.kind in {"definition", "enumeration", "fact", "overview"}
            and (lexical_like > 0 or max_rerank >= 0.12)
        ):
            status = "supported"
            reason = "Candidate evidence was retrieved for this single-fact goal; semantic audit may refine support."
        else:
            # Relationship, condition, exception, consequence, applicability, claim-check,
            # calculation, temporal, procedure and attribute goals require semantic support
            # checking. Candidate presence alone is not proof of the requested proposition.
            status = "partial"
            reason = "Relevant candidate evidence exists, but retrieval alone cannot prove this goal."
            if goal.required:
                partial.append(goal.id)
        statuses.append(GoalStatus(goal_id=goal.id, status=status, reason=reason))
    complete = not missing and not partial
    return GoalSatisfaction(
        complete=complete,
        statuses=statuses,
        missing_goal_ids=missing,
        partial_goal_ids=partial,
        audit_source="deterministic",
    )


def goal_audit_system_prompt() -> str:
    return (
        "You are a strict evidence-satisfaction auditor. Do not answer the user's question. "
        "For each evidence goal, decide whether the supplied evidence actually supports that goal. "
        "Use only the supplied evidence IDs. Co-occurrence of terms does not prove a relationship, causation, override, eligibility, entitlement, or applicability. "
        "Separate definitions do not prove that two things interact. A premise stated by the user is not evidence. "
        "Respect each goal's coverage contract. For all_supported_variants/enumerate_set/ordered_procedure/all_requested_entities/compare_variants, one matching passage is not automatically complete when the retrieved evidence indicates additional materially distinct meanings, values, steps, entities, scopes, or variants. Mark supported only when the evidence directly establishes the requested proposition and satisfies the stated coverage contract. "
        "Mark partial when evidence is relevant but a material condition/link/value is missing, missing when no useful evidence is present, and contradicted when supplied evidence directly conflicts with the premise. "
        "Recovery queries are search-only probes. They may introduce generic conceptual, role or section-heading vocabulary to improve recall, but must not invent acronym expansions, numeric thresholds, technical identifiers, places or factual answers."
    )


def goal_audit_user_prompt(question: str, plan: EvidencePlan, evidence_text: str) -> str:
    return (
        f"User question:\n{question}\n\nEvidence plan:\n{plan.prompt_block()}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Return JSON {\"goals\": [{\"goal_id\": \"g1\", \"status\": \"supported|partial|missing|contradicted\", "
        "\"evidence_ids\": [\"E1\"], \"reason\": \"short evidence-based reason\", \"recovery_queries\": [\"...\"]}]}. "
        "Return one entry for every plan goal."
    )


def goal_satisfaction_from_payload(
    plan: EvidencePlan,
    payload: Any,
    *,
    valid_evidence_ids: set[str],
    valid_evidence_ids_by_goal: dict[str, set[str]] | None = None,
    fallback: GoalSatisfaction,
) -> GoalSatisfaction:
    if not isinstance(payload, dict) or not isinstance(payload.get("goals"), list):
        return fallback

    by_id = {goal.id: goal for goal in plan.goals}
    rows: dict[str, GoalStatus] = {}
    for raw in payload["goals"]:
        if not isinstance(raw, dict):
            continue
        goal_id = str(raw.get("goal_id") or "").strip().lower()
        if goal_id not in by_id:
            continue
        status = str(raw.get("status") or "missing").strip().casefold()
        if status not in {"supported", "partial", "missing", "contradicted"}:
            status = "missing"
        goal_allowed_ids = (
            valid_evidence_ids_by_goal.get(goal_id, set())
            if valid_evidence_ids_by_goal is not None
            else valid_evidence_ids
        )
        # An auditor may only use evidence that retrieval actually attributed to this goal.
        # This prevents a strong A passage from being cited as proof for missing goal B.
        allowed_ids = valid_evidence_ids & set(goal_allowed_ids)
        evidence_ids = _unique(
            (
                str(item).strip().upper()
                for item in (raw.get("evidence_ids") or [])
                if str(item).strip().upper() in allowed_ids
            ),
            limit=12,
        )
        # The semantic audit cannot declare documentary support/contradiction without
        # pointing to at least one real evidence item. This prevents an audit model from
        # turning its own prior knowledge into a false completeness signal.
        if status in {"supported", "contradicted"} and not evidence_ids:
            status = "partial"
        _words, original_identifiers, original_numbers = _original_token_sets(plan.original)
        recovery_queries = _unique(
            (
                str(item)[:240]
                for item in (raw.get("recovery_queries") or [])
                if str(item).strip()
                and _safe_generated_query(str(item), original_identifiers, original_numbers)
            ),
            limit=3,
        )
        rows[goal_id] = GoalStatus(
            goal_id=goal_id,
            status=status,
            evidence_ids=evidence_ids,
            reason=str(raw.get("reason") or "").strip()[:500],
            recovery_queries=recovery_queries,
        )

    # Missing audit rows are not treated as success.
    statuses: list[GoalStatus] = []
    missing: list[str] = []
    partial: list[str] = []
    contradicted: list[str] = []
    for goal in plan.goals:
        status = rows.get(goal.id) or GoalStatus(
            goal_id=goal.id,
            status="missing",
            reason="The semantic audit did not return a valid status for this goal.",
        )
        statuses.append(status)
        if not goal.required:
            continue
        if status.status == "missing":
            missing.append(goal.id)
        elif status.status == "partial":
            partial.append(goal.id)
        elif status.status == "contradicted":
            # Contradiction can be a valid resolution for a claim-check goal. For other
            # goal kinds it indicates the requested premise/relation needs careful handling,
            # not another blind retrieval loop.
            if goal.kind == "claim_check":
                contradicted.append(goal.id)
            else:
                contradicted.append(goal.id)

    # A directly contradicted claim-check goal is considered resolved (the answer should
    # explain the correction). Contradicted relationship/fact goals are also evidenceful,
    # but verification remains mandatory. Missing/partial goals are what trigger recovery.
    complete = not missing and not partial
    return GoalSatisfaction(
        complete=complete,
        statuses=statuses,
        missing_goal_ids=missing,
        partial_goal_ids=partial,
        contradicted_goal_ids=contradicted,
        audit_source="semantic",
    )


def recovery_queries_for_goals(plan: EvidencePlan, satisfaction: GoalSatisfaction) -> dict[str, list[str]]:
    by_status = {status.goal_id: status for status in satisfaction.statuses}
    result: dict[str, list[str]] = {}
    for goal in plan.goals:
        if goal.id not in set(satisfaction.missing_goal_ids + satisfaction.partial_goal_ids):
            continue
        status = by_status.get(goal.id)
        queries = _unique(
            [*goal.search_queries, *goal.entity_terms, goal.question, *(status.recovery_queries if status else [])],
            limit=4,
        )
        result[goal.id] = queries
    return result
