from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ike.retrieval.query_plan import QueryPlan


AnswerShape = Literal[
    "fact",
    "scalar",
    "definition",
    "procedure",
    "enumeration",
    "comparison",
    "relationship",
    "calculation",
    "synthesis",
]
PremiseStatus = Literal[
    "unverified",
    "supported",
    "partially_supported",
    "contradicted",
    "not_established",
    "conditionally_true",
]


class QueryQualifier(BaseModel):
    """Domain-neutral qualifier attached to an entity/relation hypothesis."""

    model_config = ConfigDict(extra="forbid")

    dimension: str = ""
    value: str = ""

    @field_validator("dimension", "value")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:240]


class PremiseClaim(BaseModel):
    """A user-supplied proposition that must be verified, not assumed true."""

    model_config = ConfigDict(extra="forbid")

    text: str
    status: PremiseStatus = "unverified"

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:800]


class QueryFrame(BaseModel):
    """Typed, domain-neutral query understanding contract.

    The frame expands retrieval; it never replaces ``original``.  Any generated canonical
    term, paraphrase, acronym expansion or typo candidate is a search hypothesis only until
    documentary evidence supports it.
    """

    model_config = ConfigDict(extra="forbid")

    original: str
    answer_shape: AnswerShape = "fact"
    entities: list[str] = Field(default_factory=list, max_length=24)
    relations: list[str] = Field(default_factory=list, max_length=16)
    qualifiers: list[QueryQualifier] = Field(default_factory=list, max_length=24)
    explicit_scope: list[str] = Field(default_factory=list, max_length=16)
    premise_claims: list[PremiseClaim] = Field(default_factory=list, max_length=8)
    canonical_terms: list[str] = Field(default_factory=list, max_length=24)
    alternate_phrasings: list[str] = Field(default_factory=list, max_length=12)
    acronym_expansions: list[str] = Field(default_factory=list, max_length=12)
    typo_candidates: list[str] = Field(default_factory=list, max_length=12)
    ambiguity: list[str] = Field(default_factory=list, max_length=12)
    requires_cross_document_reasoning: bool = False

    @field_validator(
        "entities",
        "relations",
        "explicit_scope",
        "canonical_terms",
        "alternate_phrasings",
        "acronym_expansions",
        "typo_candidates",
        "ambiguity",
    )
    @classmethod
    def normalize_list(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in values or []:
            value = re.sub(r"\s+", " ", str(raw or "")).strip()[:500]
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def retrieval_hypotheses(self, *, limit: int = 16) -> list[str]:
        """Return bounded search hypotheses while preserving the exact user query first."""

        values = [
            self.original,
            *self.alternate_phrasings,
            *self.canonical_terms,
            *self.acronym_expansions,
            *self.typo_candidates,
        ]
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= max(1, limit):
                break
        return result


def deterministic_query_frame(question: str, plan: QueryPlan) -> QueryFrame:
    """Create a safe frame without adding undocumented domain vocabulary.

    This fallback intentionally derives only what the deterministic planner already knows.
    Semantic/corpus-aware enrichers may append hypotheses later, but Q0 remains intact.
    """

    answer_shape: AnswerShape = "fact"
    if plan.intent == "procedure":
        answer_shape = "procedure"
    elif "enumeration" in (plan.facets or []):
        answer_shape = "enumeration"
    elif plan.lookup_term:
        answer_shape = "definition"
    elif re.search(r"\b(?:compare|difference|versus|\bvs\b)\b", question, re.IGNORECASE):
        answer_shape = "comparison"
    elif re.search(r"\b(?:calculate|compute|sum|percentage|percent)\b", question, re.IGNORECASE):
        answer_shape = "calculation"
    elif re.search(
        r"\b(?:max(?:imum)?|min(?:imum)?|highest|lowest|how\s+many|how\s+much|"
        r"rate|ratio|capacity|dimension|dimensions|length|width|height|weight|value)\b",
        question,
        re.IGNORECASE,
    ):
        answer_shape = "scalar"

    entities = list(plan.entity_terms or [])
    if plan.entity_term and plan.entity_term not in entities:
        entities.append(plan.entity_term)
    if plan.rolling_stock and plan.rolling_stock not in entities:
        entities.append(plan.rolling_stock)

    explicit_scope = [
        value
        for value in [plan.line, plan.rolling_stock, *(plan.scope_terms or [])]
        if value
    ]

    premise_claims: list[PremiseClaim] = []
    if re.search(
        r"(?:\b(?:right|correct|true)\??$|(?:isn['’]?t|aren['’]?t|doesn['’]?t|don['’]?t)\s+it\??$)",
        question.strip(),
        re.IGNORECASE,
    ):
        premise_claims.append(PremiseClaim(text=question, status="unverified"))

    return QueryFrame(
        original=question,
        answer_shape=answer_shape,
        entities=entities,
        relations=list(plan.attribute_terms or []),
        explicit_scope=explicit_scope,
        premise_claims=premise_claims,
        alternate_phrasings=list(plan.semantic_queries[1:] if plan.semantic_queries else []),
        canonical_terms=list(plan.content_terms or []),
        requires_cross_document_reasoning=bool(plan.coverage_sensitive),
    )
