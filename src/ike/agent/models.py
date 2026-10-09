from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AnswerRequirement(BaseModel):
    """One distinct thing the user's final answer must cover."""

    id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=700)


class ResearchTask(BaseModel):
    """One AI-defined research operation."""

    id: str = Field(min_length=1, max_length=64)
    purpose: str = Field(min_length=1, max_length=700)
    requirement_ids: list[str] = Field(default_factory=list, max_length=12)
    kind: Literal[
        "search",
        "enumerate",
        "source_lookup",
        "structure",
        "context",
    ] = "search"
    query: str = Field(default="", max_length=1400)
    query_variants: list[str] = Field(default_factory=list, max_length=4)
    search_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    exact_terms: list[str] = Field(default_factory=list, max_length=8)
    coverage_facets: list[
        Literal[
            "rolling_stock",
            "line_code",
            "document_family",
            "document_type",
            "document",
        ]
    ] = Field(default_factory=list, max_length=3)
    source_query: str = Field(default="", max_length=500)
    document_id: str | None = None
    chunk_id: str | None = None
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    top_k: int = Field(default=14, ge=4, le=80)
    context_before: int = Field(default=0, ge=0, le=6)
    context_after: int = Field(default=0, ge=0, le=8)
    context_hits: int = Field(default=0, ge=0, le=3)

    @field_validator(
        "requirement_ids",
        "query_variants",
        "exact_terms",
        "coverage_facets",
        "depends_on",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class ResearchPlan(BaseModel):
    """AI-created research graph for the user's complete information need."""

    interpretation: str = Field(min_length=1, max_length=1200)
    needs_corpus: bool = True
    answer_requirements: list[AnswerRequirement] = Field(
        min_length=1,
        max_length=12,
    )
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def _validate_requirement_mapping(self):
        requirement_ids = {item.id for item in self.answer_requirements}
        task_ids = {task.id for task in self.tasks}

        for task in self.tasks:
            unknown_requirements = set(task.requirement_ids) - requirement_ids
            if unknown_requirements:
                raise ValueError(
                    "Research task references unknown requirement IDs: "
                    + ", ".join(sorted(unknown_requirements))
                )
            unknown_dependencies = set(task.depends_on) - task_ids
            if unknown_dependencies:
                raise ValueError(
                    "Research task references unknown dependency IDs: "
                    + ", ".join(sorted(unknown_dependencies))
                )

        if self.needs_corpus:
            if not self.tasks:
                raise ValueError(
                    "A corpus-grounded plan must include at least one research task."
                )
            mapped = {
                requirement_id
                for task in self.tasks
                for requirement_id in task.requirement_ids
            }
            missing = requirement_ids - mapped
            if missing:
                raise ValueError(
                    "Every answer requirement must map to at least one research task. "
                    "Missing: " + ", ".join(sorted(missing))
                )
        return self


class RequirementAssessment(BaseModel):
    requirement_id: str = Field(min_length=1, max_length=64)
    status: Literal["supported", "partial", "missing"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    note: str = Field(default="", max_length=900)


class EvidenceSelectionAssessment(BaseModel):
    requirement_id: str = Field(min_length=1, max_length=64)
    status: Literal["covered", "partial", "missing"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    coverage_groups: list[str] = Field(default_factory=list, max_length=40)
    note: str = Field(default="", max_length=800)


class EvidenceSelectionDecision(BaseModel):
    """Compact AI shortlist before expensive answer reasoning."""

    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    requirement_assessments: list[EvidenceSelectionAssessment] = Field(
        min_length=1,
        max_length=12,
    )
    suspected_gaps: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("selected_evidence_ids", "suspected_gaps", mode="before")
    @classmethod
    def _coerce_selection_lists(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class AnswerDecision(BaseModel):
    """Evidence reasoning plus final-answer decision."""

    status: Literal["answer", "needs_evidence"]
    answer: str = Field(default="", max_length=12000)
    requirement_assessments: list[RequirementAssessment] = Field(
        min_length=1,
        max_length=12,
    )
    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    missing_requirements: list[str] = Field(default_factory=list, max_length=12)
    gap_tasks: list[ResearchTask] = Field(default_factory=list, max_length=6)
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator(
        "selected_evidence_ids",
        "missing_requirements",
        mode="before",
    )
    @classmethod
    def _coerce_answer_lists(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class FinalAnswerDecision(BaseModel):
    """Final grounded response after the optional gap round."""

    answer: str = Field(min_length=1, max_length=12000)
    requirement_assessments: list[RequirementAssessment] = Field(
        min_length=1,
        max_length=12,
    )
    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("selected_evidence_ids", mode="before")
    @classmethod
    def _coerce_selected(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value
