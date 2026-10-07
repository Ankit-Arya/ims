from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from ike.agent.models import (
    AnswerDecision,
    FinalAnswerDecision,
    ResearchPlan,
)
from ike.agent.prompts import answer_system_prompt, answer_user_prompt
from ike.agent.research import ResearchBundle
from ike.core.config import get_settings
from ike.services.llm import LLMClient

DecisionT = TypeVar("DecisionT", bound=BaseModel)


class EvidenceAnswerAgent:
    """Reason across evidence and either answer or request one precise gap round."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.settings = get_settings()
        self.llm = llm or LLMClient()

    @staticmethod
    def _coverage_error(
        plan: ResearchPlan,
        decision: BaseModel,
        bundle: ResearchBundle,
        *,
        allow_gap: bool,
    ) -> str:
        expected = {item.id for item in plan.answer_requirements}
        assessments = getattr(decision, "requirement_assessments", [])
        actual = {
            str(item.requirement_id)
            for item in assessments
            if getattr(item, "requirement_id", None)
        }
        available_evidence = set(bundle.evidence_id_by_chunk.values())
        parts: list[str] = []

        missing = expected - actual
        unknown = actual - expected
        if missing:
            parts.append(
                "missing requirement assessments: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            parts.append(
                "unknown requirement assessments: "
                + ", ".join(sorted(unknown))
            )

        invalid_selected = set(
            getattr(decision, "selected_evidence_ids", [])
        ) - available_evidence
        if invalid_selected:
            parts.append(
                "selected evidence IDs were not supplied by retrieval: "
                + ", ".join(sorted(invalid_selected))
            )

        incomplete_ids: list[str] = []
        for assessment in assessments:
            requirement_id = str(assessment.requirement_id)
            evidence_ids = set(assessment.evidence_ids or [])
            invalid = evidence_ids - available_evidence
            if invalid:
                parts.append(
                    f"{requirement_id} references nonexistent evidence IDs: "
                    + ", ".join(sorted(invalid))
                )
            if (
                plan.needs_corpus
                and assessment.status == "supported"
                and not evidence_ids
            ):
                parts.append(
                    f"{requirement_id} is marked supported without documentary evidence."
                )
            if assessment.status != "supported":
                incomplete_ids.append(requirement_id)

        if allow_gap:
            status = getattr(decision, "status", "")
            gap_tasks = getattr(decision, "gap_tasks", [])
            if incomplete_ids and status != "needs_evidence":
                parts.append(
                    "partial/missing requirements must trigger status=needs_evidence: "
                    + ", ".join(sorted(incomplete_ids))
                )
            if status == "needs_evidence":
                if not gap_tasks:
                    parts.append(
                        "status=needs_evidence requires at least one targeted gap task."
                    )
                routing_only = [
                    task.id
                    for task in gap_tasks
                    if task.kind == "source_lookup"
                ]
                if routing_only:
                    parts.append(
                        "The gap round is the final retrieval round, so source_lookup "
                        "alone cannot satisfy a factual requirement. Use search or "
                        "enumerate with source_query to route and retrieve content in "
                        "one task. Invalid gap tasks: "
                        + ", ".join(routing_only)
                    )
                mapped_gap_ids = {
                    requirement_id
                    for task in gap_tasks
                    for requirement_id in task.requirement_ids
                }
                unmapped = set(incomplete_ids) - mapped_gap_ids
                if unmapped:
                    parts.append(
                        "gap tasks do not cover incomplete requirement IDs: "
                        + ", ".join(sorted(unmapped))
                    )

        return "; ".join(parts)

    def _generate_checked(
        self,
        *,
        question: str,
        operational_context: dict,
        recent_history: list[dict],
        plan: ResearchPlan,
        bundle: ResearchBundle,
        allow_gap: bool,
        schema_model: type[DecisionT],
    ) -> tuple[DecisionT, int, int]:
        total_input = 0
        total_output = 0
        feedback = ""

        for _attempt in range(2):
            user_prompt = answer_user_prompt(
                question=question,
                operational_context=operational_context,
                recent_history=recent_history,
                plan=plan.model_dump(),
                observations=bundle.observations[-44:],
                evidence=bundle.evidence_rows(
                    max_items=self.settings.agent_evidence_catalog_limit
                ),
            )
            if feedback:
                user_prompt += (
                    "\n\nSTRUCTURAL VALIDATION ERROR FROM THE PREVIOUS RESPONSE:\n"
                    + feedback
                    + "\nReturn a corrected decision. Do not downgrade an incomplete "
                    "requirement to supported unless the supplied evidence really covers it."
                )

            decision, result = self.llm.generate_structured(
                system=answer_system_prompt(allow_gap=allow_gap),
                user=user_prompt,
                schema_model=schema_model,
                strong=True,
                max_output_tokens=self.settings.agent_answer_max_output_tokens,
                reasoning_effort=self.settings.agent_answer_reasoning,
            )
            total_input += result.input_tokens
            total_output += result.output_tokens

            feedback = self._coverage_error(
                plan,
                decision,
                bundle,
                allow_gap=allow_gap,
            )
            if not feedback:
                return decision, total_input, total_output

        raise ValueError(
            "Answer agent failed requirement/evidence validation: " + feedback
        )

    def decide(
        self,
        *,
        question: str,
        operational_context: dict,
        recent_history: list[dict],
        plan: ResearchPlan,
        bundle: ResearchBundle,
    ) -> tuple[AnswerDecision, int, int]:
        return self._generate_checked(
            question=question,
            operational_context=operational_context,
            recent_history=recent_history,
            plan=plan,
            bundle=bundle,
            allow_gap=True,
            schema_model=AnswerDecision,
        )

    def finalize(
        self,
        *,
        question: str,
        operational_context: dict,
        recent_history: list[dict],
        plan: ResearchPlan,
        bundle: ResearchBundle,
    ) -> tuple[FinalAnswerDecision, int, int]:
        return self._generate_checked(
            question=question,
            operational_context=operational_context,
            recent_history=recent_history,
            plan=plan,
            bundle=bundle,
            allow_gap=False,
            schema_model=FinalAnswerDecision,
        )
