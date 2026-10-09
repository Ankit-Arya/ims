from __future__ import annotations

from ike.agent.models import EvidenceSelectionDecision, ResearchPlan
from ike.agent.prompts import (
    evidence_selector_system_prompt,
    evidence_selector_user_prompt,
)
from ike.agent.research import ResearchBundle
from ike.core.config import get_settings
from ike.services.llm import LLMClient


class EvidenceSelectionAgent:
    """Cheap semantic evidence triage before strong answer reasoning."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.settings = get_settings()
        self.llm = llm or LLMClient()

    def select(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        bundle: ResearchBundle,
    ) -> tuple[EvidenceSelectionDecision | None, int, int]:
        if not self.settings.agent_evidence_selector_enabled:
            return None, 0, 0

        rows = bundle.evidence_selection_rows(
            max_items=self.settings.agent_evidence_selector_max_candidates,
            excerpt_chars=self.settings.agent_evidence_selector_excerpt_chars,
        )
        if len(rows) < self.settings.agent_evidence_selector_min_candidates:
            return None, 0, 0

        decision, result = self.llm.generate_structured(
            system=evidence_selector_system_prompt(),
            user=evidence_selector_user_prompt(
                question=question,
                plan=plan.model_dump(),
                evidence=rows,
            ),
            schema_model=EvidenceSelectionDecision,
            strong=False,
            max_output_tokens=self.settings.agent_evidence_selector_max_output_tokens,
            reasoning_effort=self.settings.agent_evidence_selector_reasoning,
        )

        expected_requirements = {
            requirement.id for requirement in plan.answer_requirements
        }
        assessed_requirements = {
            assessment.requirement_id
            for assessment in decision.requirement_assessments
        }
        available_evidence = {
            str(row["evidence_id"])
            for row in rows
        }

        if assessed_requirements != expected_requirements:
            raise ValueError(
                "Evidence selector did not assess every planned requirement."
            )

        selected = set(decision.selected_evidence_ids)
        if selected - available_evidence:
            raise ValueError(
                "Evidence selector referenced evidence outside the supplied candidate set."
            )

        assessment_evidence_order: list[str] = []
        for assessment in decision.requirement_assessments:
            assessment_evidence_order.extend(assessment.evidence_ids)
            evidence_ids = set(assessment.evidence_ids)
            if evidence_ids - available_evidence:
                raise ValueError(
                    "Evidence selector assessment referenced unknown evidence."
                )
            if assessment.status == "covered" and not evidence_ids:
                raise ValueError(
                    "Evidence selector marked a requirement covered without evidence."
                )

        decision.selected_evidence_ids = list(
            dict.fromkeys(
                [
                    *assessment_evidence_order,
                    *decision.selected_evidence_ids,
                ]
            )
        )[:24]

        return decision, result.input_tokens, result.output_tokens
