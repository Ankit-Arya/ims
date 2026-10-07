from __future__ import annotations

from uuid import UUID

from ike.agent.models import ResearchPlan
from ike.agent.prompts import planner_system_prompt, planner_user_prompt
from ike.core.config import get_settings
from ike.services.llm import LLMClient


class QueryIntelligenceAgent:
    """Understand the user's full information need and create a research graph."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.settings = get_settings()
        self.llm = llm or LLMClient()

    def plan(
        self,
        question: str,
        *,
        operational_context: dict,
        selected_document_ids: list[UUID] | None,
        recent_history: list[dict],
    ) -> tuple[ResearchPlan, int, int]:
        plan, result = self.llm.generate_structured(
            system=planner_system_prompt(),
            user=planner_user_prompt(
                question=question,
                operational_context=operational_context,
                selected_document_ids=[
                    str(value) for value in (selected_document_ids or [])
                ],
                recent_history=recent_history,
            ),
            schema_model=ResearchPlan,
            strong=True,
            max_output_tokens=self.settings.agent_planner_max_output_tokens,
            reasoning_effort=self.settings.agent_planner_reasoning,
        )
        return plan, result.input_tokens, result.output_tokens
