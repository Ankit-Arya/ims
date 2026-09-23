from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnswerPlan:
    direct_focus: str = "Answer the user's information need directly from the evidence."
    sections: list[str] = field(default_factory=list)
    use_table: bool = False
    table_purpose: str | None = None
    include_related_context: bool = True
    related_context: list[str] = field(default_factory=list)
    scenario_dimensions: list[str] = field(default_factory=list)

    def prompt_block(self) -> str:
        sections = ", ".join(self.sections) if self.sections else "adapt structure to the evidence"
        related = "; ".join(self.related_context) if self.related_context else "only materially useful nearby context"
        scenarios = ", ".join(self.scenario_dimensions) if self.scenario_dimensions else "none identified"
        return (
            f"Direct focus: {self.direct_focus}\n"
            f"Suggested sections: {sections}\n"
            f"Table useful: {'yes' if self.use_table else 'no'}"
            + (f" ({self.table_purpose})" if self.table_purpose else "")
            + f"\nScenario dimensions: {scenarios}\n"
            f"Related context to include when evidence supports it: {related}"
        )


def answer_plan_from_payload(payload: Any) -> AnswerPlan:
    if not isinstance(payload, dict):
        return AnswerPlan()

    def string_list(name: str, limit: int) -> list[str]:
        raw = payload.get(name) or []
        if not isinstance(raw, list):
            return []
        return [str(item).strip()[:160] for item in raw if str(item).strip()][:limit]

    return AnswerPlan(
        direct_focus=str(payload.get("direct_focus") or "Answer the user's information need directly from the evidence.")[:500],
        sections=string_list("sections", 8),
        use_table=bool(payload.get("use_table", False)),
        table_purpose=(str(payload.get("table_purpose")).strip()[:300] if payload.get("table_purpose") else None),
        include_related_context=bool(payload.get("include_related_context", True)),
        related_context=string_list("related_context", 6),
        scenario_dimensions=string_list("scenario_dimensions", 6),
    )
