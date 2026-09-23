#!/usr/bin/env python3
"""DB-free smoke checks for IMS 0.7.0 compositional evidence planning.

These checks validate question structure only. They contain no expected domain answers.
"""
from __future__ import annotations

from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import (
    build_deterministic_evidence_plan,
    plan_query_specs,
    should_use_semantic_planner,
)


def plan(question: str):
    query_plan = build_query_plan(question)
    evidence_plan = build_deterministic_evidence_plan(question, query_plan)
    return query_plan, evidence_plan


def main() -> None:
    cases = [
        "BIC TCMS TIMS",
        "What is cooling unit and where is it located?",
        "Compare AX1 BX2 CX3",
        "Is the emergency limit always 20 km/h?",
        "If the primary pump stops then the backup system starts; what action is required?",
        "pension gratuity eligibility",
        "overtime",
    ]
    for question in cases:
        query_plan, evidence_plan = plan(question)
        print(f"\nQUESTION: {question}")
        print(f"  query_intent={query_plan.intent}")
        print(f"  strategy={evidence_plan.strategy}")
        print(f"  planner_source={evidence_plan.planner_source}")
        print(f"  needs_research={evidence_plan.needs_research}")
        print(f"  needs_verification={evidence_plan.needs_verification}")
        print(f"  semantic_planner={should_use_semantic_planner(question, query_plan, evidence_plan)}")
        for goal in evidence_plan.goals:
            print(
                f"  {goal.id}: kind={goal.kind} required={goal.required} "
                f"entities={goal.entity_terms} queries={goal.search_queries}"
            )

    _, identifiers = plan("BIC TCMS TIMS")
    assert identifiers.entities == ["BIC", "TCMS", "TIMS"]
    assert len(identifiers.required_goals) == 3
    specs = plan_query_specs(identifiers, max_queries=14)
    assert [item["text"] for item in specs[:3]] == ["BIC", "TCMS", "TIMS"]

    _, relation = plan("Compare AX1 BX2 CX3")
    assert relation.strategy == "comparison"
    assert any(goal.kind == "comparison" and goal.required for goal in relation.goals)

    _, facets = plan("What is cooling unit and where is it located?")
    assert {goal.kind for goal in facets.required_goals} == {"definition", "attribute"}

    q, conditional = plan(
        "If the primary pump stops then the backup system starts; what action is required?"
    )
    assert conditional.needs_verification
    assert should_use_semantic_planner(q.original, q, conditional)

    print("\nPASS: IMS 0.7.0 compositional planning smoke checks")


if __name__ == "__main__":
    main()
