#!/usr/bin/env python3
"""DB-free structural smoke checks for IMS 0.8 retrieval control plane.

The examples are synthetic. No expected production-domain answer is encoded here.
"""
from __future__ import annotations

from pathlib import Path

from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.search_plan import relaxed_websearch_expression, table_retrieval_relevant
from ike.workflows.evidence_planning import (
    build_deterministic_evidence_plan,
    should_use_semantic_planner,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    natural = "Who activates the cooling plant and when?"
    query_plan = build_query_plan(natural)
    evidence_plan = build_deterministic_evidence_plan(natural, query_plan)
    assert should_use_semantic_planner(natural, query_plan, evidence_plan)
    assert "cooling" in relaxed_websearch_expression(natural).casefold()
    assert not table_retrieval_relevant(
        goal_kinds=["responsibility"], facets=[], question="Who starts the cooling plant?"
    )

    multi = "AX1 BX2 CX3"
    qp_multi = build_query_plan(multi)
    ep_multi = build_deterministic_evidence_plan(multi, qp_multi)
    assert ep_multi.strategy == "multi_lookup"
    assert not should_use_semantic_planner(multi, qp_multi, ep_multi)
    assert all(goal.coverage_contract == "all_supported_variants" for goal in ep_multi.goals)

    graph = (root / "src/ike/workflows/qa_graph.py").read_text()
    engine = (root / "src/ike/retrieval/engine.py").read_text()
    assert 'graph.add_node("priority_search", self._priority_search)' in graph
    assert 'source_stage="priority"' in graph
    assert "goal_local_rerank_enabled" in engine
    assert "_section_navigation_candidates" in engine
    assert "_relaxed_lexical" in engine
    assert engine.index("if goal_expansions and not include_base_query:") < engine.index(
        "for spec in goal_specs:"
    )

    production = "\n".join(
        (root / name).read_text().casefold()
        for name in [
            "src/ike/retrieval/search_plan.py",
            "src/ike/retrieval/source_policy.py",
            "src/ike/workflows/query_repair.py",
        ]
    )
    for forbidden in [
        "open the station at least ten minutes",
        "battery isolation contactor",
        "brake isolating cock",
        "bogie isolation cock",
    ]:
        assert forbidden not in production

    print("PASS: IMS 0.8.0 retrieval control-plane structural checks")


if __name__ == "__main__":
    main()
