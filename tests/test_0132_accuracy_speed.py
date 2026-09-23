from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import (
    build_deterministic_evidence_plan,
    should_use_semantic_planner,
)
from ike.workflows.routing import extract_lookup_term, extract_source_scope


def test_definition_source_scope_is_separated_from_target():
    question = "definition of Incident as per MRGR"
    assert extract_lookup_term(question) == "INCIDENT"
    assert extract_source_scope(question) == "MRGR"

    plan = build_query_plan(question)
    assert plan.intent == "definition"
    assert plan.lookup_term == "INCIDENT"
    assert plan.source_scope == "MRGR"


def test_simple_definition_does_not_use_semantic_planner():
    for question in (
        "Definition of Train Operator",
        "What is incident",
        "definition of Incident as per MRGR",
    ):
        query_plan = build_query_plan(question)
        evidence_plan = build_deterministic_evidence_plan(question, query_plan)
        assert any(goal.coverage_contract == "primary_definition" for goal in evidence_plan.required_goals)
        assert should_use_semantic_planner(question, query_plan, evidence_plan) is False


def test_vcb_open_gets_operational_rewrites():
    plan = build_query_plan("VCB open showing")
    assert plan.intent == "troubleshooting"
    joined = " | ".join(plan.semantic_queries).casefold()
    assert "vcb tripping vcb gets opened indication" in joined
    assert "vcb open trip troubleshooting reset close occ procedure" in joined


def test_missing_target_speed_gets_operational_rewrites():
    plan = build_query_plan("Speed is not available but signal is green")
    assert plan.intent == "troubleshooting"
    joined = " | ".join(plan.semantic_queries).casefold()
    assert "target speed not available not received atp troubleshooting" in joined
    assert "atp information unavailable atp mcb reset ros rm occ procedure" in joined
