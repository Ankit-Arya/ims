from types import SimpleNamespace

import pytest

from uuid import uuid4

from ike.retrieval.applicability import applicability_score
from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.types import Candidate, Evidence
from ike.retrieval.table_context import table_metadata_context
from ike.services.okf import document_topic_terms
from ike.workflows.evidence_planning import (
    build_deterministic_evidence_plan,
    evidence_plan_from_payload,
    plan_query_specs,
    should_repair_with_semantic_planner,
)
from ike.workflows.routing import extract_source_scope


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Types and amount of allowances as per entitlement", None),
        ("Claims for employees attending a meeting away from Delhi", None),
        ("Reimbursement as per pay scale or designation", None),
        ("Definition of incident as per MRGR 2020", "MRGR 2020"),
        ("What are the leave rules according to the HR Compendium 2025", "HR Compendium 2025"),
    ],
)
def test_source_scope_requires_real_document_like_name(question, expected):
    assert extract_source_scope(question) == expected


def _plan(question: str):
    qp = build_query_plan(question)
    return qp, build_deterministic_evidence_plan(question, qp)


def test_different_designations_is_not_conflict_or_precedence():
    question = "All types of claims for different designations when they are on meeting away from Delhi"
    qp, plan = _plan(question)

    assert qp.source_scope is None
    assert plan.strategy != "relationship"
    assert all(goal.relation != "conflict_or_precedence" for goal in plan.goals)
    assert {goal.kind for goal in plan.goals} >= {"applicability", "enumeration", "attribute"}


def test_allowance_types_and_amounts_split_set_from_rate_lookup():
    qp, plan = _plan("Types and amount of allowances as per entitlement")

    assert qp.source_scope is None
    assert [goal.kind for goal in plan.goals] == ["enumeration", "attribute"]
    assert plan.strategy == "enumeration"
    assert "policy_entitlement_plan" in plan.warnings


def test_long_expense_claim_gets_inputs_not_one_scalar_goal():
    question = (
        "I need to claim for a meeting I attended in Hyderabad, what all do I need to claim "
        "and mention total amount too. I left home on 18 September and came back on 28 September. "
        "Hotel was booked for the stay and I travelled by 3AC both ways."
    )
    qp, plan = _plan(question)

    kinds = {goal.kind for goal in plan.goals}
    assert {"applicability", "enumeration", "attribute", "procedure", "calculation"} <= kinds
    assert plan.requires_decomposition is True
    assert should_repair_with_semantic_planner(question, qp, plan) is True


def test_semantic_repair_does_not_reopen_bounded_operational_multipart():
    question = (
        "If 3 BICs of an RS1 train are isolated, what operating restrictions apply, "
        "can the train remain in service, and what action must the TO take?"
    )
    qp, plan = _plan(question)

    assert plan.strategy == "multi_hop"
    assert should_repair_with_semantic_planner(question, qp, plan) is False


def test_okf_topic_terms_capture_structure_and_table_labels():
    chunks = [
        SimpleNamespace(
            section_path=["TRAVELLING ALLOWANCE / DAILY ALLOWANCE", "COMPENSATION FOR STAY AND PERSONAL EXPENSES"],
            content_kind="text",
            contextual_text="",
            text="",
        ),
        *[
            SimpleNamespace(
                section_path=["HOTEL ENTITLEMENT"],
                content_kind="table",
                contextual_text="Designation Hotel rate Daily Allowance Reimbursement",
                text="",
            )
            for _ in range(3)
        ],
    ]

    terms = set(document_topic_terms(chunks))
    assert {"travelling", "allowance", "compensation", "stay", "hotel", "entitlement", "designation", "reimbursement"} <= terms


def test_generation_uses_raw_prose_and_marks_ancestor_navigation():
    candidate = Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=1,
        page_from=10,
        page_to=10,
        section_path=["PART A", "STALE PARENT", "LOCAL RULE"],
        content_kind="text",
        text="Actual governing sentence.",
        contextual_text="PART A\nSTALE PARENT\nLOCAL RULE\nActual governing sentence.",
        document_title="Policy.pdf",
        filename="Policy.pdf",
        revision=None,
        authority=None,
    )

    block = Evidence("E1", candidate).prompt_block()

    assert "Local section: LOCAL RULE" in block
    assert "Ancestor navigation only (not proof of applicability): PART A / STALE PARENT" in block
    assert "Content:\nActual governing sentence." in block
    assert "Content:\nPART A" not in block


def test_table_context_uses_nearest_headings_not_full_ancestor_chain():
    candidate = SimpleNamespace(
        content_kind="table",
        section_path=["TOP", "STALE PARENT", "LOCAL TABLE"],
        source_metadata={},
    )

    context = table_metadata_context(candidate)

    assert context == ["LOCAL TABLE"]
    assert "TOP" not in context


def test_broad_allowance_plan_emits_compact_policy_probe():
    q = "Types and amount of allowances as per entitlement"
    qp = build_query_plan(q)
    plan = build_deterministic_evidence_plan(q, qp)
    probes = [spec["text"].casefold() for spec in plan_query_specs(plan)]
    assert "allowance categories rates pay scale" in probes
    assert "allowance rates limits pay scale designation" in probes


def test_anaphoric_precautions_inherit_shared_scenario():
    q = (
        "A train becomes immobile between stations and passengers must be detrained. "
        "What should TO, SC and TC do, and what precautions apply?"
    )
    qp = build_query_plan(q)
    plan = build_deterministic_evidence_plan(q, qp)
    assert plan.strategy == "multi_hop"
    assert "multipart_shared_context:" in " ".join(plan.warnings)
    assert len(plan.goals) == 2
    assert all(goal.kind == "procedure" for goal in plan.goals)
    assert "immobile" in plan.goals[1].question.casefold()
    assert "detrained" in plan.goals[1].question.casefold()


def test_okf_topics_drop_conversational_function_words():
    chunks = [
        SimpleNamespace(
            section_path=[
                "When they were provided another option",
                "Travel Allowance and Hotel Reimbursement",
            ],
            content_kind="text",
            contextual_text="",
            text="",
        )
    ]
    topics = document_topic_terms(chunks)
    assert "travel" in topics
    assert "allowance" in topics
    assert "hotel" in topics
    assert "reimbursement" in topics
    for noise in ("when", "they", "were", "provided", "another"):
        assert noise not in topics


def test_primary_rolling_stock_controls_applicability_over_secondary_mentions():
    plan = build_query_plan("What restriction applies to an RS1 train?")
    wrong_primary = SimpleNamespace(
        document_profile={
            "primary_rolling_stock": ["RS-2"],
            "rolling_stock": ["RS-2", "RS-1"],
        },
        document_title="RS2 manual comparing RS1",
        section_path=[],
    )
    right_primary = SimpleNamespace(
        document_profile={
            "primary_rolling_stock": ["RS-1"],
            "rolling_stock": ["RS-1", "RS-2"],
        },
        document_title="RS1 operating manual",
        section_path=[],
    )
    assert applicability_score(plan, right_primary) > 0
    assert applicability_score(plan, wrong_primary) < 0


def test_index_document_is_demoted_for_substantive_procedure_query():
    plan = build_query_plan("What procedure should the Train Operator follow during an emergency?")
    index_candidate = SimpleNamespace(
        document_profile={},
        document_title="SC Index.pdf",
        section_path=["INDEX"],
    )
    procedure_candidate = SimpleNamespace(
        document_profile={},
        document_title="Emergency procedure.pdf",
        section_path=["Train Operator procedure"],
    )
    assert applicability_score(plan, index_candidate) < applicability_score(plan, procedure_candidate)


def test_policy_query_budget_round_robins_all_required_goals():
    q = (
        "I need to claim for a meeting I attended in Hyderabad, what all do I need to claim "
        "and mention total amount too. I left home on 18 September and came back on 28 September. "
        "Hotel was booked for the stay and I travelled by 3AC both ways."
    )
    qp, plan = _plan(q)
    specs = plan_query_specs(plan, max_queries=5)

    assert [spec["goal_ids"][0] for spec in specs] == ["g1", "g2", "g3", "g4", "g5"]
    assert specs[0]["text"] == "official tour travel entitlement applicability"
    assert specs[1]["text"] == "official tour travel lodging daily allowance local conveyance"
    assert specs[4]["text"] == "official tour daily allowance hotel duration fare calculation"


def test_employee_benefit_wording_uses_policy_entitlement_path():
    q = (
        "I worked on Sunday for 6 hours. Am I eligible for compensatory rest, "
        "conveyance or refreshment, and does it differ for executives and non-executives?"
    )
    qp, plan = _plan(q)

    assert "policy_entitlement_plan" in plan.warnings
    assert plan.strategy == "conditional"
    assert [goal.kind for goal in plan.goals] == ["applicability"]
    first_probe = plan.goals[0].search_queries[0].casefold()
    assert "compensatory rest" in first_probe
    assert "conveyance" in first_probe
    assert "refreshment" in first_probe


def test_semantic_policy_repair_keeps_deterministic_bridge_first():
    q = "All types of claims for different designations when they are on meeting away from Delhi"
    qp, fallback = _plan(q)
    payload = {
        "strategy": "enumeration",
        "entities": ["claims", "meeting", "Delhi"],
        "needs_research": True,
        "needs_verification": True,
        "goals": [
            {
                "id": "g1",
                "kind": "applicability",
                "question": "Establish claim eligibility for a meeting away from Delhi.",
                "search_queries": [
                    "claims for official meeting away from Delhi eligibility applicability",
                    "meeting outside Delhi claim eligibility conditions",
                ],
                "entity_terms": ["claims", "meeting", "Delhi"],
                "required": True,
                "depends_on": [],
                "qualifiers": ["away from Delhi"],
                "coverage_contract": "conditional_rule",
            },
            {
                "id": "g2",
                "kind": "enumeration",
                "question": "Identify claim categories by designation.",
                "search_queries": [
                    "all types of claims by designation for meeting away from Delhi",
                    "designation wise claims meeting outside Delhi",
                ],
                "entity_terms": ["claims", "meeting", "Delhi"],
                "required": True,
                "depends_on": ["g1"],
                "qualifiers": ["different designations"],
                "coverage_contract": "enumerate_set",
            },
            {
                "id": "g3",
                "kind": "attribute",
                "question": "Establish rates and limits by designation.",
                "search_queries": [
                    "claim rates limits by designation meeting away from Delhi",
                    "designation wise amount entitlement outside Delhi",
                ],
                "entity_terms": ["claims", "meeting", "Delhi"],
                "required": True,
                "depends_on": ["g1", "g2"],
                "qualifiers": ["different designations"],
                "coverage_contract": "all_requested_entities",
            },
        ],
    }

    repaired = evidence_plan_from_payload(q, payload, fallback)

    assert repaired.planner_source == "semantic"
    assert repaired.goals[0].search_queries[0] == "official tour travel entitlement applicability"
    assert repaired.goals[1].search_queries[0] == "official tour travel lodging daily allowance local conveyance"
    assert repaired.goals[2].search_queries[0] == "official tour daily allowance hotel lodging rates designation pay scale"
