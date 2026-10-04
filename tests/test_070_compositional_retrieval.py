from __future__ import annotations

from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import (
    EvidenceGoal,
    EvidencePlan,
    GoalSatisfaction,
    GoalStatus,
    build_deterministic_evidence_plan,
    evidence_plan_from_payload,
    goal_satisfaction_from_payload,
    plan_query_specs,
    recovery_queries_for_goals,
    should_use_semantic_planner,
    standalone_identifiers,
)


def evidence_plan(question: str) -> EvidencePlan:
    return build_deterministic_evidence_plan(question, build_query_plan(question))


def test_bare_multiple_identifiers_are_independent_required_goals():
    plan = evidence_plan("BIC TCMS TIMS")
    assert plan.strategy == "multi_lookup"
    assert plan.entities == ["BIC", "TCMS", "TIMS"]
    assert [goal.entity_terms for goal in plan.required_goals] == [["BIC"], ["TCMS"], ["TIMS"]]
    assert all(goal.kind == "definition" for goal in plan.required_goals)
    specs = plan_query_specs(plan, max_queries=14)
    assert [item["text"] for item in specs[:3]] == ["BIC", "TCMS", "TIMS"]
    assert [item["goal_ids"] for item in specs[:3]] == [["g1"], ["g2"], ["g3"]]


def test_definition_cue_does_not_collapse_three_identifiers_into_one_entity():
    plan = evidence_plan("What is BIC, TCMS and TIMS?")
    assert plan.entities == ["BIC", "TCMS", "TIMS"]
    assert len(plan.required_goals) == 3


def test_slash_separated_identifiers_are_split_without_splitting_hyphenated_codes():
    assert standalone_identifiers("Compare ATO/ATP with RS-10") == ["ATO", "ATP", "RS-10"]


def test_explicit_relationship_has_independent_entities_and_relationship_goal():
    plan = evidence_plan("Compare AX1 BX2 CX3")
    assert plan.strategy == "comparison"
    relation = [goal for goal in plan.goals if goal.kind == "comparison"]
    assert len(relation) == 1
    assert relation[0].required is True
    assert set(relation[0].entity_terms) == {"AX1", "BX2", "CX3"}
    assert any(query == "AX1 BX2" for query in relation[0].search_queries)
    # Definitions are useful context but do not themselves prove the comparison.
    assert all(not goal.required for goal in plan.goals if goal.kind == "definition")


def test_multi_facet_entity_creates_independent_requirements():
    plan = evidence_plan("What is cooling unit and where is it located?")
    assert plan.requires_decomposition is True
    assert {goal.kind for goal in plan.required_goals} == {"definition", "attribute"}
    assert all(goal.entity_terms == ["cooling unit"] for goal in plan.required_goals)


def test_conditional_procedure_is_not_reduced_to_plain_procedure_in_fallback():
    question = "If the sensor fails then the backup valve closes and the pump trips, what procedure applies?"
    plan = evidence_plan(question)
    assert plan.strategy == "conditional"
    assert plan.requires_decomposition is True
    assert plan.needs_verification is True
    assert len(plan.required_goals) >= 2
    assert should_use_semantic_planner(question, build_query_plan(question), plan) is True


def test_rich_multi_identifier_condition_uses_semantic_planner_but_keeps_identifier_seeds():
    question = "If AX1 fails while BX2 is active, what should CX3 indicate and why?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert should_use_semantic_planner(question, query_plan, plan) is True
    assert {term for goal in plan.goals for term in goal.entity_terms} >= {"AX1", "BX2", "CX3"}
    assert any(goal.required and goal.kind in {"condition", "consequence", "scenario"} for goal in plan.goals)


def test_yes_no_or_possible_false_premise_is_semantically_planned_and_verified():
    question = "Is the emergency limit always 20 km/h?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.strategy == "claim_check"
    assert plan.needs_verification is True
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_terse_unstructured_multi_term_input_can_use_semantic_planner():
    question = "pension gratuity eligibility"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_lowercase_single_term_is_still_an_atomic_term_goal():
    plan = evidence_plan("overtime")
    assert plan.strategy == "single"
    assert plan.goals[0].kind == "definition"
    assert plan.goals[0].entity_terms == ["overtime"]


def test_primary_semantic_plan_replaces_wrong_deterministic_constraint_with_structure_goal():
    question = "How many parts are there in RULEBOOK and name them"
    fallback = evidence_plan(question)
    assert any(goal.kind == "attribute" for goal in fallback.goals)

    payload = {
        "interpretation": "Count and name the top-level parts in the named rulebook.",
        "answer_shape": "enumeration",
        "source_hints": ["RULEBOOK"],
        "strategy": "enumeration",
        "entities": ["RULEBOOK"],
        "needs_research": True,
        "needs_verification": True,
        "goals": [
            {
                "id": "g1",
                "kind": "enumeration",
                "question": "Identify the complete top-level part structure in RULEBOOK.",
                "search_queries": ["RULEBOOK parts", "RULEBOOK contents headings"],
                "entity_terms": ["RULEBOOK"],
                "required": True,
                "coverage_contract": "enumerate_set",
                "retrieval_tools": ["document_structure"],
            }
        ],
    }
    plan = evidence_plan_from_payload(
        question,
        payload,
        fallback,
        preserve_fallback_semantics=False,
    )

    assert plan.planner_source == "semantic"
    assert plan.answer_shape == "enumeration"
    assert plan.source_hints == ["RULEBOOK"]
    assert len(plan.required_goals) == 1
    assert plan.required_goals[0].kind == "enumeration"
    assert plan.required_goals[0].coverage_contract == "enumerate_set"
    assert plan.required_goals[0].retrieval_tools == ["document_structure"]
    assert all(goal.kind != "attribute" for goal in plan.required_goals)


def test_primary_semantic_plan_preserves_source_section_relationship_instead_of_identifier_split():
    question = "Show part VII of RULEBOOK"
    fallback = evidence_plan(question)
    payload = {
        "interpretation": "Show the requested part inside the named source.",
        "answer_shape": "synthesis",
        "source_hints": ["RULEBOOK"],
        "strategy": "single",
        "entities": ["part VII", "RULEBOOK"],
        "needs_research": True,
        "needs_verification": True,
        "goals": [
            {
                "id": "g1",
                "kind": "overview",
                "question": "Retrieve and explain part VII from RULEBOOK.",
                "search_queries": ["part VII RULEBOOK", "RULEBOOK part VII"],
                "entity_terms": ["part VII", "RULEBOOK"],
                "required": True,
                "relation": "part_of_source",
                "coverage_contract": "enumerate_set",
                "retrieval_tools": ["section_navigation"],
            }
        ],
    }
    plan = evidence_plan_from_payload(
        question,
        payload,
        fallback,
        preserve_fallback_semantics=False,
    )

    assert len(plan.required_goals) == 1
    assert plan.required_goals[0].relation == "part_of_source"
    assert plan.required_goals[0].retrieval_tools == ["section_navigation"]
    assert plan.source_hints == ["RULEBOOK"]


def test_semantic_planner_cannot_introduce_new_identifier_or_number_in_search_query():
    question = "If the primary pump stops then the backup system starts; what action is required?"
    fallback = evidence_plan(question)
    payload = {
        "strategy": "conditional",
        "needs_research": True,
        "needs_verification": True,
        "entities": ["primary pump", "backup system"],
        "goals": [
            {
                "id": "g1",
                "kind": "condition",
                "question": "Establish what happens when the primary pump stops.",
                "search_queries": ["primary pump stops", "ZX99 activates at 42%"],
                "entity_terms": ["primary pump"],
                "required": True,
            }
        ],
    }
    plan = evidence_plan_from_payload(question, payload, fallback)
    all_queries = [query for goal in plan.goals for query in goal.search_queries]
    assert all("ZX99" not in query for query in all_queries)
    assert all("42" not in query for query in all_queries)


def test_semantic_planner_omission_does_not_silently_drop_user_clause():
    question = "sensor fails then backup valve closes then pump trips; what action applies"
    fallback = evidence_plan(question)
    assert len(fallback.goals) >= 3
    payload = {
        "strategy": "multi_hop",
        "goals": [
            {
                "id": "g1",
                "kind": "condition",
                "question": "Establish evidence for the sensor failure.",
                "search_queries": ["sensor fails"],
                "entity_terms": ["sensor"],
                "required": True,
            }
        ],
    }
    plan = evidence_plan_from_payload(question, payload, fallback, max_goals=8)
    joined = " ".join(query for goal in plan.goals for query in goal.search_queries).casefold()
    assert "backup valve" in joined
    assert "pump trips" in joined
    assert "preserved_omitted_user_clause_goal" in plan.warnings


def test_goal_audit_forces_replan_when_evidence_supports_the_wrong_plan():
    plan = EvidencePlan(
        original="Explain the requested section in RULEBOOK",
        interpretation="Define RULEBOOK",
        goals=[EvidenceGoal(
            id="g1", kind="definition", question="Define RULEBOOK",
            search_queries=["RULEBOOK definition"], entity_terms=["RULEBOOK"], required=True,
        )],
        needs_verification=True,
        planner_source="semantic",
    )
    fallback = GoalSatisfaction(
        complete=True,
        statuses=[GoalStatus(goal_id="g1", status="supported", evidence_ids=["E1"])],
        audit_source="deterministic",
    )
    result = goal_satisfaction_from_payload(
        plan,
        {
            "plan_aligned": False,
            "plan_issue": "The user asked for content inside a section, not a source definition.",
            "replan_needed": True,
            "goals": [{
                "goal_id": "g1", "status": "supported", "evidence_ids": ["E1"],
                "reason": "The evidence defines the source.", "recovery_queries": [],
            }],
        },
        valid_evidence_ids={"E1"},
        valid_evidence_ids_by_goal={"g1": {"E1"}},
        fallback=fallback,
    )

    assert result.complete is False
    assert result.plan_aligned is False
    assert result.replan_needed is True
    assert result.partial_goal_ids == ["g1"]
    assert result.statuses[0].status == "partial"


def test_goal_audit_cannot_mark_supported_without_real_evidence_id():
    plan = EvidencePlan(
        original="Compare unit alpha and unit beta",
        strategy="comparison",
        goals=[
            EvidenceGoal(
                id="g1",
                kind="comparison",
                question="Establish the documented comparison.",
                search_queries=["unit alpha unit beta"],
                required=True,
            )
        ],
        requires_decomposition=True,
        needs_research=True,
        needs_verification=True,
    )
    fallback = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="partial")],
        partial_goal_ids=["g1"],
    )
    result = goal_satisfaction_from_payload(
        plan,
        {"goals": [{"goal_id": "g1", "status": "supported", "evidence_ids": []}]},
        valid_evidence_ids={"E1"},
        fallback=fallback,
    )
    assert result.complete is False
    assert result.partial_goal_ids == ["g1"]


def test_recovery_search_prefers_goal_queries_before_model_suggestions():
    plan = EvidencePlan(
        original="unit alpha status",
        goals=[
            EvidenceGoal(
                id="g1",
                kind="attribute",
                question="Establish unit alpha status.",
                search_queries=["unit alpha status"],
                entity_terms=["unit alpha"],
            )
        ],
        requires_decomposition=True,
    )
    satisfaction = GoalSatisfaction(
        complete=False,
        statuses=[
            GoalStatus(
                goal_id="g1",
                status="partial",
                recovery_queries=["model suggested alternative wording"],
            )
        ],
        partial_goal_ids=["g1"],
    )
    queries = recovery_queries_for_goals(plan, satisfaction)["g1"]
    assert queries[0] == "unit alpha status"


def test_compositional_logic_is_not_encoded_around_regression_answers():
    # Regression examples may exist in tests/docs, but the new production planner must not
    # contain expected expansions, locations, compensation rules or other target answers.
    from pathlib import Path

    production = Path("src/ike/workflows/evidence_planning.py").read_text().casefold()
    forbidden_expected_answers = [
        "battery isolation contactor",
        "brake isolating cock",
        "bogie isolation cock",
        "train control and monitoring system",
        "train integrated management system",
        "shahdara",
        "kashmere gate",
    ]
    assert all(value not in production for value in forbidden_expected_answers)


def test_confirmation_suffix_is_a_claim_check_not_an_asserted_fact():
    question = "There is no backup mode, correct?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.strategy == "claim_check"
    assert plan.needs_research is True
    assert plan.needs_verification is True
    assert any(goal.kind == "claim_check" and goal.required for goal in plan.goals)


def test_conflicting_sources_require_precedence_evidence_not_fake_enumeration():
    question = "Two manuals give different limits for unit alpha; which applies?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert "which applies" not in [value.casefold() for value in query_plan.entity_terms]
    assert plan.strategy == "relationship"
    assert plan.needs_research is True
    assert plan.needs_verification is True
    assert any(goal.relation == "conflict_or_precedence" for goal in plan.required_goals)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_explicit_relationship_cannot_be_lost_behind_definition_shortcut():
    question = "What is pump and how does it relate to valve?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.strategy == "relationship"
    assert plan.needs_verification is True
    assert any(goal.kind == "relationship" and goal.required for goal in plan.goals)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_single_identifier_is_preserved_inside_a_complex_question():
    question = "What is ZX99 and why does it fail?"
    plan = evidence_plan(question)
    assert plan.entities == ["ZX99"]
    assert any("ZX99" in goal.entity_terms for goal in plan.goals)
    assert plan.needs_research is True
    assert plan.needs_verification is True


def test_definition_with_context_word_keeps_core_term():
    plan = evidence_plan("What does isolation mean here?")
    assert plan.goals[0].kind == "definition"
    assert plan.goals[0].entity_terms == ["ISOLATION"]
    assert any(query.casefold() == "isolation" for query in plan.goals[0].search_queries)


def test_negative_document_inventory_is_never_treated_as_normal_top_k_absence():
    question = "Which documents do not mention unit alpha?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.needs_research is True
    assert plan.needs_verification is True
    assert "corpus_negative_requires_exhaustive_inventory" in plan.warnings
    assert any("corpus_negative_requires_exhaustive_inventory" in goal.qualifiers for goal in plan.required_goals)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_source_scoped_indirect_question_can_use_semantic_planning():
    question = "According to the selected document, what is the limit?"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_unicode_single_term_is_preserved_as_an_atomic_goal():
    plan = evidence_plan("ओवरटाइम")
    assert plan.strategy == "single"
    assert plan.goals[0].kind == "definition"
    assert plan.goals[0].entity_terms == ["ओवरटाइम"]


def test_unicode_multiword_question_can_use_semantic_planner():
    question = "ओवरटाइम की पात्रता क्या है"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert should_use_semantic_planner(question, query_plan, plan) is True


def test_goal_auditor_cannot_borrow_evidence_from_another_goal():
    plan = EvidencePlan(
        original="alpha beta",
        strategy="multi_entity",
        goals=[
            EvidenceGoal(id="g1", kind="fact", question="alpha", search_queries=["alpha"], required=True),
            EvidenceGoal(id="g2", kind="fact", question="beta", search_queries=["beta"], required=True),
        ],
        requires_decomposition=True,
        needs_research=True,
        needs_verification=True,
    )
    fallback = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="supported"), GoalStatus(goal_id="g2", status="missing")],
        missing_goal_ids=["g2"],
    )
    result = goal_satisfaction_from_payload(
        plan,
        {
            "goals": [
                {"goal_id": "g1", "status": "supported", "evidence_ids": ["E1"]},
                {"goal_id": "g2", "status": "supported", "evidence_ids": ["E1"]},
            ]
        },
        valid_evidence_ids={"E1"},
        valid_evidence_ids_by_goal={"g1": {"E1"}, "g2": set()},
        fallback=fallback,
    )
    assert result.complete is False
    assert result.partial_goal_ids == ["g2"]
