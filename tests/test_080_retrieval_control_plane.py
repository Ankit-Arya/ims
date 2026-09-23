from __future__ import annotations

from pathlib import Path

from ike.retrieval.vocabulary import fuzzy_query_tokens
from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.search_plan import (
    content_tokens,
    materially_novel_query,
    relaxed_websearch_expression,
    table_retrieval_relevant,
)
from ike.retrieval.source_policy import SourcePolicy, source_name_matches, split_patterns
from ike.workflows.query_frame import deterministic_query_frame
from ike.workflows.evidence_planning import (
    EvidenceGoal,
    EvidencePlan,
    GoalSatisfaction,
    GoalStatus,
    build_deterministic_evidence_plan,
    satisfaction_from_trace,
    semantic_planner_system_prompt,
    should_use_semantic_planner,
)
from ike.workflows.query_repair import repair_queries_from_payload, repair_system_prompt


def test_normal_multiword_question_uses_fast_semantic_planning():
    question = "Who activates the cooling plant and when?"
    query_plan = build_query_plan(question)
    evidence_plan = build_deterministic_evidence_plan(question, query_plan)
    assert should_use_semantic_planner(question, query_plan, evidence_plan) is True


def test_bare_multi_identifier_lookup_keeps_deterministic_fast_path():
    question = "AX1 BX2 CX3"
    query_plan = build_query_plan(question)
    evidence_plan = build_deterministic_evidence_plan(question, query_plan)
    assert evidence_plan.strategy == "multi_lookup"
    assert should_use_semantic_planner(question, query_plan, evidence_plan) is False


def test_semantic_planner_treats_generated_terms_as_search_only_hypotheses():
    prompt = semantic_planner_system_prompt(10, 4).casefold()
    assert "search-only hypotheses" in prompt
    assert "section-heading vocabulary" in prompt
    assert "must never be treated as facts" in prompt


def test_relaxed_lexical_representation_removes_request_scaffolding_without_adding_terms():
    question = "Who activates the cooling plant and when?"
    tokens = [item.casefold() for item in content_tokens(question)]
    assert tokens == ["activates", "cooling", "plant"]
    expression = relaxed_websearch_expression(question).casefold()
    assert '"activates"' in expression
    assert '"cooling"' in expression
    assert '"plant"' in expression
    assert '"who"' not in expression
    assert '"when"' not in expression


def test_retrieval_repair_rejects_near_duplicate_and_accepts_material_strategy_change():
    attempted = ["cooling plant activation responsibility"]
    assert materially_novel_query("cooling plant activation responsibility", attempted) is False
    assert materially_novel_query("operator duties plant startup", attempted) is True


def test_table_lane_is_not_boosted_for_plain_responsibility_question():
    assert table_retrieval_relevant(
        goal_kinds=["responsibility"],
        facets=[],
        question="Who starts the cooling plant?",
    ) is False
    assert table_retrieval_relevant(
        goal_kinds=["temporal"],
        facets=[],
        question="When does the cooling plant start?",
    ) is False
    assert table_retrieval_relevant(
        goal_kinds=["enumeration"],
        facets=[],
        question="List all cooling plant limits",
    ) is True


def test_source_policy_patterns_are_configuration_driven():
    assert split_patterns("Rulebook; Safety Manual, Standard") == [
        "Rulebook",
        "Safety Manual",
        "Standard",
    ]
    assert source_name_matches("Operations Rulebook 2026", "ops.pdf", "rulebook") is True
    assert source_name_matches("Local Layout", "layout.pdf", "rulebook") is False
    policy = SourcePolicy(priority_patterns=["Rulebook"])
    assert policy.has_priority_stage is False
    assert policy.as_dict()["priority_patterns"] == ["Rulebook"]


def test_coverage_contracts_do_not_claim_complete_from_candidate_presence_only():
    plan = EvidencePlan(
        original="Define component alpha",
        goals=[
            EvidenceGoal(
                id="g1",
                kind="definition",
                question="Define component alpha",
                search_queries=["component alpha"],
                entity_terms=["component alpha"],
                required=True,
            )
        ],
    )
    assert plan.goals[0].coverage_contract == "all_supported_variants"
    result = satisfaction_from_trace(
        plan,
        {
            "goal_stats": {
                "g1": {
                    "evidence_count": 1,
                    "lexical_like_evidence_count": 1,
                    "max_rerank_score": 0.8,
                }
            }
        },
    )
    assert result.complete is False
    assert result.partial_goal_ids == ["g1"]


def test_repair_controller_allows_conceptual_search_terms_but_blocks_invented_identifier_and_number():
    question = "Who starts the cooling plant?"
    plan = EvidencePlan(
        original=question,
        goals=[
            EvidenceGoal(
                id="g1",
                kind="responsibility",
                question="Establish who starts the cooling plant.",
                search_queries=["cooling plant starts"],
                entity_terms=["cooling plant"],
                required=True,
            )
        ],
        requires_decomposition=True,
    )
    satisfaction = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="missing")],
        missing_goal_ids=["g1"],
    )
    payload = {
        "goals": [
            {
                "goal_id": "g1",
                "queries": [
                    "operator duties cooling plant startup",
                    "ZX99 cooling plant startup",
                    "cooling plant startup 42 minutes",
                ],
            }
        ]
    }
    repaired = repair_queries_from_payload(
        payload,
        question=question,
        plan=plan,
        satisfaction=satisfaction,
        attempted_queries=["cooling plant starts"],
        max_queries_per_goal=4,
    )
    assert repaired == {"g1": ["operator duties cooling plant startup"]}
    prompt = repair_system_prompt(4).casefold()
    assert "change the retrieval relation" in prompt
    assert "search-only" in prompt


def test_recovery_probes_are_admitted_before_seed_queries_when_base_is_suppressed():
    source = Path("src/ike/retrieval/engine.py").read_text()
    repair_guard = "if goal_expansions and not include_base_query:"
    seed_loop = "for spec in goal_specs:"
    assert repair_guard in source
    assert source.index(repair_guard) < source.index(seed_loop)
    assert "only changed retrieval strategy" in source


def test_governing_reference_cannot_short_circuit_global_retrieval():
    graph_source = Path("src/ike/workflows/qa_graph.py").read_text()
    engine_source = Path("src/ike/retrieval/engine.py").read_text()
    assert 'graph.add_node("priority_search", self._priority_search)' not in graph_source
    assert '"priority": "priority_search"' not in graph_source
    assert "stopped after the bounded priority-source probe" not in graph_source
    assert "_governing_document_ids" in engine_source
    assert 'candidate.evidence_lane = "governing"' in engine_source
    assert 'source_stage="priority"' not in graph_source


def test_routing_is_boost_not_hidden_hard_scope():
    source = Path("src/ike/workflows/qa_graph.py").read_text()
    assert 'hard_scope_ids = state.get("document_ids")' in source
    assert 'boost_document_ids=boost_ids' in source
    assert 'state.get("document_ids") or (routed_ids' not in source


def test_query_frame_preserves_original_query_and_classifies_generic_scalar():
    question = "Max speed at which train can operate"
    plan = build_query_plan(question)
    frame = deterministic_query_frame(question, plan)
    assert frame.original == question
    assert frame.retrieval_hypotheses()[0] == question
    assert frame.answer_shape == "scalar"


def test_generic_superlative_can_activate_table_lane_without_domain_keyword_rule():
    assert table_retrieval_relevant(
        goal_kinds=["fact"],
        facets=[],
        question="What is the maximum permitted value?",
    ) is True


def test_production_retrieval_control_plane_contains_no_regression_answers():
    files = [
        Path("src/ike/retrieval/search_plan.py"),
        Path("src/ike/retrieval/source_policy.py"),
        Path("src/ike/workflows/query_repair.py"),
        Path("src/ike/workflows/evidence_planning.py"),
        Path("src/ike/workflows/qa_graph.py"),
        Path("src/ike/retrieval/engine.py"),
    ]
    production = "\n".join(path.read_text().casefold() for path in files)
    forbidden = [
        "open the station at least ten minutes",
        "battery isolation contactor",
        "brake isolating cock",
        "bogie isolation cock",
        "kashmere gate",
        "shahdara",
    ]
    assert all(value not in production for value in forbidden)


def test_scalar_lookup_does_not_pay_semantic_planning_tax():
    question = "Max speed at which train can operate"
    query_plan = build_query_plan(question)
    evidence_plan = build_deterministic_evidence_plan(question, query_plan)
    assert evidence_plan.strategy == "single"
    assert should_use_semantic_planner(question, query_plan, evidence_plan) is False


def test_corpus_fuzzy_tokens_protect_technical_identifiers():
    tokens = fuzzy_query_tokens("wakeup prosses ATP RS-10 PSD/ATO", limit=8)
    assert "wakeup" in tokens
    assert "prosses" in tokens
    assert "atp" not in tokens
    assert all("rs" not in token for token in tokens)
    assert "psd" not in tokens
    assert "ato" not in tokens


def test_retrieval_engine_preserves_exact_q0_before_expansions():
    source = Path("src/ike/retrieval/engine.py").read_text()
    original = 'add_raw_query(question, origin="original")'
    semantic = 'for value in query_plan.semantic_queries:'
    assert original in source
    assert semantic in source
    assert source.index(original) < source.index(semantic)


def test_speculative_fanout_flag_is_safe_until_parallel_sessions_are_implemented():
    source = Path("src/ike/core/config.py").read_text()
    assert "speculative_retrieval_enabled: bool = False" in source


def test_query_frame_marks_confirmation_premise_unverified():
    question = "After six months absence only two trips are required, right?"
    plan = build_query_plan(question)
    frame = deterministic_query_frame(question, plan)
    assert len(frame.premise_claims) == 1
    assert frame.premise_claims[0].status == "unverified"
    assert frame.premise_claims[0].text == question
