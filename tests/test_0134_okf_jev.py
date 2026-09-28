from types import SimpleNamespace
from uuid import uuid4

import httpx

from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.types import Candidate
from ike.workflows.evidence_planning import build_deterministic_evidence_plan
from ike.services.jev import JevEvidenceJudge
from ike.services.okf import render_document_concept


def _candidate(index: int, score: float) -> Candidate:
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=index,
        page_from=1,
        page_to=1,
        section_path=["Procedure"],
        content_kind="text",
        text=f"passage {index}",
        contextual_text=f"passage {index}",
        document_title="Manual",
        filename="manual.pdf",
        revision="1",
        authority="Ops",
        final_retrieval_score=score,
        rank_method="hybrid_fusion",
    )


def _settings(mode: str):
    return SimpleNamespace(
        jev_mode=mode,
        jev_provider="typesafe",
        typesafe_api_key="test",
        jev_base_url="https://api.typesafe.ai",
        jev_model="jev-latest",
        jev_timeout_seconds=1.0,
        jev_max_candidates=2,
        jev_candidate_chars=1000,
        jev_weight=0.9,
        jev_fail_open=True,
        jev_complex_enabled=True,
        jev_complex_candidate_chars=800,
        jev_max_goal_pairs=8,
        jev_max_goals_per_candidate=2,
    )


def _client():
    def handler(request):
        import json
        payload = json.loads(request.content.decode("utf-8"))
        answers = {}
        for key in payload["questions"]:
            score = 0.2 if key.endswith("_0") else 3.8
            answers[key] = {"type": "score", "score": score, "confidence": 0.9}
        return httpx.Response(200, json={"model": "jev-latest", "answers": answers})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_jev_apply_reorders_with_typed_judgment():
    candidates = [_candidate(0, 0.9), _candidate(1, 0.8)]
    client = _client()
    ranked, trace = JevEvidenceJudge(_settings("apply"), client).judge_and_apply("question", candidates)
    assert trace["called"] is True
    assert ranked[0].ordinal == 1
    assert ranked[0].judge_score > ranked[1].judge_score
    client.close()


def test_jev_shadow_does_not_reorder():
    candidates = [_candidate(0, 0.9), _candidate(1, 0.8)]
    client = _client()
    ranked, _ = JevEvidenceJudge(_settings("shadow"), client).judge_and_apply("question", candidates)
    assert [c.ordinal for c in ranked] == [0, 1]
    assert "jev_shadow" in ranked[0].sources
    client.close()


def test_okf_concept_has_required_v02_shape():
    document = SimpleNamespace(
        id=uuid4(), title="Operations Manual", original_filename="ops.pdf", checksum_sha256="a" * 64,
        source_role="operating_procedure", family_key="ops", lifecycle_status="active",
        authority="Operations", revision="5", effective_from=None, effective_to=None, page_count=10,
    )
    chunk = SimpleNamespace(
        ordinal=0, page_from=1, page_to=1, section_path=["Emergency"], content_kind="text",
        text="x", contextual_text="x", metadata={},
    )
    text = render_document_concept(
        document,
        [chunk],
        {"document_type": "operating_manual", "primary_line_codes": ["Line 7"]},
    )
    assert text.startswith("---\n")
    assert 'type: "IMS Document Knowledge"' in text
    assert "sources:" in text
    assert "generated:" in text
    assert 'resource: "ims://documents/' in text


def test_jev_complex_mode_scores_atomic_goals_without_reordering_shadow():
    c1 = _candidate(0, 0.9)
    c2 = _candidate(1, 0.8)
    c1.sources.add("goal:g1")
    c2.sources.add("goal:g2")

    g1 = SimpleNamespace(
        id="g1", question="What is the flood-entry condition?", kind="condition",
        required=True, qualifiers=[], coverage_contract="conditional_rule",
    )
    g2 = SimpleNamespace(
        id="g2", question="What speed restriction applies during flooding?", kind="procedure",
        required=True, qualifiers=[], coverage_contract="ordered_procedure",
    )
    plan = SimpleNamespace(goals=[g1, g2], requires_decomposition=True)

    def handler(request):
        import json
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["state"]["complex_query"] is True
        assert len(payload["state"]["evidence_goals"]) == 2
        assert len(payload["questions"]) == 2
        answers = {}
        for key in payload["questions"]:
            score = 3.8 if "0_g1" in key else 3.4
            answers[key] = {"type": "score", "score": score, "confidence": 0.95}
        return httpx.Response(200, json={"model": "jev-latest", "answers": answers})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ranked, trace = JevEvidenceJudge(_settings("shadow"), client).judge_and_apply(
        "Explain flood operation conditions and restrictions",
        [c1, c2],
        evidence_plan=plan,
    )
    assert [c.ordinal for c in ranked] == [0, 1]
    assert trace["complex_mode"] is True
    assert trace["goal_count"] == 2
    assert trace["goal_pair_count"] == 2
    assert trace["question_count"] == 2
    assert trace["judgments"][0]["goal_scores"]["g1"] > 0.9
    assert trace["judgments"][1]["goal_scores"]["g2"] > 0.8
    assert c1.goal_rerank_scores == {}
    assert c2.goal_rerank_scores == {}
    client.close()


def _plan(question: str):
    qp = build_query_plan(question)
    return build_deterministic_evidence_plan(question, qp)


def test_multipart_setup_clause_becomes_shared_context_not_goal():
    plan = _plan(
        "If an axle counter section fails, when can it be reset, who is authorised to reset it, "
        "what checks are required, and what should be done if reset is unsuccessful?"
    )
    assert plan.strategy == "multi_hop"
    assert plan.requires_decomposition is True
    assert len(plan.goals) == 4
    assert plan.warnings == ["multipart_shared_context:an axle counter section fails"]
    assert all(
        "shared_context:an axle counter section fails" in goal.qualifiers
        for goal in plan.goals
    )
    assert all(not goal.question.casefold().startswith("if an axle") for goal in plan.goals)


def test_separately_forces_independent_subqueries():
    plan = _plan(
        "What should TO and SC do during flooding, and separately what should they do "
        "during detrainment of an immobile train?"
    )
    assert plan.strategy == "multi_lookup"
    assert plan.warnings == ["multipart_independent_subqueries"]
    assert len(plan.goals) == 2
    assert all(goal.qualifiers == ["independent_subquery"] for goal in plan.goals)


def test_independent_subqueries_keep_entities_local():
    plan = _plan("What is BIC, what does MRGR say about alcohol, and how is an axle counter reset?")
    assert plan.strategy == "multi_lookup"
    assert [goal.entity_terms for goal in plan.goals] == [["BIC"], ["MRGR"], []]
    assert all(
        "What is BIC, what does MRGR" not in " ".join(goal.search_queries)
        for goal in plan.goals
    )


def test_constrained_attribute_not_rewritten_as_identifier_definitions():
    plan = _plan("What is the flooding speed limit for RS17 in ATO mode?")
    assert plan.strategy == "conditional"
    assert plan.requires_decomposition is False
    assert len(plan.goals) == 1
    assert plan.goals[0].kind == "attribute"
    assert "flooding speed limit" in plan.goals[0].question.casefold()
    assert "rolling_stock:RS-17" in plan.goals[0].qualifiers
    assert plan.warnings == ["constrained_attribute_query"]


def test_source_setup_is_inherited_context():
    plan = _plan(
        "As per MRGR 2020, what are the duties of a Train Operator before starting, "
        "while running, and after stopping a train?"
    )
    assert plan.strategy == "multi_hop"
    assert len(plan.goals) == 1
    assert "shared_context:MRGR 2020" in plan.goals[0].qualifiers
    assert "inherited_context:MRGR 2020" in plan.goals[0].qualifiers


def test_plural_technical_identifier_is_preserved_for_shared_fault_context():
    plan = _plan(
        "If 3 BICs of an RS1 train are isolated, what operating restrictions apply, "
        "can the train remain in service, and what action must the TO take?"
    )
    assert plan.strategy == "multi_hop"
    assert all("BIC" in goal.entity_terms for goal in plan.goals)
    assert all("RS1" in goal.entity_terms for goal in plan.goals)
    assert all("BIC RS1" in goal.search_queries for goal in plan.goals)
