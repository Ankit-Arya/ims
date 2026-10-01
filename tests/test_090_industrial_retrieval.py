from uuid import uuid4

from ike.retrieval.context_assembly import assemble_draft_context
from ike.retrieval.evidence_ledger import monotonic_merge
from ike.retrieval.types import Candidate, Evidence
from ike.workflows.evidence_planning import (
    EvidenceGoal,
    EvidencePlan,
    GoalSatisfaction,
    GoalStatus,
    build_deterministic_evidence_plan,
)
from ike.retrieval.query_plan import build_query_plan
from ike.workflows.query_understanding import classify_retrieval_effort


def _evidence(i: int, *, goal: str = "g1", doc=None, score: float = 0.5) -> Evidence:
    candidate = Candidate(
        chunk_id=uuid4(),
        document_id=doc or uuid4(),
        ordinal=i,
        page_from=i,
        page_to=i,
        section_path=["Section", str(i)],
        content_kind="text",
        text=f"evidence {i}",
        contextual_text=f"evidence {i}",
        document_title=f"Document {i}",
        filename=f"document-{i}.pdf",
        revision=None,
        authority=None,
        rerank_score=score,
        sources={f"goal:{goal}", f"goal:{goal}:lexical"},
    )
    return Evidence(evidence_id=f"E{i}", candidate=candidate)


def test_plain_term_uses_primary_definition_and_fast_effort():
    question = "drunkenness"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.goals[0].coverage_contract == "primary_definition"
    effort = classify_retrieval_effort(question, query_plan, plan)
    assert effort.level == "fast"


def test_uppercase_identifier_keeps_variant_discovery():
    question = "BIC"
    query_plan = build_query_plan(question)
    plan = build_deterministic_evidence_plan(question, query_plan)
    assert plan.goals[0].coverage_contract == "all_supported_variants"
    assert classify_retrieval_effort(question, query_plan, plan).level == "research"


def test_enumeration_routes_to_research():
    plan = EvidencePlan(
        original="list all responsibilities",
        strategy="overview",
        goals=[EvidenceGoal(id="g1", kind="responsibility", question="all responsibilities")],
        needs_research=True,
    )
    query_plan = build_query_plan("all responsibilities of controller")
    effort = classify_retrieval_effort("all responsibilities of controller", query_plan, plan)
    assert effort.level == "research"


def test_single_bounded_all_requested_entities_goal_is_focused_not_full_research():
    question = "Rate according to designation"
    query_plan = build_query_plan(question)
    plan = EvidencePlan(
        original=question,
        strategy="single",
        goals=[
            EvidenceGoal(
                id="g1",
                kind="attribute",
                question="Establish the rate according to designation.",
                coverage_contract="all_requested_entities",
                required=True,
            )
        ],
    )
    assert query_plan.coverage_sensitive is False
    assert classify_retrieval_effort(question, query_plan, plan).level == "focused"


def test_monotonic_merge_protects_prior_supported_evidence():
    primary = [_evidence(i) for i in range(1, 6)]
    recovered = [_evidence(i + 10) for i in range(1, 6)]
    satisfaction = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="partial", evidence_ids=["E1", "E2"])],
        partial_goal_ids=["g1"],
    )
    result = monotonic_merge(primary, recovered, limit=5, satisfaction=satisfaction)
    chunk_ids = {item.candidate.chunk_id for item in result.evidence}
    assert primary[0].candidate.chunk_id in chunk_ids
    assert primary[1].candidate.chunk_id in chunk_ids



def test_monotonic_merge_preserves_primary_evidence_ids_and_allocates_recovery_ids():
    primary = [_evidence(1), _evidence(2)]
    recovered = [_evidence(20), _evidence(21)]
    satisfaction = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="partial", evidence_ids=["E2"])],
        partial_goal_ids=["g1"],
    )
    result = monotonic_merge(primary, recovered, limit=4, satisfaction=satisfaction)
    by_chunk = {item.candidate.chunk_id: item.evidence_id for item in result.evidence}
    assert by_chunk[primary[0].candidate.chunk_id] == "E1"
    assert by_chunk[primary[1].candidate.chunk_id] == "E2"
    recovered_ids = {by_chunk[item.candidate.chunk_id] for item in recovered}
    assert recovered_ids == {"E3", "E4"}


def test_monotonic_merge_admits_targeted_recovery_even_when_primary_is_full():
    primary = [_evidence(i, score=0.2) for i in range(1, 6)]
    primary[0].candidate.rerank_score = 0.9
    recovered = [_evidence(20, score=0.95), _evidence(21, score=0.8)]
    satisfaction = GoalSatisfaction(
        complete=False,
        statuses=[GoalStatus(goal_id="g1", status="partial", evidence_ids=["E1"])],
        partial_goal_ids=["g1"],
    )
    result = monotonic_merge(primary, recovered, limit=5, satisfaction=satisfaction)
    chunk_ids = {item.candidate.chunk_id for item in result.evidence}
    assert primary[0].candidate.chunk_id in chunk_ids
    assert recovered[0].candidate.chunk_id in chunk_ids
    assert recovered[1].candidate.chunk_id in chunk_ids

def test_draft_context_is_goal_balanced_and_keeps_ledger_ids(monkeypatch):
    evidence = []
    doc = uuid4()
    for index in range(1, 7):
        evidence.append(_evidence(index, goal="g1", doc=doc, score=1 - index / 20))
    for index in range(7, 11):
        evidence.append(_evidence(index, goal="g2", score=0.3))
    plan = EvidencePlan(
        original="compare",
        strategy="comparison",
        goals=[
            EvidenceGoal(id="g1", kind="fact", question="left"),
            EvidenceGoal(id="g2", kind="fact", question="right"),
        ],
        requires_decomposition=True,
    )
    draft, trace = assemble_draft_context(evidence, effort="focused", plan=plan, satisfaction=None)
    assert any("goal:g1" in item.candidate.sources for item in draft)
    assert any("goal:g2" in item.candidate.sources for item in draft)
    assert {item.evidence_id for item in draft}.issubset({item.evidence_id for item in evidence})
    assert trace["draft_context_count"] == len(draft)
    assert trace["draft_context_token_estimate"] > 0
