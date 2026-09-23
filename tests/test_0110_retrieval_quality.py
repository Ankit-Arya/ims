from uuid import uuid4

from ike.retrieval.normalization import canonical_identifier, technical_identifier_variants
from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.types import Candidate, Evidence
from ike.workflows.routing import extract_lookup_term, is_coverage_question
from ike.workflows.verification_policy import retrieval_quality_issues


def evidence(text: str, *, title: str = "Manual.pdf", section: str = "Section") -> Evidence:
    return Evidence(
        evidence_id="E1",
        candidate=Candidate(
            chunk_id=uuid4(), document_id=uuid4(), ordinal=1,
            page_from=1, page_to=1, section_path=[section], content_kind="text",
            text=text, contextual_text=text, document_title=title, filename=title,
            revision=None, authority=None, rerank_score=0.8,
        ),
    )


def test_sc7a_identifier_variants_cover_canonical_document_spelling():
    variants = {value.casefold() for value in technical_identifier_variants("Sc 7a")}
    assert "sc-07a" in variants
    assert "sc07a" in variants
    assert canonical_identifier("Sc 7a") == "SC-07A"
    assert extract_lookup_term("Sc 7a") == "SC 7A"
    plan = build_query_plan("Sc 7a")
    assert "SC-07A" in plan.exact_terms


def test_how_to_and_action_required_are_procedure_coverage_queries():
    assert is_coverage_question("how to deal in high speed wind conditions")
    assert is_coverage_question("what action required if fire at concourse level")


def test_line_scope_mismatch_forces_recovery():
    q = "I'm on Line-7, wind speed is 65, please give instruction"
    plan = build_query_plan(q)
    ev = evidence("Line-8 high wind procedure. Wind speed 65 kmph.", title="Line 8 wind.pdf")
    issues = retrieval_quality_issues(q, [ev], {"coverage_complete": True}, plan)
    assert "explicit_line_scope_not_represented" in issues


def test_scalar_request_without_number_forces_recovery():
    q = "Coupling speeds"
    plan = build_query_plan(q)
    ev = evidence("Coupling is performed after permission and confirmation.")
    issues = retrieval_quality_issues(q, [ev], {"coverage_complete": True}, plan)
    assert "scalar_value_evidence_missing" in issues


def test_scalar_request_with_number_passes_scalar_check():
    q = "Train speed during coupling"
    plan = build_query_plan(q)
    ev = evidence("Train will run at restricted speed of 1.5 kmph in Coupling mode.")
    issues = retrieval_quality_issues(q, [ev], {"coverage_complete": True}, plan)
    assert "scalar_value_evidence_missing" not in issues


def test_enumeration_requires_real_coverage_discovery():
    q = "List all elevated and underground sections"
    plan = build_query_plan(q)
    ev = evidence("Line 2 contains elevated and underground sections.")
    issues = retrieval_quality_issues(
        q, [ev], {"coverage_complete": False, "coverage_documents_discovered": 0}, plan
    )
    assert "enumeration_coverage_incomplete" in issues
