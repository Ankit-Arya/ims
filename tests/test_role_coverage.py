from uuid import uuid4

from ike.retrieval.role import (
    extract_role_aliases,
    phrase_initialism,
    role_candidate_score,
    role_section_key,
)
from ike.retrieval.types import Candidate
from ike.workflows.routing import (
    extract_role_subject,
    is_role_coverage_question,
    role_search_queries,
)


def candidate(text: str, *, section: list[str] | None = None, ordinal: int = 1) -> Candidate:
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=ordinal,
        page_from=1,
        page_to=1,
        section_path=section or [],
        content_kind="text",
        text=text,
        contextual_text=text,
        document_title="Test manual",
        filename="test.pdf",
        revision=None,
        authority=None,
    )


def test_role_question_detection_and_subject_extraction():
    assert is_role_coverage_question("All duties of SC")
    assert is_role_coverage_question("What are the responsibilities of station controller?")
    assert is_role_coverage_question("List every function of the Traffic Controller during failure")
    assert extract_role_subject("All duties of SC") == "SC"
    assert extract_role_subject("all duties of sc") == "SC"
    assert extract_role_subject("What are the responsibilities of station controller?") == "station controller"
    assert extract_role_subject("List every function of the Traffic Controller during failure") == "Traffic Controller"
    assert not is_role_coverage_question("door isolation procedure")


def test_role_alias_resolution_from_structural_heading_is_generic():
    texts = [
        "39. Responsibilities of Station Controller -- Every Station Controller shall open the station.",
        "Security Controller (SC) shall coordinate with security staff.",
        "Unrelated responsibilities of Train Operator",
    ]
    aliases = extract_role_aliases("SC", texts)
    assert "Station Controller" in aliases
    assert "Security Controller" in aliases
    assert all(phrase_initialism(alias) == "SC" for alias in aliases)


def test_role_queries_include_full_name_and_identifier():
    queries = role_search_queries("SC", ["Station Controller"])
    assert "Responsibilities of Station Controller" in queries
    assert "Station Controller shall" in queries
    assert "Responsibilities of SC" in queries


def test_canonical_responsibility_section_scores_over_incidental_mention():
    canonical = candidate(
        "39. Responsibilities of Station Controller -- Every Station Controller shall open the station.",
        section=["CHAPTER VI", "39. Responsibilities of Station Controller"],
    )
    incidental = candidate("The Station Controller was informed about the event.")
    assert role_candidate_score(["Station Controller", "SC"], canonical) > role_candidate_score(
        ["Station Controller", "SC"], incidental
    )


def test_role_section_key_preserves_section_diversity():
    doc_id = uuid4()
    a = candidate("A", section=["Rule 39"])
    b = candidate("B", section=["Rule 49"])
    b.document_id = doc_id
    a.document_id = doc_id
    assert role_section_key(a) != role_section_key(b)
