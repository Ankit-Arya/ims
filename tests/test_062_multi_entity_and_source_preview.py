from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ike.retrieval.evidence_shape import enumeration_shape_score
from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.table_context import table_query_affinity
from ike.retrieval.types import Candidate
from ike.workflows.evidence_planning import build_deterministic_evidence_plan
from ike.workflows.routing import extract_entity_term, extract_entity_terms

ROOT = Path(__file__).parents[1]


def test_coordinated_enumeration_is_split_into_independent_entities():
    plan = build_query_plan("List of depots and crew controls")

    assert [value.casefold() for value in plan.entity_terms] == ["depots", "crew controls"]
    assert plan.entity_term is None
    assert plan.coverage_kind == "entity_attribute"
    assert plan.answer_type == "structured_list"

    lexical = {value.casefold() for value in plan.lexical_queries}
    exact = {value.casefold() for value in plan.exact_terms}
    semantic = {value.casefold() for value in plan.semantic_queries}

    assert {"depots", "depot", "crew controls", "crew control"}.issubset(lexical)
    assert {"depots", "depot", "crew controls", "crew control"}.issubset(exact)
    assert {"depots", "crew controls"}.issubset(semantic)
    assert "depots and crew controls" not in exact
    assert "depots and crew controls" not in lexical


def test_nested_list_request_wrapper_is_removed_before_entity_segmentation():
    question = "Provide list of crew controls, depots, interchange stations"
    plan = build_query_plan(question)

    assert [value.casefold() for value in plan.entity_terms] == [
        "crew controls",
        "depots",
        "interchange stations",
    ]
    assert "list of crew controls" not in {value.casefold() for value in plan.entity_terms}

    evidence_plan = build_deterministic_evidence_plan(question, plan)
    assert [goal.entity_terms[0].casefold() for goal in evidence_plan.goals] == [
        "crew controls",
        "depots",
        "interchange stations",
    ]
    assert all(goal.coverage_contract == "enumerate_set" for goal in evidence_plan.goals)


def test_compound_name_is_not_blindly_split_on_and():
    question = "List of safety and security equipment"
    assert extract_entity_terms(question) == ["safety and security equipment"]
    assert extract_entity_term(question) == "safety and security equipment"


def test_explicit_three_item_list_is_split_and_balanced_for_planning():
    plan = build_query_plan("Show stations, depots and OCCs")
    assert [value.casefold() for value in plan.entity_terms] == ["stations", "depots", "occs"]
    assert plan.entity_term is None
    assert "station" in {value.casefold() for value in plan.lexical_queries}
    assert "depot" in {value.casefold() for value in plan.lexical_queries}


def test_enumeration_shape_prefers_set_bearing_structure_over_plain_mention():
    structured = Candidate(
        chunk_id=uuid4(), document_id=uuid4(), ordinal=1,
        page_from=1, page_to=1, section_path=["Depot Directory"],
        content_kind="table",
        text="Depot | Line\nAlpha Depot | L1\nBeta Depot | L2\nGamma Depot | L3\nDelta Depot | L4",
        contextual_text="Depot Directory\nDepot | Line\nAlpha Depot | L1\nBeta Depot | L2\nGamma Depot | L3\nDelta Depot | L4",
        document_title="Reference.pdf", filename="Reference.pdf", revision=None, authority=None,
    )
    mention = Candidate(
        chunk_id=uuid4(), document_id=uuid4(), ordinal=2,
        page_from=2, page_to=2, section_path=["Operating Notes"],
        content_kind="text",
        text="The train proceeds towards the depot after service.",
        contextual_text="Operating Notes. The train proceeds towards the depot after service.",
        document_title="Reference.pdf", filename="Reference.pdf", revision=None, authority=None,
    )

    assert enumeration_shape_score("depots", structured) > 0.55
    assert enumeration_shape_score("depots", structured) > enumeration_shape_score("depots", mention)


def test_table_affinity_ignores_conversational_function_words():
    candidate = Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=1,
        page_from=1,
        page_to=1,
        section_path=["General Information"],
        content_kind="table",
        text="Crew Controls | 02 | Example A, Example B",
        contextual_text="Crew Controls | 02 | Example A, Example B",
        document_title="Reference.pdf",
        filename="Reference.pdf",
        revision=None,
        authority=None,
    )
    focused = table_query_affinity("List of depots and crew controls", candidate)
    noisy = table_query_affinity("what is the and of for", candidate)
    assert focused > 0
    assert noisy == 0


def test_copy_answer_and_inline_source_hover_contract():
    js = (ROOT / "src" / "ike" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "src" / "ike" / "web" / "static" / "app.css").read_text(encoding="utf-8")

    assert "copyTextToClipboard" in js
    assert 'class="copy-answer secondary"' in js
    assert "navigator.clipboard?.writeText" in js
    assert "button.textContent = 'Copied'" in js

    assert 'data-evidence-id="$1"' in js
    assert "bindCitationPreviews" in js
    assert "showSourcePreview" in js
    assert "citation.excerpt" in js
    assert "sourceCitationPreview" in js
    assert ".source-citation-preview" in css
    assert ".inline-citation.source-preview-enabled:hover" in css
