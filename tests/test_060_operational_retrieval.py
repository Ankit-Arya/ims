from uuid import uuid4

from ike.retrieval.normalization import lexical_form_variants, technical_identifier_variants
from ike.retrieval.overview import overview_name_matches
from ike.retrieval.query_plan import build_query_plan
from ike.retrieval.table_context import retrieval_text, table_query_affinity
from ike.retrieval.types import Candidate, Evidence
from ike.retrieval.role import extract_role_aliases


def candidate(**overrides):
    values = dict(
        chunk_id=uuid4(), document_id=uuid4(), ordinal=1, page_from=3, page_to=3,
        section_path=["Crew controls"], content_kind="table", text="Cab | Left side",
        contextual_text="Crew controls\nControl | Location\nCab | Left side",
        document_title="Manual", filename="manual.pdf", revision=None, authority=None,
        source_metadata={"captions": ["Crew controls and locations"]},
    )
    values.update(overrides)
    return Candidate(**values)


def test_identifier_variants_keep_original_and_bridge_rs_spacing():
    variants = technical_identifier_variants("procedure for RS 10")
    assert variants[0] == "procedure for RS 10"
    assert "procedure for RS10" in variants
    assert "procedure for RS-10" in variants


def test_query_plan_treats_dmrc_as_scope_and_extracts_rs():
    plan = build_query_plan("Procedure for RS10 in DMRC")
    assert plan.original == "Procedure for RS10 in DMRC"
    assert plan.rolling_stock == "RS-10"
    assert "DMRC" in plan.scope_terms
    assert any("RS-10" in item for item in plan.semantic_queries)


def test_mrgr_is_configurable_overview_identity():
    assert overview_name_matches("Metro Railway General Rules", "02. MRGR 2020.pdf", "MRGR")
    assert not overview_name_matches("RS-10 Manual", "rs10.pdf", "MRGR")


def test_table_rerank_and_prompt_keep_header_context():
    c = candidate()
    text = retrieval_text(c)
    assert "Crew controls" in text
    assert "Control | Location" in text
    assert table_query_affinity("location of crew controls", c) > 0
    block = Evidence("E1", c).prompt_block()
    assert "Crew controls" in block
    assert "Control | Location" in block


def test_role_alias_canonicalization_strips_grammatical_wrapper():
    aliases = extract_role_aliases("SC", ["Responsibilities of the Station Controller (SC)"])
    assert "Station Controller" in aliases
    assert all(not value.lower().startswith(("the ", "and ")) for value in aliases)


def test_simple_fts_surface_variants_help_terse_table_queries():
    variants = lexical_form_variants("locations of crew controls")
    assert "location of crew control" in variants
