from ike.core.config import get_settings
from ike.retrieval.normalization import canonical_query_text, technical_identifier_variants
from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import build_deterministic_evidence_plan


def test_red_general_alert_decomposes_definition_and_authority():
    q = "What is red/ general alert and who impose it in DMRC"
    qp = build_query_plan(q)
    ep = build_deterministic_evidence_plan(q, qp)

    assert [x.casefold() for x in qp.entity_terms] == ["red alert", "general alert"]
    assert "definition" in qp.facets
    assert "authority" in qp.facets

    goals = {(g.kind, g.entity_terms[0].casefold()): g for g in ep.goals}
    assert ("definition", "red alert") in goals
    assert ("authority", "red alert") in goals
    assert ("definition", "general alert") in goals
    assert ("authority", "general alert") in goals
    assert goals[("authority", "red alert")].coverage_contract == "authority_proof"


def test_authorised_and_authorized_definition_normalize_equivalently():
    a = build_query_plan("Definition of Authorized officer")
    b = build_query_plan("Definition of “authorised officer”")
    assert a.lookup_term.casefold() == "authorised officer"
    assert b.lookup_term.casefold() == "authorised officer"
    assert a.intent == b.intent == "definition"


def test_numbered_acceptance_prefix_is_not_retrieval_content():
    q = "17. TO kit bag items"
    qp = build_query_plan(q)
    ep = build_deterministic_evidence_plan(q, qp)
    assert qp.normalized == "TO kit bag items"
    assert all(not query.startswith("17.") for goal in ep.goals for query in goal.search_queries)


def test_operational_acronym_survives_fault_framing():
    qp = build_query_plan("What must be done if 3 BIC of RS1 faulty")
    assert qp.intent == "troubleshooting"
    assert "BIC" in qp.entity_terms
    assert qp.rolling_stock == "RS-1"


def test_grammar_number_is_not_technical_identifier():
    variants = technical_identifier_variants("What must be done if 3 BIC of RS1 faulty")
    joined = " | ".join(variants).casefold()
    assert "if3" not in joined
    assert "if-3" not in joined
    assert "if 03" not in joined


def test_speed_question_uses_scalar_with_condition_contract():
    q = "train speed during coupling"
    qp = build_query_plan(q)
    ep = build_deterministic_evidence_plan(q, qp)
    assert ep.goals[0].coverage_contract == "scalar_with_condition"


def test_interactive_research_cross_encoder_is_off_by_default():
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.interactive_research_cross_encoder_enabled is False
