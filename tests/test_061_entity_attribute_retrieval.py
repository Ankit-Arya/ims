from ike.retrieval.normalization import lexical_form_variants
from ike.retrieval.query_plan import build_query_plan
from ike.workflows.routing import (
    extract_entity_term,
    extract_lookup_term,
    is_complex_question,
    is_entity_attribute_coverage_question,
    query_facets,
)


def test_compound_definition_entity_is_not_truncated():
    question = "What is Crew control and what's it's location"
    assert extract_lookup_term(question) == "Crew control"
    assert extract_entity_term(question) == "Crew control"
    assert query_facets(question) == ["definition", "location"]
    assert is_entity_attribute_coverage_question(question)
    assert is_complex_question(question)


def test_query_plan_separates_semantic_and_lexical_forms_for_entity_attribute():
    plan = build_query_plan("What is Crew control and what's it's location")
    assert plan.entity_term == "Crew control"
    assert plan.coverage_kind == "entity_attribute"
    assert plan.facets == ["definition", "location"]
    assert "Crew control" in plan.semantic_queries
    assert "Crew control location" in plan.semantic_queries
    assert "Crew control" in plan.lexical_queries
    assert "Crew controls" in plan.lexical_queries
    assert "Crew Control".casefold() in {value.casefold() for value in plan.exact_terms}


def test_simple_fts_plural_bridge_only_changes_terminal_head():
    variants = lexical_form_variants("Automatic Train Operation")
    assert "Automatic Train Operations" in variants
    assert "Automatic Trains Operation" not in variants
    assert "Automatics Train Operation" not in variants


def test_where_form_does_not_include_location_predicate_in_entity():
    plan = build_query_plan("Where are fire extinguishers located?")
    assert plan.entity_term == "fire extinguishers"
    assert plan.facets == ["location"]
    assert "fire extinguisher" in {value.casefold() for value in plan.lexical_queries}


def test_linewise_list_activates_generic_entity_set_coverage():
    plan = build_query_plan("List of all crew controls line wise")
    assert plan.entity_term == "crew controls"
    assert "enumeration" in plan.facets
    assert plan.coverage_kind == "entity_attribute"
    assert "crew control" in {value.casefold() for value in plan.lexical_queries}


def test_what_does_mean_keeps_only_the_entity():
    assert extract_lookup_term("What does ATO mean?") == "ATO"
    assert extract_lookup_term("What does ATP stand for?") == "ATP"
