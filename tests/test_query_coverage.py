from ike.workflows.routing import (
    coverage_search_query,
    is_coverage_question,
    is_entity_attribute_coverage_question,
)


def test_procedure_is_coverage_sensitive():
    assert is_coverage_question("door isolation procedure")
    assert is_coverage_question("steps for resetting a failed door")


def test_coverage_query_is_deterministic_and_topic_focused():
    assert coverage_search_query("What is the door isolation procedure across all manuals?") == "door isolation"
    assert coverage_search_query("Provide steps for brake isolation") == "brake isolation"


def test_entity_attribute_queries_are_coverage_sensitive():
    assert is_entity_attribute_coverage_question("locations of crew controls")
    assert is_entity_attribute_coverage_question("list of all crew controls line wise")
    assert is_entity_attribute_coverage_question("What is Crew Control and where is it located?")
    assert not is_entity_attribute_coverage_question("What is Crew Control?")
