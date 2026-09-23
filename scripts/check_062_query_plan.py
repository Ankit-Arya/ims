#!/usr/bin/env python3
"""Fast DB-free smoke check for IMS 0.6.2 query interpretation."""

from ike.retrieval.query_plan import build_query_plan


def main() -> None:
    cases = {
        "List of depots and crew controls": {
            "entities": ["depots", "crew controls"],
            "facets": ["enumeration"],
        },
        "What is Crew control and what's it's location": {
            "entities": ["crew control"],
            "facets": ["definition", "location"],
        },
    }
    for question, expected in cases.items():
        plan = build_query_plan(question)
        entities = [value.casefold() for value in plan.entity_terms]
        assert entities == expected["entities"], (question, entities)
        assert plan.facets == expected["facets"], (question, plan.facets)
        assert plan.coverage_kind == "entity_attribute", (question, plan.coverage_kind)
        print(f"QUESTION: {question}")
        print(f"  entity_terms={plan.entity_terms}")
        print(f"  lexical_queries={plan.lexical_queries}")
        print(f"  exact_terms={plan.exact_terms}")
        print(f"  facets={plan.facets}")
    print("PASS: IMS 0.6.2 multi-entity/facet query-plan checks")


if __name__ == "__main__":
    main()
