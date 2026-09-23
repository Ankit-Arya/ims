#!/usr/bin/env python3
"""Fast, DB-free regression check for IMS 0.6.1 query interpretation."""

from __future__ import annotations

from ike.retrieval.normalization import lexical_form_variants
from ike.retrieval.query_plan import build_query_plan


def main() -> None:
    compound = build_query_plan("What is Crew control and what's it's location")
    assert compound.entity_term == "Crew control", compound
    assert compound.lookup_term == "Crew control", compound
    assert compound.facets == ["definition", "location"], compound
    assert compound.coverage_kind == "entity_attribute", compound
    assert "Crew control" in compound.lexical_queries, compound
    assert "Crew controls" in compound.lexical_queries, compound

    where = build_query_plan("Where are fire extinguishers located?")
    assert where.entity_term == "fire extinguishers", where
    assert where.facets == ["location"], where

    acronym = build_query_plan("What does ATO mean?")
    assert acronym.lookup_term == "ATO", acronym

    variants = lexical_form_variants("Automatic Train Operation")
    assert "Automatic Train Operations" in variants, variants
    assert "Automatic Trains Operation" not in variants, variants

    print("PASS: IMS 0.6.1 compound-entity/facet query-plan checks")
    print("entity:", compound.entity_term)
    print("facets:", compound.facets)
    print("coverage_kind:", compound.coverage_kind)
    print("semantic_queries:", compound.semantic_queries)
    print("lexical_queries:", compound.lexical_queries)
    print("exact_terms:", compound.exact_terms)


if __name__ == "__main__":
    main()
