from ike.retrieval.query_plan import build_query_plan


def test_high_wind_gets_operational_semantic_queries():
    plan = build_query_plan("high wind")
    joined = " | ".join(plan.semantic_queries).casefold()
    assert "high wind speed train movement procedure" in joined
    assert "thunderstorm cyclone train operating instruction" in joined


def test_coupling_gets_procedure_queries():
    plan = build_query_plan("coupling")
    joined = " | ".join(plan.semantic_queries).casefold()
    assert "train coupling procedure" in joined
    assert "rescue train coupling procedure" in joined


def test_stuck_between_stations_does_not_generate_identifier_noise():
    plan = build_query_plan("Train is stuck between 2 station")
    assert not any("between2" in q or "between-2" in q for q in plan.semantic_queries)
    joined = " | ".join(plan.semantic_queries).casefold()
    assert "immobile train mid section procedure" in joined
    assert "rescue of immobilised revenue train" in joined
