from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import (
    build_deterministic_evidence_plan,
    should_use_semantic_planner,
)


def _plan(question: str):
    query_plan = build_query_plan(question)
    evidence_plan = build_deterministic_evidence_plan(question, query_plan)
    return query_plan, evidence_plan


def test_wrong_station_announcement_is_troubleshooting_not_claim_check():
    query_plan, evidence_plan = _plan("Wrong station announcement in metro")
    assert query_plan.intent == "troubleshooting"
    assert evidence_plan.strategy == "procedure"
    assert evidence_plan.goals[0].kind == "procedure"
    assert not should_use_semantic_planner(
        "Wrong station announcement in metro", query_plan, evidence_plan
    )
    search_text = " ".join(evidence_plan.goals[0].search_queries).casefold()
    assert "pa pis" in search_text
    assert "manual announcement" in search_text


def test_other_fault_phrasings_use_troubleshooting_path():
    for question in (
        "announcement of wrong station is happening",
        "what shall TO do if announcement of wrong station is happening in metro",
        "PIS showing incorrect station",
        "doors are not closing",
        "signal is blank",
    ):
        query_plan, evidence_plan = _plan(question)
        assert query_plan.intent == "troubleshooting", question
        assert evidence_plan.goals[0].kind == "procedure", question


def test_explicit_verification_stays_claim_check_capable():
    question = "Is it mentioned anywhere that wrong station announcements have occurred?"
    query_plan, evidence_plan = _plan(question)
    assert query_plan.intent != "troubleshooting"
    assert evidence_plan.strategy == "claim_check"
    assert any(goal.kind == "claim_check" for goal in evidence_plan.goals)


def test_realtime_start_does_not_depend_on_session_storage():
    html = open("src/ike/web/templates/index.html", encoding="utf-8").read()
    js = open("src/ike/web/static/app.js", encoding="utf-8").read()
    assert "Required:</strong> Line and Operational role" in html
    transition = js.index("$('realtimeSession').classList.remove('hidden')")
    storage = js.index("sessionStorage.setItem('ims-realtime-context'")
    assert transition < storage
    assert "try { sessionStorage.setItem('ims-realtime-context'" in js
