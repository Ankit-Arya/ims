from pathlib import Path

from ike.schemas.query import OperationalContext, QueryRequest


def test_realtime_request_contract():
    request = QueryRequest(
        question="wind 65",
        experience="realtime",
        mode="direct",
        operational_context=OperationalContext(
            line="Line 7",
            rolling_stock="RS3",
            role="TO",
            department="Operations",
            operating_mode="UTO",
        ),
    )
    assert request.experience == "realtime"
    block = request.operational_context.prompt_block()
    assert "Line=Line 7" in block
    assert "Operational role=TO" in block
    assert "Operating mode=UTO" in block


def test_qna_remains_default_experience():
    request = QueryRequest(question="Explain high wind procedures", mode="research")
    assert request.experience == "qa"
    assert request.operational_context is None


def test_realtime_ui_contract():
    html = Path("src/ike/frontend/templates/index.html").read_text(encoding="utf-8")
    js = Path("src/ike/frontend/static/app.js").read_text(encoding="utf-8")
    assert 'data-tab="realtime"' in html
    assert 'id="rtLine"' in html
    assert 'id="rtRole"' in html
    assert "experience:'realtime'" in js
    assert "operational_context:realtimeContext" in js
