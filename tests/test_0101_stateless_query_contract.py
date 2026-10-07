from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agent_service_keeps_recent_history_internal_and_user_scoped() -> None:
    source = _source("src/ike/agent/service.py")
    assert "def _recent_history" in source
    assert "QueryLog.user_id == self.user.id" in source
    assert "agent_recent_history_count" in source

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "AgenticQAService":
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "__init__":
                continue
            names = [arg.arg for arg in item.args.args + item.args.kwonlyargs]
            assert "structured_context" not in names
            assert "conversation_context" not in names
            return
    raise AssertionError("AgenticQAService.__init__ not found")


def test_query_execution_has_one_agent_runtime_and_no_legacy_fallback() -> None:
    source = _source("src/ike/services/query_execution.py")
    assert "AgenticQAService" in source
    assert "QAGraphService" not in source
    assert "falling_back" not in source


def test_query_api_exposes_no_mutable_context_endpoint() -> None:
    source = _source("src/ike/api/routes/query.py")
    assert '@router.get("/context")' not in source
    assert '@router.delete("/context"' not in source


def test_frontend_has_no_persistent_context_controls() -> None:
    html = _source("src/ike/frontend/templates/index.html")
    javascript = _source("src/ike/frontend/static/app.js")
    assert "activeContextBar" not in html
    assert "clearContextBtn" not in html
    assert "loadQueryContext" not in javascript
    assert "clearQueryContext" not in javascript
    assert "/api/v1/query/context" not in javascript
