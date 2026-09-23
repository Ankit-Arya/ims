from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_qa_graph_has_no_cross_query_context_parameter() -> None:
    tree = ast.parse(_source("src/ike/workflows/qa_graph.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QAGraphService":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    names = [arg.arg for arg in item.args.args + item.args.kwonlyargs]
                    assert "structured_context" not in names
                    return
    raise AssertionError("QAGraphService.__init__ not found")


def test_query_execution_does_not_read_or_write_conversation_context() -> None:
    source = _source("src/ike/services/query_execution.py")
    assert "conversation_context" not in source
    assert '"cross_query_context_used": False' in source


def test_query_api_exposes_no_context_endpoint() -> None:
    source = _source("src/ike/api/routes/query.py")
    assert '@router.get("/context")' not in source
    assert '@router.delete("/context"' not in source
    assert "conversation_context" not in source


def test_web_ui_has_no_persistent_context_controls() -> None:
    html = _source("src/ike/web/templates/index.html")
    javascript = _source("src/ike/web/static/app.js")
    assert "activeContextBar" not in html
    assert "clearContextBtn" not in html
    assert "loadQueryContext" not in javascript
    assert "clearQueryContext" not in javascript
    assert "/api/v1/query/context" not in javascript


def test_legacy_context_store_is_inert() -> None:
    source = _source("src/ike/services/conversation_context.py")
    assert "Redis" not in source
    assert "setex" not in source
    assert "ims:qa-context" not in source
    assert "return {}" in source
