from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_visuals_are_lazy_acl_checked_and_server_path_derived():
    service = read("src/ike/services/visuals.py")
    route = read("src/ike/api/routes/visuals.py")
    assert "document_access_clause(user)" in service
    assert "Path(document.storage_path)" in service
    assert "client" not in service.lower() or "client-provided" not in service.lower()
    assert "visual_cache_dir" in service
    assert "fitz.open(source)" in service
    assert "render/{token}" in route


def test_visuals_are_rendered_from_retrieved_evidence_without_reprocessing():
    execution = read("src/ike/services/query_execution.py")
    service = read("src/ike/services/visuals.py")
    js = read("src/ike/frontend/static/app.js")
    assert "resolve_visual_evidence" in execution
    assert "candidate.page_from" in service
    assert "candidate.source_metadata" in service
    assert "visualEvidenceHtml" in js
    assert "/api/v1/visuals/render/" in js
