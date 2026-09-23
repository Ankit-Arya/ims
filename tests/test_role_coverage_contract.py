from pathlib import Path


def test_role_coverage_is_query_side_only():
    engine = Path("src/ike/retrieval/engine.py").read_text()
    graph = Path("src/ike/workflows/qa_graph.py").read_text()
    worker = Path("src/ike/ingestion/processor.py").read_text()

    assert "_role_coverage_candidates" in engine
    assert "is_role_coverage_question" in graph
    assert "role_coverage" not in worker


def test_trace_exposes_role_diagnostics_and_candidate_previews():
    engine = Path("src/ike/retrieval/engine.py").read_text()
    for key in (
        '"role_subject"',
        '"role_aliases"',
        '"role_sections_discovered"',
        '"role_documents_discovered"',
        '"fused_candidate_preview"',
        '"reranked_candidate_preview"',
    ):
        assert key in engine
