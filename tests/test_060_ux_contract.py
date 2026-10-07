from pathlib import Path

ROOT = Path(__file__).parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_real_cancel_and_streaming_contract():
    route = read("src/ike/api/routes/query.py")
    llm = read("src/ike/services/llm.py")
    js = read("src/ike/frontend/static/app.js")
    assert '/{request_id}/cancel' in route
    assert 'QueryCancelled' in route
    assert 'answer_delta' in route
    assert 'response.output_text.delta' in llm
    assert 'AbortController' in js
    assert 'stopActiveQuery' in js


def test_copy_answer_contract():
    js = read("src/ike/frontend/static/app.js")
    assert 'copy-answer' in js
    assert 'navigator.clipboard' in js
    assert "button.textContent = 'Copied'" in js


def test_agent_owns_semantic_completeness_and_python_has_no_saturation_rule():
    prompts = read("src/ike/agent/prompts.py")
    answer = read("src/ike/agent/answer.py")
    search = read("src/ike/retrieval/search_engine.py")

    assert "Every answer requirement" in prompts
    assert "partial/missing requirements must trigger" in answer
    assert "status=needs_evidence" in prompts
    assert "saturation" not in search
    assert "coverage_status" not in search


def test_no_ingestion_pipeline_change_for_table_awareness():
    pipeline = read("src/ike/ingestion/docling_pipeline.py")
    assert "repeat_table_header=True" in pipeline
    assert "contextualize(chunk=chunk)" in pipeline
    assert "source_metadata=parsed_chunk.metadata" in read("src/ike/ingestion/processor.py")
