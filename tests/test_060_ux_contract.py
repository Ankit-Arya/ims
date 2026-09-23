from pathlib import Path

ROOT = Path(__file__).parents[1]

def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


def test_real_cancel_and_streaming_contract():
    route = read('src/ike/api/routes/query.py')
    llm = read('src/ike/services/llm.py')
    js = read('src/ike/web/static/app.js')
    assert '/{request_id}/cancel' in route
    assert 'QueryCancelled' in route
    assert 'answer_delta' in route
    assert 'response.output_text.delta' in llm
    assert 'AbortController' in js
    assert 'stopActiveQuery' in js


def test_copy_answer_contract():
    js = read('src/ike/web/static/app.js')
    assert 'copy-answer' in js
    assert 'navigator.clipboard' in js
    assert "button.textContent = 'Copied'" in js


def test_semantic_and_coverage_work_together():
    graph = read('src/ike/workflows/qa_graph.py')
    assert 'Coverage-sensitive questions still need semantic expansion' in graph
    assert 'if coverage_sensitive:' not in graph.split('def _expand',1)[1].split('def _retrieve_expanded',1)[0]


def test_no_ingestion_pipeline_change_for_table_awareness():
    pipeline = read('src/ike/ingestion/docling_pipeline.py')
    assert 'repeat_table_header=True' in pipeline
    assert 'contextualize(chunk=chunk)' in pipeline
    assert 'source_metadata=parsed_chunk.metadata' in read('src/ike/ingestion/processor.py')
