from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_query_embeddings_are_batched_before_per_query_searches():
    engine = read("src/ike/retrieval/engine.py")
    client = read("src/ike/services/inference_client.py")
    assert "query_vectors, embedding_timing = self.inference.embed_queries(queries)" in engine
    assert "def embed_queries" in client
    assert '"texts": texts' in client
    assert "query_embedding_queue_wait" in engine
    assert "query_embedding_execution" in engine


def test_role_alias_fallback_is_bounded_and_subject_lookup_runs_first():
    engine = read("src/ike/retrieval/engine.py")
    config = read("src/ike/core/config.py")
    assert "role_alias_fallback_scan_top_k" in config
    lookup_pos = engine.index("lookup = self._lookup_candidates")
    fallback_pos = engine.index("structural = self._role_alias_source_candidates")
    assert lookup_pos < fallback_pos
    assert "self.settings.role_alias_fallback_scan_top_k" in engine


def test_helpful_answer_contract_is_rich_but_evidence_bound():
    graph = read("src/ike/workflows/qa_graph.py")
    env = read(".env.example")
    assert "HELPFUL_CONTEXT_MODE=rich" in env
    assert "proactively include useful, closely related documentary context" in graph
    assert "never fill documentary gaps from general knowledge" in graph
    assert "Start with the direct answer" in graph
    assert "meaningful H2/H3 headings" in graph
    assert "full form" in graph and "where/when it is used" in graph


def test_verifier_repairs_issues_without_replacing_structure_by_default():
    graph = read("src/ike/workflows/qa_graph.py")
    assert "Do NOT rewrite the answer" in graph
    assert "targeted corrections" in graph
    assert "Preserve good organization, headings, tables" in graph


def test_expensive_stages_have_queue_and_execution_observability():
    engine = read("src/ike/retrieval/engine.py")
    metrics = read("src/ike/services/query_metrics.py")
    runtime = read("inference_service/runtime.py")
    assert "query_embedding_queue_wait" in engine
    assert "rerank_queue_wait" in engine
    assert "dense_search" in engine
    assert "lexical_search" in engine
    assert "exact_search" in engine
    assert "ike_qa_admission_wait_seconds" in metrics
    assert "ike_inference_queue_wait_seconds" in runtime
    assert "ike_inference_execution_seconds" in runtime


def test_role_alias_cache_key_is_user_acl_scope_and_corpus_revision_aware():
    engine = read("src/ike/retrieval/engine.py")
    cache = read("src/ike/retrieval/cache.py")
    assert "_acl_corpus_revision" in engine
    assert "str(user.id)" in engine
    assert "str(user.role)" in engine
    assert "str(user.department" in engine
    assert "tuple(sorted(str(item) for item in (document_ids or [])))" in engine
    assert "corpus_revision" in engine
    assert "IMS deliberately does not cache final answers across users" in cache


def test_helpful_context_is_rich_by_default_and_bounded():
    config = Path('src/ike/core/config.py').read_text(encoding='utf-8')
    graph = Path('src/ike/workflows/qa_graph.py').read_text(encoding='utf-8')
    assert 'helpful_context_mode: Literal["off", "relevant", "rich"] = "rich"' in config
    assert 'helpful_context_max_sections: int = 6' in config
    assert 'Assume the user may not know the precise follow-up questions to ask' in graph
    assert 'do not flood the answer with incidental mentions' in graph
    assert 'Answer ONLY from supplied evidence' in graph
