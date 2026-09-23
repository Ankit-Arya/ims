"""Static release guardrails for IMS 0.9.1 corpus-intelligence optimization."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
corpus = (ROOT / "src/ike/retrieval/corpus_intelligence.py").read_text()
qa = (ROOT / "src/ike/workflows/qa_graph.py").read_text()
backfill = (ROOT / "scripts/build_retrieval_intelligence.py").read_text()
config = (ROOT / "src/ike/core/config.py").read_text()
version = (ROOT / "src/ike/__init__.py").read_text()

assert 'INDEX_VERSION = "0.9.1"' in corpus
assert "embed_documents(" not in corpus
assert '"node_type": "concept"' not in corpus
assert "_normalized_centroid" in corpus
assert '"terms": terms' in corpus
assert "routing_safe" in corpus
assert "coverage_below_hint_threshold" in corpus
assert "if corpus_discovery.routing_safe else []" in qa
assert "retrieval_intelligence_terms_per_section" in config
assert "retrieval_intelligence_min_hint_coverage_ratio" in config
assert "existing_chunk_centroids" in backfill
assert '0.9.1' in version
print("PASS: IMS 0.9.1 corpus-intelligence efficiency and coverage-safety checks")
