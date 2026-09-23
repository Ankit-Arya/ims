"""Static release contract for IMS 0.9.0 industrial retrieval fabric."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "migration": ROOT / "migrations/versions/0004_retrieval_intelligence.py",
    "corpus_intelligence": ROOT / "src/ike/retrieval/corpus_intelligence.py",
    "evidence_ledger": ROOT / "src/ike/retrieval/evidence_ledger.py",
    "context_assembly": ROOT / "src/ike/retrieval/context_assembly.py",
    "effort_router": ROOT / "src/ike/workflows/query_understanding.py",
}
for name, path in checks.items():
    assert path.exists(), f"missing {name}: {path}"

engine = (ROOT / "src/ike/retrieval/engine.py").read_text()
graph = (ROOT / "src/ike/workflows/qa_graph.py").read_text()
inference = (ROOT / "inference_service/main.py").read_text()
models = (ROOT / "src/ike/db/models.py").read_text()
version = (ROOT / "src/ike/__init__.py").read_text()

assert any(marker in version for marker in ('__version__ = "0.9.0"', '__version__ = "0.9.1"'))
assert "class RetrievalNode" in models
assert '"/rerank/batch"' in inference
assert "rerank_many(" in engine
assert "CorpusIntelligence" in engine
assert "monotonic_merge" in graph
assert "assemble_draft_context" in graph
assert 'profile="priority_probe"' in graph
assert "retrieval_effort" in graph
assert "corpus_discovery" in graph

# Regression answers/domain examples must not become production-code rules.
production = "\n".join(
    path.read_text(errors="ignore")
    for path in (ROOT / "src").rglob("*.py")
)
for forbidden in (
    "open the station at least ten minutes",
    "Link Line between Line-2 and 3",
    "Battery Isolation Contactor",
    "Brake Isolating Cock",
    "Bogie Isolation Cock",
):
    assert forbidden.casefold() not in production.casefold(), f"hard-coded regression fact: {forbidden}"

print("PASS: IMS 0.9 industrial retrieval fabric structural checks")
