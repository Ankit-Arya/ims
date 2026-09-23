from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docling_models_are_prefetched_into_persistent_artifacts_path():
    bootstrap = (ROOT / "src" / "ike" / "services" / "model_bootstrap.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "src" / "ike" / "ingestion" / "docling_pipeline.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "docling-tools" in bootstrap
    assert "layout" in bootstrap and "tableformer" in bootstrap
    assert "artifacts_path=settings.docling_artifacts_path" in pipeline
    assert "model-bootstrap" in compose
    assert "service_completed_successfully" in compose
