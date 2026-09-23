from pathlib import Path


def test_docling_pipeline_rejects_partial_conversion_results():
    source = Path("src/ike/ingestion/docling_pipeline.py").read_text(encoding="utf-8")
    assert "status != ConversionStatus.SUCCESS" in source
    assert "has_timeout_errors" in source
    assert "partial operational documents are not accepted" in source


def test_docling_timeout_can_be_disabled_and_fresh_default_is_longer():
    source = Path("src/ike/ingestion/docling_pipeline.py").read_text(encoding="utf-8")
    config = Path("src/ike/core/config.py").read_text(encoding="utf-8")
    assert "None if settings.docling_timeout_seconds <= 0" in source
    assert "docling_timeout_seconds: int = 3600" in config
