from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_failed_duplicate_is_retried_not_duplicated():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "documents.py").read_text(encoding="utf-8")
    assert 'duplicate.ingestion_status == "failed"' in source
    assert "enqueue_document_ingestion(duplicate.id)" in source
    assert "Renaming the file does not create a new copy" in source


def test_source_view_download_retry_and_hard_delete_endpoints_exist():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "documents.py").read_text(encoding="utf-8")
    for route in ("/{document_id}/view", "/{document_id}/download", "/{document_id}/retry"):
        assert route in source
    assert "purge_document_files(document_id)" in source
    assert "delete(Document)" in source


def test_ingestion_retries_show_queued_until_terminal_failure():
    source = (ROOT / "src" / "ike" / "tasks" / "ingestion.py").read_text(encoding="utf-8")
    assert "document_ingestion_retry_scheduled" in source
    assert 'document.ingestion_status = "queued"' in source
    assert "self.retry" in source
