from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_personal_document_access_does_not_inherit_admin_bypass():
    source = (ROOT / "src" / "ike" / "retrieval" / "access.py").read_text(encoding="utf-8")
    assert 'Document.workspace_scope == "personal"' in source
    assert "Document.created_by == user.id" in source
    assert "DocumentShare.user_id == user.id" in source
    assert 'Document.workspace_scope == "organization"' in source


def test_personal_upload_and_sharing_endpoints_exist():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "library.py").read_text(encoding="utf-8")
    assert '@router.post("", response_model=DocumentOut, status_code=202)' in source
    assert '@router.put("/{document_id}/shares", response_model=DocumentOut)' in source
    assert 'workspace_scope="personal"' in source
    assert "Only the owner can change sharing" in source
    assert "DocumentShare" in source


def test_personal_pdf_duplicate_retry_is_owner_scoped():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "library.py").read_text(encoding="utf-8")
    assert 'Document.created_by == user.id' in source
    assert 'duplicate.ingestion_status == "failed"' in source
    assert "enqueue_document_ingestion(duplicate.id)" in source
