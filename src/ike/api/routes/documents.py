import re
import shutil
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user, require_analyst
from ike.db.models import Document, User
from ike.db.session import get_db
from ike.retrieval.access import document_access_clause
from ike.schemas.documents import DocumentAuthorityUpdate, DocumentOut
from ike.services.document_views import can_manage_document, document_out
from ike.services.storage import InvalidPdfError, LocalStorage, UploadTooLargeError
from ike.services.task_dispatch import enqueue_document_ingestion

router = APIRouter(prefix="/documents", tags=["documents"])


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _accessible_document(
    db: Session,
    user: User,
    document_id: UUID,
    *,
    ready: bool = False,
) -> Document | None:
    return db.scalar(
        select(Document).where(
            Document.id == document_id,
            document_access_clause(user, require_ready=ready, require_effective=False),
        )
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[DocumentOut]:
    """List the organisation knowledge library only.

    Personal/shared PDFs live under /library. Retrieval may still use both when
    the user asks across all accessible documents.
    """
    stmt = (
        select(Document)
        .where(
            Document.workspace_scope == "organization",
            document_access_clause(user, require_ready=False, require_effective=False),
        )
        .order_by(Document.created_at.desc())
    )
    return [document_out(db, user, document) for document in db.scalars(stmt).all()]


@router.post("", response_model=DocumentOut, status_code=202)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    family_key: str | None = Form(default=None),
    revision: str | None = Form(default=None),
    authority: str | None = Form(default=None),
    source_role: str | None = Form(default=None),
    authority_level: int | None = Form(default=None),
    supersedes_document_id: UUID | None = Form(default=None),
    department: str | None = Form(default=None),
    access_scope: str = Form(default="organization"),
    allowed_roles: str | None = Form(default=None),
    allowed_departments: str | None = Form(default=None),
    effective_from: date | None = Form(default=None),
    effective_to: date | None = Form(default=None),
    allow_duplicate: bool = Form(default=False),
    user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> DocumentOut:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="This release accepts PDF files only")
    if access_scope not in {"organization", "department", "restricted"}:
        raise HTTPException(status_code=422, detail="access_scope must be organization, department, or restricted")
    final_title = (title or file.filename or "document.pdf").strip()
    if len(final_title) < 2 or len(final_title) > 500:
        raise HTTPException(status_code=422, detail="title must contain 2 to 500 characters")
    role_list = _csv(allowed_roles)
    invalid_roles = sorted(set(role_list) - {"admin", "analyst", "user"})
    if invalid_roles:
        raise HTTPException(status_code=422, detail=f"Unknown allowed role(s): {', '.join(invalid_roles)}")
    department_list = _csv(allowed_departments)
    effective_department = (department or user.department or "").strip() or None
    if access_scope == "department" and not effective_department:
        raise HTTPException(status_code=422, detail="department access requires a document or user department")
    if access_scope == "restricted" and not role_list and not department_list:
        raise HTTPException(status_code=422, detail="restricted access requires allowed_roles and/or allowed_departments")
    if effective_from and effective_to and effective_to < effective_from:
        raise HTTPException(status_code=422, detail="effective_to cannot be earlier than effective_from")
    if any(len(value) > 120 for value in department_list):
        raise HTTPException(status_code=422, detail="allowed department names must be 120 characters or fewer")
    normalized_source_role = re.sub(
        r"[^a-z0-9]+", "_", (source_role or "").strip().casefold()
    ).strip("_") or None
    if normalized_source_role and len(normalized_source_role) > 64:
        raise HTTPException(status_code=422, detail="source_role must be 64 characters or fewer")
    if authority_level is not None and not 0 <= authority_level <= 100:
        raise HTTPException(status_code=422, detail="authority_level must be between 0 and 100")
    if supersedes_document_id is not None:
        superseded = _accessible_document(db, user, supersedes_document_id)
        if superseded is None or not can_manage_document(user, superseded):
            raise HTTPException(
                status_code=422,
                detail="supersedes_document_id must reference a document you manage",
            )

    document_id = uuid4()
    storage = LocalStorage()
    try:
        stored = storage.save_upload(file, document_id)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidPdfError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    duplicate = db.scalar(
        select(Document).where(
            Document.workspace_scope == "organization",
            Document.checksum_sha256 == stored.sha256,
            document_access_clause(user, require_ready=False, require_effective=False),
        )
    )
    if duplicate and not allow_duplicate:
        shutil.rmtree(stored.path.parent, ignore_errors=True)
        if duplicate.ingestion_status == "failed":
            if not can_manage_document(user, duplicate):
                raise HTTPException(
                    status_code=409,
                    detail="An identical failed document already exists. Its uploader or an administrator can retry it.",
                )
            duplicate.ingestion_status = "queued"
            duplicate.ingestion_error = None
            db.commit()
            db.refresh(duplicate)
            enqueue_document_ingestion(duplicate.id)
            return document_out(db, user, duplicate)
        if duplicate.ingestion_status in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail=f"An identical document already exists and is {duplicate.ingestion_status}. Renaming the file does not create a new copy.",
            )
        raise HTTPException(
            status_code=409,
            detail="An identical active document is already indexed. Renaming the file does not create a new copy.",
        )

    document = Document(
        id=document_id,
        family_key=family_key,
        title=final_title,
        original_filename=stored.original_filename,
        checksum_sha256=stored.sha256,
        storage_path=str(stored.path),
        revision=revision,
        authority=authority,
        source_role=normalized_source_role,
        authority_level=authority_level,
        supersedes_document_id=supersedes_document_id,
        department=effective_department,
        workspace_scope="organization",
        access_scope=access_scope,
        allowed_roles=role_list,
        allowed_departments=department_list,
        effective_from=effective_from,
        effective_to=effective_to,
        lifecycle_status="active",
        ingestion_status="queued",
        created_by=user.id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    enqueue_document_ingestion(document.id)
    return document_out(db, user, document)


@router.patch("/{document_id}/authority", response_model=DocumentOut)
def update_document_authority(
    document_id: UUID,
    payload: DocumentAuthorityUpdate,
    user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> DocumentOut:
    """Update retrieval authority metadata without reprocessing document content."""

    document = _accessible_document(db, user, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not can_manage_document(user, document):
        raise HTTPException(status_code=403, detail="You do not manage this document")

    updates = payload.model_dump(exclude_unset=True)
    if "source_role" in updates:
        document.source_role = re.sub(
            r"[^a-z0-9]+", "_", str(updates["source_role"] or "").strip().casefold()
        ).strip("_") or None
    if "authority_level" in updates:
        document.authority_level = updates["authority_level"]
    if "supersedes_document_id" in updates:
        supersedes_id = updates["supersedes_document_id"]
        if supersedes_id == document_id:
            raise HTTPException(
                status_code=422, detail="A document cannot supersede itself"
            )
        if supersedes_id is not None:
            superseded = _accessible_document(db, user, supersedes_id)
            if superseded is None or not can_manage_document(user, superseded):
                raise HTTPException(
                    status_code=422,
                    detail="supersedes_document_id must reference a document you manage",
                )
        document.supersedes_document_id = supersedes_id

    db.commit()
    db.refresh(document)
    return document_out(db, user, document)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> DocumentOut:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_out(db, user, document)


@router.get("/{document_id}/view", response_class=FileResponse)
def view_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(document.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail="The source PDF is no longer present on storage")
    return FileResponse(path, media_type="application/pdf", filename=document.original_filename, content_disposition_type="inline")


@router.get("/{document_id}/download", response_class=FileResponse)
def download_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(document.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail="The source PDF is no longer present on storage")
    return FileResponse(path, media_type="application/pdf", filename=document.original_filename, content_disposition_type="attachment")


@router.post("/{document_id}/reindex", status_code=202)
def reindex_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not can_manage_document(user, document):
        raise HTTPException(status_code=403, detail="You do not manage this document")
    if document.ingestion_status in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail=f"Document is already {document.ingestion_status}")
    document.ingestion_status = "queued"
    document.ingestion_error = None
    db.commit()
    enqueue_document_ingestion(document_id)
    return {"document_id": str(document_id), "status": "queued"}


@router.post("/{document_id}/retry", status_code=202)
def retry_failed_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not can_manage_document(user, document):
        raise HTTPException(status_code=403, detail="You do not manage this document")
    if document.ingestion_status != "failed":
        raise HTTPException(status_code=409, detail=f"Retry is only available for failed documents; current status is {document.ingestion_status}")
    document.ingestion_status = "queued"
    document.ingestion_error = None
    db.commit()
    enqueue_document_ingestion(document_id)
    return {"document_id": str(document_id), "status": "queued"}


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    document = _accessible_document(db, user, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not can_manage_document(user, document):
        raise HTTPException(status_code=403, detail="You do not manage this document")
    if document.ingestion_status in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="A document cannot be deleted while indexing is active. Wait for it to finish or fail, then delete it.",
        )
    db.execute(delete(Document).where(Document.id == document_id))
    db.commit()
    LocalStorage().purge_document_files(document_id)
