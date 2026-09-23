import shutil
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user
from ike.db.models import Document, DocumentShare, User
from ike.db.session import get_db
from ike.retrieval.access import document_access_clause
from ike.schemas.documents import DocumentOut, DocumentShareUpdate
from ike.services.document_views import document_out
from ike.services.storage import InvalidPdfError, LocalStorage, UploadTooLargeError
from ike.services.task_dispatch import enqueue_document_ingestion

router = APIRouter(prefix="/library", tags=["personal library"])


def _parse_user_ids(value: str | None) -> list[UUID]:
    if not value:
        return []
    result: list[UUID] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            try:
                result.append(UUID(item))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="shared_user_ids contains an invalid user id") from exc
    return list(dict.fromkeys(result))


def _validated_share_users(db: Session, owner: User, user_ids: list[UUID]) -> list[User]:
    ids = [user_id for user_id in user_ids if user_id != owner.id]
    if not ids:
        return []
    users = list(db.scalars(select(User).where(User.id.in_(ids), User.is_active.is_(True))).all())
    if {user.id for user in users} != set(ids):
        raise HTTPException(status_code=422, detail="One or more selected users are unavailable")
    return users


def _replace_shares(db: Session, document: Document, owner: User, user_ids: list[UUID]) -> None:
    if document.created_by != owner.id or document.workspace_scope != "personal":
        raise HTTPException(status_code=403, detail="Only the owner can change sharing for this PDF")
    users = _validated_share_users(db, owner, user_ids)
    db.execute(delete(DocumentShare).where(DocumentShare.document_id == document.id))
    for shared_user in users:
        db.add(
            DocumentShare(
                document_id=document.id,
                user_id=shared_user.id,
                shared_by=owner.id,
            )
        )


@router.get("", response_model=list[DocumentOut])
def list_personal_library(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[DocumentOut]:
    stmt = (
        select(Document)
        .where(
            Document.workspace_scope == "personal",
            document_access_clause(user, require_ready=False, require_effective=False),
        )
        .order_by(Document.created_at.desc())
    )
    return [document_out(db, user, document) for document in db.scalars(stmt).all()]


@router.post("", response_model=DocumentOut, status_code=202)
def upload_personal_pdf(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    shared_user_ids: str | None = Form(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentOut:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="This release accepts PDF files only")
    final_title = (title or file.filename or "document.pdf").strip()
    if len(final_title) < 2 or len(final_title) > 500:
        raise HTTPException(status_code=422, detail="title must contain 2 to 500 characters")
    share_ids = _parse_user_ids(shared_user_ids)
    _validated_share_users(db, user, share_ids)

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
            Document.workspace_scope == "personal",
            Document.created_by == user.id,
            Document.checksum_sha256 == stored.sha256,
            Document.lifecycle_status == "active",
        )
    )
    if duplicate:
        shutil.rmtree(stored.path.parent, ignore_errors=True)
        if duplicate.ingestion_status == "failed":
            _replace_shares(db, duplicate, user, share_ids)
            duplicate.ingestion_status = "queued"
            duplicate.ingestion_error = None
            db.commit()
            db.refresh(duplicate)
            enqueue_document_ingestion(duplicate.id)
            return document_out(db, user, duplicate)
        if duplicate.ingestion_status in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail=f"This PDF is already {duplicate.ingestion_status} in My PDFs")
        raise HTTPException(status_code=409, detail="This PDF is already available in My PDFs")

    document = Document(
        id=document_id,
        title=final_title,
        original_filename=stored.original_filename,
        checksum_sha256=stored.sha256,
        storage_path=str(stored.path),
        workspace_scope="personal",
        access_scope="restricted",
        allowed_roles=[],
        allowed_departments=[],
        lifecycle_status="active",
        ingestion_status="queued",
        created_by=user.id,
    )
    db.add(document)
    db.flush()
    _replace_shares(db, document, user, share_ids)
    db.commit()
    db.refresh(document)
    enqueue_document_ingestion(document.id)
    return document_out(db, user, document)


@router.put("/{document_id}/shares", response_model=DocumentOut)
def update_personal_pdf_shares(
    document_id: UUID,
    payload: DocumentShareUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentOut:
    document = db.get(Document, document_id)
    if not document or document.workspace_scope != "personal" or document.created_by != user.id:
        raise HTTPException(status_code=404, detail="Personal PDF not found")
    _replace_shares(db, document, user, payload.user_ids)
    db.commit()
    db.refresh(document)
    return document_out(db, user, document)
