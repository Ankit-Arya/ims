from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.db.models import Document, DocumentShare, User
from ike.schemas.documents import DocumentOut, SharedUserOut


def can_manage_document(user: User, document: Document) -> bool:
    if document.workspace_scope == "personal":
        return document.created_by == user.id
    return user.role == "admin" or (user.role == "analyst" and document.created_by == user.id)


def can_share_document(user: User, document: Document) -> bool:
    return document.workspace_scope == "personal" and document.created_by == user.id


def document_out(db: Session, user: User, document: Document) -> DocumentOut:
    manage = can_manage_document(user, document)
    share = can_share_document(user, document)
    shared_with: list[SharedUserOut] = []
    if document.workspace_scope == "personal" and share:
        rows = db.execute(
            select(User.id, User.username)
            .join(DocumentShare, DocumentShare.user_id == User.id)
            .where(DocumentShare.document_id == document.id, User.is_active.is_(True))
            .order_by(User.username.asc())
        ).all()
        shared_with = [SharedUserOut(id=row.id, username=row.username) for row in rows]
    payload = DocumentOut.model_validate(document)
    return payload.model_copy(
        update={
            "can_manage": manage,
            "can_share": share,
            "is_owner": document.created_by == user.id,
            "shared_with": shared_with,
            "ingestion_error": document.ingestion_error if manage else None,
        }
    )
