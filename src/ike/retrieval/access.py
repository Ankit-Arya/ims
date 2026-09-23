from datetime import date

from sqlalchemy import and_, exists, or_
from sqlalchemy.sql.elements import ColumnElement

from ike.db.models import Document, DocumentShare, User


def document_access_clause(
    user: User,
    *,
    require_ready: bool = True,
    require_effective: bool = True,
) -> ColumnElement[bool]:
    """Return the row-level document access predicate for a signed-in user.

    Organisation documents follow role/department ACLs. Personal documents do
    not inherit administrator visibility: only the owner and explicitly shared
    users can access them through normal product endpoints.
    """

    base = [Document.lifecycle_status == "active"]
    if require_ready:
        base.append(Document.ingestion_status == "ready")
    if require_effective:
        today = date.today()
        base.extend(
            [
                or_(Document.effective_from.is_(None), Document.effective_from <= today),
                or_(Document.effective_to.is_(None), Document.effective_to >= today),
            ]
        )

    personal_access = and_(
        Document.workspace_scope == "personal",
        or_(
            Document.created_by == user.id,
            exists().where(
                and_(
                    DocumentShare.document_id == Document.id,
                    DocumentShare.user_id == user.id,
                )
            ),
        ),
    )

    if user.role == "admin":
        organisation_access: ColumnElement[bool] = Document.workspace_scope == "organization"
    else:
        choices = [Document.access_scope == "organization", Document.allowed_roles.any(user.role)]
        if user.department:
            choices.extend(
                [
                    and_(Document.access_scope == "department", Document.department == user.department),
                    Document.allowed_departments.any(user.department),
                ]
            )
        organisation_access = and_(Document.workspace_scope == "organization", or_(*choices))

    return and_(*base, or_(organisation_access, personal_access))
