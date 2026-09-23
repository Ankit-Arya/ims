from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user
from ike.db.models import Document, QueryLog, ReportJob, User
from ike.db.session import get_db
from ike.retrieval.access import document_access_clause
from ike.schemas.dashboard import DashboardSummaryOut, RecentQuestionOut

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _count(db: Session, statement) -> int:
    return int(db.scalar(statement) or 0)


@router.get("/summary", response_model=DashboardSummaryOut)
def dashboard_summary(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DashboardSummaryOut:
    accessible = document_access_clause(user, require_ready=False, require_effective=False)

    question_count = _count(
        db,
        select(func.count(QueryLog.id)).where(QueryLog.user_id == user.id),
    )
    deep_analysis_count = _count(
        db,
        select(func.count(ReportJob.id)).where(ReportJob.created_by == user.id),
    )
    available_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(accessible),
    )
    knowledge_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(
            accessible,
            Document.workspace_scope == "organization",
        ),
    )
    personal_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(
            accessible,
            Document.workspace_scope == "personal",
        ),
    )
    ready_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(
            accessible,
            Document.ingestion_status == "ready",
        ),
    )
    indexing_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(
            accessible,
            Document.ingestion_status.in_(["queued", "processing"]),
        ),
    )
    failed_pdf_count = _count(
        db,
        select(func.count(Document.id)).where(
            accessible,
            Document.ingestion_status == "failed",
        ),
    )

    recent_rows = list(
        db.scalars(
            select(QueryLog)
            .where(QueryLog.user_id == user.id)
            .order_by(QueryLog.created_at.desc())
            .limit(5)
        ).all()
    )
    recent_questions = [
        RecentQuestionOut(
            query_id=row.id,
            question=row.question,
            mode=row.mode,
            resolved_mode=str((row.retrieval_trace or {}).get("resolved_mode") or row.mode),
            confidence=str((row.retrieval_trace or {}).get("confidence") or "low"),
            created_at=row.created_at,
        )
        for row in recent_rows
    ]

    return DashboardSummaryOut(
        question_count=question_count,
        deep_analysis_count=deep_analysis_count,
        available_pdf_count=available_pdf_count,
        knowledge_pdf_count=knowledge_pdf_count,
        personal_pdf_count=personal_pdf_count,
        ready_pdf_count=ready_pdf_count,
        indexing_pdf_count=indexing_pdf_count,
        failed_pdf_count=failed_pdf_count,
        recent_questions=recent_questions,
    )
