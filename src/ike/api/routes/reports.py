from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user
from ike.db.models import Document, ReportJob, User
from ike.core.config import get_settings
from ike.db.session import get_db
from ike.retrieval.access import document_access_clause
from ike.schemas.reports import ReportCreate, ReportOut
from ike.services.task_dispatch import enqueue_report_build

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportOut, status_code=202)
def create_report(payload: ReportCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ReportJob:
    settings = get_settings()
    if len(payload.document_ids) > settings.report_max_documents:
        raise HTTPException(status_code=422, detail=f"Deep Analysis supports up to {settings.report_max_documents} documents per run; select a smaller source set")
    accessible = list(
        db.scalars(
            select(Document).where(Document.id.in_(payload.document_ids), document_access_clause(user))
        ).all()
    )
    if {d.id for d in accessible} != set(payload.document_ids):
        raise HTTPException(status_code=403, detail="One or more selected documents are unavailable, not ready, inactive, or inaccessible")
    job = ReportJob(
        created_by=user.id,
        title=payload.title,
        objective=payload.objective,
        document_ids=payload.document_ids,
        status="queued",
        progress_stage="queued",
        progress_percent=0,
        progress_message="Waiting for analysis worker",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    enqueue_report_build(job.id)
    return job


@router.get("", response_model=list[ReportOut])
def list_reports(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ReportJob]:
    stmt = select(ReportJob).where(ReportJob.created_by == user.id).order_by(ReportJob.created_at.desc()).limit(100)
    return list(db.scalars(stmt).all())


@router.get("/{report_id}", response_model=ReportOut)
def get_report(report_id: UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ReportJob:
    job = db.get(ReportJob, report_id)
    if not job or job.created_by != user.id:
        raise HTTPException(status_code=404, detail="Report not found")
    return job
