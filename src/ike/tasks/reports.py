import logging
from datetime import UTC, datetime
from uuid import UUID

from ike.db.locks import advisory_unlock, try_advisory_lock
from ike.db.models import ReportJob, User
from ike.db.session import SessionLocal
from ike.reports.workflow import ReportGraphService
from ike.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="ike.build_report", bind=True, max_retries=1)
def build_report(self, report_id: str) -> None:
    """Build one long-form report without importing ingestion/Docling code."""

    rid = UUID(report_id)
    with SessionLocal() as lock_db:
        if not try_advisory_lock(lock_db, rid):
            logger.warning(
                "report_already_running",
                extra={"report_id": report_id},
            )
            return
        try:
            with SessionLocal() as db:
                job = db.get(ReportJob, rid)
                if not job:
                    raise ValueError(f"Unknown report {report_id}")
                if job.status == "complete":
                    return
                job.status = "running"
                job.progress_stage = "prepare"
                job.progress_percent = 5
                job.progress_message = "Starting analysis"
                job.error = None
                job.started_at = datetime.now(UTC)
                job.completed_at = None
                db.commit()

            try:
                with SessionLocal() as db:
                    job = db.get(ReportJob, rid)
                    assert job is not None
                    user = db.get(User, job.created_by)
                    if not user or not user.is_active:
                        raise ValueError(
                            "Report owner is missing or inactive; "
                            "refusing to access source documents"
                        )

                    def progress(stage: str, message: str, percent: int) -> None:
                        job.progress_stage = stage
                        job.progress_percent = max(0, min(100, percent))
                        job.progress_message = message[:500]
                        db.commit()

                    workflow = ReportGraphService(
                        db,
                        user,
                        progress=progress,
                    )
                    state = workflow.run(
                        job.title,
                        job.objective,
                        job.document_ids,
                    )
                    job.result_markdown = state["result_markdown"]
                    job.citations = workflow.citation_payload(state)
                    job.input_tokens = state.get("input_tokens", 0)
                    job.output_tokens = state.get("output_tokens", 0)
                    job.status = "complete"
                    job.progress_stage = "complete"
                    job.progress_percent = 100
                    job.progress_message = "Analysis complete"
                    job.completed_at = datetime.now(UTC)
                    db.commit()
            except Exception as exc:
                logger.exception(
                    "report_failed",
                    extra={"report_id": report_id},
                )
                with SessionLocal() as db:
                    job = db.get(ReportJob, rid)
                    if job:
                        job.status = "failed"
                        job.progress_stage = "failed"
                        job.progress_message = "Analysis failed"
                        job.error = str(exc)[:8000]
                        job.completed_at = datetime.now(UTC)
                        db.commit()
                raise
        finally:
            advisory_unlock(lock_db, rid)
