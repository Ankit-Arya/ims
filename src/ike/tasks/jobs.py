import logging
from datetime import UTC, datetime
from uuid import UUID

from ike.db.locks import advisory_unlock, try_advisory_lock
from ike.db.models import Document, ReportJob, User
from ike.db.session import SessionLocal
from ike.ingestion.processor import process_document
from ike.tasks.celery_app import celery_app
from ike.workflows.report_graph import ReportGraphService

logger = logging.getLogger(__name__)


def _mark_retry_scheduled(document_id: UUID, *, attempt: int, error: Exception) -> None:
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document:
            document.ingestion_status = "queued"
            document.ingestion_error = (
                f"Attempt {attempt} failed and will retry automatically: {error}"
            )[:8000]
            db.commit()


@celery_app.task(name="ike.ingest_document", bind=True, max_retries=2)
def ingest_document(self, document_id: str) -> None:
    """Ingest a document with explicit retry state.

    `process_document` records the concrete failure. If Celery still has retries
    available we immediately move the visible document state back to `queued` so
    users do not see a terminal failure while a retry is pending.
    """

    did = UUID(document_id)
    with SessionLocal() as lock_db:
        if not try_advisory_lock(lock_db, did):
            logger.warning("document_ingestion_already_running", extra={"document_id": document_id})
            return
        try:
            try:
                process_document(did)
            except Exception as exc:
                if self.request.retries < self.max_retries:
                    attempt = int(self.request.retries) + 1
                    _mark_retry_scheduled(did, attempt=attempt, error=exc)
                    countdown = min(60, 5 * (2 ** int(self.request.retries)))
                    logger.warning(
                        "document_ingestion_retry_scheduled",
                        extra={"document_id": document_id, "attempt": attempt, "countdown": countdown},
                    )
                    raise self.retry(exc=exc, countdown=countdown)
                raise
        finally:
            advisory_unlock(lock_db, did)


@celery_app.task(name="ike.build_report", bind=True, max_retries=1)
def build_report(self, report_id: str) -> None:
    rid = UUID(report_id)
    with SessionLocal() as lock_db:
        if not try_advisory_lock(lock_db, rid):
            logger.warning("report_already_running", extra={"report_id": report_id})
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
                        raise ValueError("Report owner is missing or inactive; refusing to access source documents")
                    def progress(stage: str, message: str, percent: int) -> None:
                        job.progress_stage = stage
                        job.progress_percent = max(0, min(100, percent))
                        job.progress_message = message[:500]
                        db.commit()

                    workflow = ReportGraphService(db, user, progress=progress)
                    state = workflow.run(job.title, job.objective, job.document_ids)
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
                logger.exception("report_failed", extra={"report_id": report_id})
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
