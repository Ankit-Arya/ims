import logging
from uuid import UUID

from ike.db.locks import advisory_unlock, try_advisory_lock
from ike.db.models import Document
from ike.db.session import SessionLocal
from ike.ingestion.processor import process_document
from ike.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _mark_retry_scheduled(
    document_id: UUID,
    *,
    attempt: int,
    error: Exception,
) -> None:
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
    """Ingest one document with advisory locking and bounded retries."""

    did = UUID(document_id)
    with SessionLocal() as lock_db:
        if not try_advisory_lock(lock_db, did):
            logger.warning(
                "document_ingestion_already_running",
                extra={"document_id": document_id},
            )
            return
        try:
            try:
                process_document(did)
            except Exception as exc:
                if self.request.retries < self.max_retries:
                    attempt = int(self.request.retries) + 1
                    _mark_retry_scheduled(
                        did,
                        attempt=attempt,
                        error=exc,
                    )
                    countdown = min(
                        60,
                        5 * (2 ** int(self.request.retries)),
                    )
                    logger.warning(
                        "document_ingestion_retry_scheduled",
                        extra={
                            "document_id": document_id,
                            "attempt": attempt,
                            "countdown": countdown,
                        },
                    )
                    raise self.retry(
                        exc=exc,
                        countdown=countdown,
                    ) from exc
                raise
        finally:
            advisory_unlock(lock_db, did)
