"""Lightweight task submission boundary used by the API process.

The API must not import worker task implementations. Worker tasks pull in heavy
PDF/ML dependencies (for example Docling) that intentionally are not installed
in the API image. Celery supports sending tasks by stable name, allowing the API
to remain a small web/control-plane service while workers own execution code.
"""

from uuid import UUID

from ike.tasks.celery_app import celery_app

INGEST_DOCUMENT_TASK = "ike.ingest_document"
BUILD_REPORT_TASK = "ike.build_report"


def enqueue_document_ingestion(document_id: UUID) -> None:
    celery_app.send_task(INGEST_DOCUMENT_TASK, args=[str(document_id)], queue="ingestion")


def enqueue_report_build(report_id: UUID) -> None:
    celery_app.send_task(BUILD_REPORT_TASK, args=[str(report_id)], queue="reports")
