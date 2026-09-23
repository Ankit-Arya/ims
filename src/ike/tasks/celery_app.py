from celery import Celery

from ike.core.config import get_settings

settings = get_settings()
visibility_timeout = settings.task_visibility_timeout_seconds

celery_app = Celery("ike", broker=settings.redis_url, backend=settings.redis_url, include=["ike.tasks.jobs"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": visibility_timeout},
    result_backend_transport_options={"visibility_timeout": visibility_timeout},
    visibility_timeout=visibility_timeout,
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "ike.ingest_document": {"queue": "ingestion"},
        "ike.build_report": {"queue": "reports"},
    },
)
