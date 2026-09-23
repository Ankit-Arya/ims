import logging

import httpx
import redis
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from ike.core.config import get_settings
from ike.db.session import engine

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def ready() -> dict:
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        logger.exception("readiness_database_failed")
        checks["database"] = "error"
    try:
        redis.Redis.from_url(settings.redis_url, socket_timeout=2).ping()
        checks["queue"] = "ok"
    except Exception:
        logger.exception("readiness_queue_failed")
        checks["queue"] = "error"
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{settings.inference_url.rstrip('/')}/health")
            response.raise_for_status()
        checks["inference"] = "ok"
    except Exception:
        logger.exception("readiness_inference_failed")
        checks["inference"] = "error"
    if any(value != "ok" for value in checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", "checks": checks}
