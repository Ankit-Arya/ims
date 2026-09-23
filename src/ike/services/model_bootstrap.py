"""Prefetch Docling model artifacts into the persistent model volume.

PDF ingestion must not depend on live internet/DNS availability. This bootstrap is
run as a one-shot Docker Compose service before ingestion workers start. A sentinel
prevents repeat network calls after the model volume has been initialized.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ike.core.config import get_settings
from ike.core.logging import configure_logging

logger = logging.getLogger(__name__)
_SENTINEL = ".ike-docling-models-ready-v1"


def ensure_docling_models() -> Path:
    settings = get_settings()
    target = settings.docling_artifacts_path
    target.mkdir(parents=True, exist_ok=True)
    sentinel = target / _SENTINEL
    if sentinel.exists():
        logger.info("docling_model_cache_ready", extra={"path": str(target)})
        return target

    logger.info("docling_model_cache_bootstrap_started", extra={"path": str(target)})
    try:
        subprocess.run(
            ["docling-tools", "models", "download", "-o", str(target), "layout", "tableformer"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Docling model bootstrap failed. Check internet/DNS once, then rerun "
            "`docker compose run --rm model-bootstrap`. No document needs to be re-uploaded."
        ) from exc
    sentinel.write_text("ready\n", encoding="utf-8")
    logger.info("docling_model_cache_bootstrap_complete", extra={"path": str(target)})
    return target


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    ensure_docling_models()


if __name__ == "__main__":
    main()
