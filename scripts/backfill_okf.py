from __future__ import annotations

import logging

from sqlalchemy import select

from ike.core.config import get_settings
from ike.db.models import Chunk, Document
from ike.db.session import SessionLocal
from ike.services.metadata_enrichment import infer_operational_profile
from ike.services.okf import OKF_VERSION, document_topic_terms, write_document_concept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("okf_backfill")


def backfill() -> tuple[int, int]:
    settings = get_settings()
    completed = 0
    failed = 0
    with SessionLocal() as db:
        documents = list(
            db.scalars(
                select(Document)
                .where(Document.ingestion_status == "ready")
                .order_by(Document.created_at)
            )
        )
        for document in documents:
            try:
                chunks = list(
                    db.scalars(
                        select(Chunk)
                        .where(Chunk.document_id == document.id)
                        .order_by(Chunk.ordinal)
                    )
                )
                if not chunks:
                    logger.warning("skip_no_chunks document_id=%s", document.id)
                    failed += 1
                    continue
                metadata = dict(document.extra_metadata or {})
                profile = dict(metadata.get("operational_profile") or {})
                if settings.okf_profile_enrichment_enabled and not profile:
                    profile = infer_operational_profile(
                        document.title,
                        document.original_filename,
                        [chunk.contextual_text for chunk in chunks[:20]],
                    )
                    metadata["operational_profile"] = profile

                okf_path = write_document_concept(
                    document,
                    chunks,
                    profile,
                    settings.okf_bundle_dir,
                )
                metadata["okf"] = {
                    "status": "ready",
                    "version": OKF_VERSION,
                    "concept_path": str(okf_path),
                    "topic_terms": document_topic_terms(chunks),
                    "backfilled": True,
                }
                document.extra_metadata = metadata
                db.commit()
                completed += 1
                logger.info("okf_backfilled document_id=%s path=%s", document.id, okf_path)
            except Exception as exc:
                db.rollback()
                failed += 1
                logger.exception("okf_backfill_failed document_id=%s error=%s", document.id, exc)
        return completed, failed


if __name__ == "__main__":
    ok, failed = backfill()
    print(f"OKF backfill complete: ready={ok} failed={failed}")
    raise SystemExit(1 if failed else 0)
