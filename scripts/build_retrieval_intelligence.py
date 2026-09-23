"""Backfill IMS 0.10 corpus-intelligence nodes from existing ready chunks.

0.10 remains intentionally CPU-light: document/section vectors are derived from existing
chunk embeddings and terminology is stored on section nodes for FTS. No calls to the
document-embedding service are required.
"""
from __future__ import annotations

import argparse
import logging
import time
from uuid import UUID

from sqlalchemy import or_, select

from ike.db.models import Document
from ike.db.session import SessionLocal
from ike.retrieval.corpus_intelligence import INDEX_VERSION, build_nodes_for_document

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retrieval_intelligence_backfill")


def _needs_current_index():
    version = Document.extra_metadata["retrieval_intelligence"]["version"].astext
    status = Document.extra_metadata["retrieval_intelligence"]["status"].astext
    return or_(version.is_(None), version != INDEX_VERSION, status.is_(None), status != "ready")


def _pending_count() -> int:
    with SessionLocal() as db:
        stmt = (
            select(Document.id)
            .where(Document.ingestion_status == "ready", _needs_current_index())
        )
        return len(list(db.scalars(stmt).all()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", default=None)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="intentionally rebuild even documents already indexed with the current version",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with SessionLocal() as db:
        stmt = select(Document.id).where(Document.ingestion_status == "ready").order_by(Document.created_at)
        if args.document_id:
            stmt = stmt.where(Document.id == UUID(args.document_id))
        if not args.rebuild:
            # Missing or older 0.9.0 indexes are automatically upgraded. This makes repeated
            # --limit batches advance instead of reprocessing current 0.10 documents.
            stmt = stmt.where(_needs_current_index())
        if args.limit > 0:
            stmt = stmt.limit(args.limit)
        document_ids = list(db.scalars(stmt).all())

    total_nodes = 0
    completed = 0
    failed = 0
    batch_started = time.perf_counter()
    for position, document_id in enumerate(document_ids, start=1):
        started = time.perf_counter()
        with SessionLocal() as db:
            try:
                count = build_nodes_for_document(db, document_id)
                document = db.get(Document, document_id)
                metadata = dict(document.extra_metadata or {}) if document else {}
                metadata["retrieval_intelligence"] = {
                    "status": "ready",
                    "nodes": count,
                    "version": INDEX_VERSION,
                    "embedding_source": "existing_chunk_centroids",
                }
                if document:
                    document.extra_metadata = metadata
                db.commit()
                total_nodes += count
                completed += 1
                log.info(
                    "retrieval_intelligence_indexed document=%s nodes=%s position=%s/%s elapsed_s=%.2f",
                    document_id,
                    count,
                    position,
                    len(document_ids),
                    time.perf_counter() - started,
                )
            except Exception as exc:
                db.rollback()
                failed += 1
                log.exception("retrieval_intelligence_failed document=%s", document_id)
                with SessionLocal() as mark_db:
                    document = mark_db.get(Document, document_id)
                    if document:
                        metadata = dict(document.extra_metadata or {})
                        metadata["retrieval_intelligence"] = {
                            "status": "failed",
                            "error": str(exc)[:1000],
                            "version": INDEX_VERSION,
                        }
                        document.extra_metadata = metadata
                        mark_db.commit()

    pending = _pending_count()
    print(
        f"indexed_documents={completed} failed_documents={failed} "
        f"retrieval_nodes={total_nodes} pending_documents={pending} "
        f"elapsed_s={time.perf_counter() - batch_started:.2f}"
    )


if __name__ == "__main__":
    main()
