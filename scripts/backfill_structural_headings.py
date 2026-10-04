"""Repair split two-line structural headings in already-ingested documents.

This backfill is metadata-only: it does not re-run OCR, Docling conversion, or embeddings.
It reads each document's saved canonical Docling JSON, repairs chunk section paths, then
rebuilds retrieval-intelligence/OKF metadata from the existing chunks and embeddings.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from ike.core.config import get_settings
from ike.db.models import Chunk, Document
from ike.db.session import SessionLocal
from ike.ingestion.docling_pipeline import structural_heading_aliases, stitch_section_path
from ike.retrieval.corpus_intelligence import INDEX_VERSION, build_nodes_for_document
from ike.services.okf import OKF_VERSION, document_topic_terms, write_document_concept

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("structural_heading_backfill")


def repair_document(document_id: UUID, *, dry_run: bool = False) -> tuple[int, int]:
    settings = get_settings()
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None or document.ingestion_status != "ready":
            return 0, 0

        canonical_path = Path(document.parsed_path or "")
        if not canonical_path.is_file():
            log.warning(
                "skip_missing_canonical document=%s parsed_path=%s",
                document_id,
                canonical_path,
            )
            return 0, 0

        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        aliases = structural_heading_aliases(canonical)
        if not aliases:
            return 0, 0

        chunks = list(
            db.scalars(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.ordinal)
            ).all()
        )
        changed = 0
        for chunk in chunks:
            repaired = stitch_section_path(list(chunk.section_path or []), aliases)
            if repaired != list(chunk.section_path or []):
                chunk.section_path = repaired
                changed += 1

        if not changed or dry_run:
            if dry_run:
                db.rollback()
            return changed, len(aliases)

        # Section paths changed but vectors did not. Rebuild only the lightweight
        # hierarchical/term index from existing chunk embeddings.
        db.flush()
        node_count = build_nodes_for_document(db, document_id)

        metadata = dict(document.extra_metadata or {})
        retrieval_meta = dict(metadata.get("retrieval_intelligence") or {})
        retrieval_meta.update(
            {
                "status": "ready",
                "nodes": node_count,
                "version": INDEX_VERSION,
                "embedding_source": "existing_chunk_centroids",
                "structural_heading_backfilled": True,
            }
        )
        metadata["retrieval_intelligence"] = retrieval_meta

        profile = dict(metadata.get("operational_profile") or {})
        try:
            okf_path = write_document_concept(
                document,
                chunks,
                profile,
                settings.okf_bundle_dir,
            )
        except Exception:
            # OKF is helpful metadata but must not make a section-path repair fail.
            log.exception("okf_refresh_failed document=%s", document_id)
        else:
            metadata["okf"] = {
                **dict(metadata.get("okf") or {}),
                "status": "ready",
                "version": OKF_VERSION,
                "concept_path": str(okf_path),
                "topic_terms": document_topic_terms(chunks),
                "structural_heading_backfilled": True,
            }

        document.extra_metadata = metadata
        db.commit()
        return changed, len(aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with SessionLocal() as db:
        stmt = (
            select(Document.id)
            .where(Document.ingestion_status == "ready")
            .order_by(Document.created_at)
        )
        if args.document_id:
            stmt = stmt.where(Document.id == UUID(args.document_id))
        if args.limit > 0:
            stmt = stmt.limit(args.limit)
        document_ids = list(db.scalars(stmt).all())

    documents_changed = 0
    chunks_changed = 0
    aliases_found = 0
    for position, document_id in enumerate(document_ids, start=1):
        try:
            changed, aliases = repair_document(document_id, dry_run=args.dry_run)
        except Exception:
            log.exception(
                "structural_heading_backfill_failed document=%s position=%s/%s",
                document_id,
                position,
                len(document_ids),
            )
            continue
        aliases_found += aliases
        chunks_changed += changed
        if changed:
            documents_changed += 1
            log.info(
                "structural_heading_repaired document=%s chunks=%s aliases=%s position=%s/%s",
                document_id,
                changed,
                aliases,
                position,
                len(document_ids),
            )

    print(
        "structural_heading_backfill "
        f"documents_changed={documents_changed} chunks_changed={chunks_changed} "
        f"aliases_found={aliases_found} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
