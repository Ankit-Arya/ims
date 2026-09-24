import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import delete

from ike.core.config import get_settings
from ike.db.models import Chunk, Document
from ike.retrieval.corpus_intelligence import build_nodes_for_document
from ike.db.session import SessionLocal
from ike.ingestion.docling_pipeline import get_docling_pipeline
from ike.services.inference_client import InferenceClient
from ike.services.metadata_enrichment import infer_operational_profile
from ike.services.okf import write_document_concept
from ike.services.storage import LocalStorage

logger = logging.getLogger(__name__)


def process_document(document_id: UUID) -> None:
    settings = get_settings()
    storage = LocalStorage()
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if not document:
            raise ValueError(f"Unknown document {document_id}")
        document.ingestion_status = "processing"
        document.ingestion_error = None
        db.commit()

    try:
        pipeline = get_docling_pipeline()
        parsed = pipeline.parse(Path(document.storage_path))
        if not parsed.chunks:
            raise RuntimeError("Document parsing produced zero retrievable chunks")
        canonical_path = storage.parsed_json_path(document_id)
        pipeline.save_canonical(parsed.canonical, canonical_path)

        inference = InferenceClient.for_ingestion()
        embeddings: list[list[float]] = []
        batch_size = max(1, settings.ingestion_embedding_batch_size)
        texts = [c.contextual_text for c in parsed.chunks]
        for start in range(0, len(texts), batch_size):
            batch_vectors, _timing = inference.embed_documents(texts[start : start + batch_size])
            embeddings.extend(batch_vectors)
        if len(embeddings) != len(parsed.chunks):
            raise RuntimeError("Embedding service returned an unexpected vector count")
        if embeddings and len(embeddings[0]) != settings.embedding_dim:
            raise RuntimeError(f"Embedding dimension mismatch: expected {settings.embedding_dim}, got {len(embeddings[0])}")

        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                raise ValueError(f"Document disappeared during ingestion: {document_id}")
            db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            for parsed_chunk, vector in zip(parsed.chunks, embeddings, strict=True):
                db.add(
                    Chunk(
                        document_id=document_id,
                        ordinal=parsed_chunk.ordinal,
                        page_from=parsed_chunk.page_from,
                        page_to=parsed_chunk.page_to,
                        section_path=parsed_chunk.section_path,
                        content_kind=parsed_chunk.content_kind,
                        text=parsed_chunk.text,
                        contextual_text=parsed_chunk.contextual_text,
                        source_metadata=parsed_chunk.metadata,
                        embedding=vector,
                    )
                )
            document.page_count = parsed.page_count
            document.parsed_path = str(canonical_path)
            document.ingestion_status = "ready"
            document.ingestion_error = None
            if settings.okf_enabled:
                metadata = dict(document.extra_metadata or {})
                profile = dict(metadata.get("operational_profile") or {})
                if settings.okf_profile_enrichment_enabled and not profile:
                    profile = infer_operational_profile(
                        document.title,
                        document.original_filename,
                        [chunk.contextual_text for chunk in parsed.chunks[:20]],
                    )
                    metadata["operational_profile"] = profile
                try:
                    okf_path = write_document_concept(document, parsed.chunks, profile, settings.okf_bundle_dir)
                    metadata["okf"] = {"status": "ready", "version": "0.2", "concept_path": str(okf_path)}
                except Exception as okf_exc:
                    logger.exception("okf_generation_failed", extra={"document_id": str(document_id)})
                    metadata["okf"] = {"status": "failed", "version": "0.2", "error": str(okf_exc)[:1000]}
                document.extra_metadata = metadata
            db.commit()

        # 0.9 corpus intelligence is a secondary index. A failure here must never roll back
        # or invalidate a successfully parsed/chunked document; ordinary chunk retrieval
        # remains a complete fallback.
        try:
            with SessionLocal() as db:
                count = build_nodes_for_document(db, document_id, inference=inference)
                document = db.get(Document, document_id)
                if document:
                    metadata = dict(document.extra_metadata or {})
                    metadata["retrieval_intelligence"] = {"status": "ready", "nodes": count, "version": "0.9.1"}
                    document.extra_metadata = metadata
                db.commit()
        except Exception as intelligence_exc:
            logger.exception("retrieval_intelligence_ingestion_failed", extra={"document_id": str(document_id)})
            with SessionLocal() as db:
                document = db.get(Document, document_id)
                if document:
                    metadata = dict(document.extra_metadata or {})
                    metadata["retrieval_intelligence"] = {
                        "status": "failed",
                        "error": str(intelligence_exc)[:1000],
                        "version": "0.9.1",
                    }
                    document.extra_metadata = metadata
                    db.commit()

        logger.info("document_ingested", extra={"document_id": str(document_id), "chunks": len(parsed.chunks), "pages": parsed.page_count})
    except Exception as exc:
        logger.exception("document_ingestion_failed", extra={"document_id": str(document_id)})
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document:
                document.ingestion_status = "failed"
                document.ingestion_error = str(exc)[:8000]
                db.commit()
        raise
