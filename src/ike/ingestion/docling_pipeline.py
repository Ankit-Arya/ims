import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    OcrMode,
    ThreadedPdfPipelineOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from ike.core.config import get_settings


@dataclass(slots=True)
class ParsedChunk:
    ordinal: int
    text: str
    contextual_text: str
    page_from: int | None
    page_to: int | None
    section_path: list[str]
    content_kind: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ParsedDocument:
    page_count: int
    chunks: list[ParsedChunk]
    canonical: dict[str, Any]


class DoclingPipeline:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.docling_artifacts_path.exists():
            raise RuntimeError(
                f"Docling model artifacts are missing at {settings.docling_artifacts_path}. "
                "Run the model-bootstrap service before ingestion."
            )
        options = ThreadedPdfPipelineOptions(
            artifacts_path=settings.docling_artifacts_path,
            accelerator_options=AcceleratorOptions(
                num_threads=max(1, settings.docling_num_threads_per_document),
                device="cpu",
            ),
            ocr_batch_size=max(1, settings.docling_ocr_batch_size),
            layout_batch_size=max(1, settings.docling_layout_batch_size),
            table_batch_size=max(1, settings.docling_table_batch_size),
        )
        options.do_ocr = settings.docling_ocr
        options.do_table_structure = settings.docling_table_structure
        options.document_timeout = (
            None if settings.docling_timeout_seconds <= 0 else float(settings.docling_timeout_seconds)
        )
        options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
        # Required for Docling's font/style signal when inferring heading depth.
        options.generate_parsed_pages = True
        if settings.docling_table_structure:
            options.table_structure_options.mode = (
                TableFormerMode.ACCURATE if settings.docling_table_mode == "accurate" else TableFormerMode.FAST
            )
        if settings.docling_ocr:
            options.ocr_options = TesseractCliOcrOptions(
                lang=settings.ocr_language_list,
                mode=OcrMode(settings.docling_ocr_mode),
            )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
        )
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=settings.embedding_model,
            max_tokens=settings.chunk_max_tokens,
        )
        self.chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, repeat_table_header=True)

    def parse(self, source: Path) -> ParsedDocument:
        result = self.converter.convert(source, raises_on_error=False)

        # Docling can return PARTIAL_SUCCESS when the configured document timeout
        # is reached (or when one or more pages fail). For an operational knowledge
        # system, partial extraction must never be silently promoted to `ready`.
        # Fail closed so the existing good chunks remain untouched until a complete
        # reprocess succeeds.
        status = getattr(result, "status", None)
        has_timeout = bool(
            getattr(result, "has_timeout_errors", lambda: False)()
        )
        if status != ConversionStatus.SUCCESS:
            errors = getattr(result, "errors", None) or []
            messages = []
            for error in errors[:8]:
                message = getattr(error, "error_message", None) or str(error)
                page_no = getattr(error, "page_no", None)
                if page_no is not None:
                    message = f"page {page_no}: {message}"
                messages.append(message)
            detail = "; ".join(messages) if messages else "no detailed Docling error was supplied"
            if has_timeout:
                raise RuntimeError(
                    f"Docling conversion timed out and returned a partial document: {detail}. "
                    "Increase DOCLING_TIMEOUT_SECONDS or reduce ingestion concurrency, then reprocess."
                )
            raise RuntimeError(
                f"Docling conversion was incomplete (status={status}): {detail}. "
                "The document was not indexed because partial operational documents are not accepted."
            )

        doc = result.document
        chunks: list[ParsedChunk] = []
        for ordinal, chunk in enumerate(self.chunker.chunk(dl_doc=doc)):
            text = (chunk.text or "").strip()
            if not text:
                continue
            contextual = self.chunker.contextualize(chunk=chunk).strip()
            pages: list[int] = []
            labels: list[str] = []
            for item in getattr(chunk.meta, "doc_items", []) or []:
                label = getattr(item, "label", None)
                if label:
                    labels.append(str(label))
                for prov in getattr(item, "prov", []) or []:
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None:
                        pages.append(int(page_no))
            unique_pages = sorted(set(pages))
            headings = [str(x) for x in (getattr(chunk.meta, "headings", None) or [])]
            kind = "table" if any("table" in label.lower() for label in labels) else "text"
            meta = chunk.meta.export_json_dict() if hasattr(chunk.meta, "export_json_dict") else {}
            chunks.append(
                ParsedChunk(
                    ordinal=len(chunks),
                    text=text,
                    contextual_text=contextual or text,
                    page_from=unique_pages[0] if unique_pages else None,
                    page_to=unique_pages[-1] if unique_pages else None,
                    section_path=headings,
                    content_kind=kind,
                    metadata=meta,
                )
            )
        canonical = doc.export_to_dict()
        return ParsedDocument(page_count=len(doc.pages), chunks=chunks, canonical=canonical)

    @staticmethod
    def save_canonical(payload: dict[str, Any], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def get_docling_pipeline() -> DoclingPipeline:
    """Reuse heavyweight parser/tokenizer objects within each Celery worker process."""
    return DoclingPipeline()
