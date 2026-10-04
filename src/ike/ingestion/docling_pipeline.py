import json
import re
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


_STRUCTURAL_PREFIX_RE = re.compile(
    r"^(?:chapter|part|book|volume|appendix|annex(?:ure)?)\s*[-–—:]?\s*(?:[ivxlcdm]+|\d+|[a-z])\s*$",
    re.IGNORECASE,
)
_RULE_LIKE_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*(?:[.)-]|\s)")


def _clean_heading(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def structural_heading_aliases(canonical: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Recover split two-line structural headings from Docling canonical items.

    Some PDFs render a structural marker such as CHAPTER IV on one line and its
    descriptive title on the next. Docling can label both as section headers while
    assigning the second line a deeper level; HybridChunker may then omit that title
    from subsequent chunk heading paths. This repair is document-generic.
    """

    texts = canonical.get("texts") if isinstance(canonical, dict) else None
    if not isinstance(texts, list):
        return {}

    def page_of(item: dict[str, Any]) -> int | None:
        for prov in item.get("prov") or []:
            if isinstance(prov, dict) and prov.get("page_no") is not None:
                try:
                    return int(prov["page_no"])
                except (TypeError, ValueError):
                    return None
        return None

    aliases: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(texts[:-1]):
        if not isinstance(raw, dict) or raw.get("label") != "section_header":
            continue
        prefix = _clean_heading(raw.get("text") or raw.get("orig"))
        if not _STRUCTURAL_PREFIX_RE.fullmatch(prefix):
            continue
        try:
            prefix_level = int(raw.get("level") or 0)
        except (TypeError, ValueError):
            prefix_level = 0
        prefix_page = page_of(raw)

        for next_raw in texts[index + 1 : index + 5]:
            if not isinstance(next_raw, dict):
                break
            label = str(next_raw.get("label") or "")
            if label in {"page_header", "page_footer"}:
                continue
            if label != "section_header":
                break
            title = _clean_heading(next_raw.get("text") or next_raw.get("orig"))
            if not title or _STRUCTURAL_PREFIX_RE.fullmatch(title) or _RULE_LIKE_HEADING_RE.match(title):
                break
            try:
                title_level = int(next_raw.get("level") or 0)
            except (TypeError, ValueError):
                title_level = 0
            title_page = page_of(next_raw)
            if prefix_page is not None and title_page is not None and abs(title_page - prefix_page) > 1:
                break
            if prefix_level and title_level and title_level <= prefix_level:
                break
            if len(title) > 180:
                break
            aliases[prefix.casefold()] = (f"{prefix} {title}", title)
            break
    return aliases


def stitch_section_path(
    headings: list[str],
    aliases: dict[str, tuple[str, str]],
) -> list[str]:
    """Apply canonical two-line heading repair to one chunk path."""

    result: list[str] = []
    index = 0
    while index < len(headings):
        heading = _clean_heading(headings[index])
        if not heading:
            index += 1
            continue
        alias = aliases.get(heading.casefold())
        if alias is None:
            result.append(heading)
            index += 1
            continue

        combined, title = alias
        result.append(combined)
        if index + 1 < len(headings) and _clean_heading(headings[index + 1]).casefold() == title.casefold():
            index += 2
        else:
            index += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for heading in result:
        key = heading.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(heading)
    return deduped


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
        canonical = doc.export_to_dict()
        heading_aliases = structural_heading_aliases(canonical)
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
            headings = stitch_section_path(
                [str(x) for x in (getattr(chunk.meta, "headings", None) or [])],
                heading_aliases,
            )
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
        return ParsedDocument(page_count=len(doc.pages), chunks=chunks, canonical=canonical)

    @staticmethod
    def save_canonical(payload: dict[str, Any], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def get_docling_pipeline() -> DoclingPipeline:
    """Reuse heavyweight parser/tokenizer objects within each Celery worker process."""
    return DoclingPipeline()
