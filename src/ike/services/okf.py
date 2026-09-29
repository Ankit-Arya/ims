from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from ike.db.models import Document


OKF_VERSION = "0.3"


class ChunkLike(Protocol):
    ordinal: int
    page_from: int | None
    page_to: int | None
    section_path: list[str]
    content_kind: str
    text: str
    contextual_text: str
    metadata: dict


_TOPIC_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_TOPIC_STOP = {
    # Structural boilerplate / organisation-name noise.
    "part", "section", "chapter", "annexure", "table", "contents", "page", "pages", "item", "items",
    "general", "dmrc", "delhi", "metro", "rail", "railway", "corporation", "limited", "ltd",
    "document", "documents", "procedure", "procedures", "rules", "rule",
    # Function words and conversational prose are not document topics.
    "the", "and", "for", "with", "from", "into", "this", "that", "these", "those", "shall", "will",
    "would", "should", "could", "can", "may", "must", "have", "has", "had", "having", "are", "is",
    "was", "were", "be", "been", "being", "do", "does", "did", "done", "not", "only", "also",
    "any", "all", "both", "each", "either", "neither", "other", "another", "same", "such", "more",
    "most", "less", "than", "then", "there", "here", "where", "when", "which", "what", "who", "whose",
    "why", "how", "under", "over", "above", "below", "between", "before", "after", "during", "through",
    "until", "upon", "about", "against", "without", "within", "outside", "inside", "per", "via",
    "their", "them", "they", "his", "her", "hers", "him", "its", "our", "ours", "your", "yours",
    "one", "two", "three", "first", "second", "third", "etc", "note", "notes", "details", "date", "time",
    "year", "years", "month", "months", "day", "days", "provided", "provide", "provides", "providing",
    "required", "require", "requires", "needed", "need", "needs", "sent", "came", "left", "asked",
    "question", "questions", "answer", "answers", "yes", "no", "attend", "attended", "attending",
    "mention", "mentioned", "mentioning", "travelled", "traveled", "back",
}


def document_topic_terms(chunks: Iterable[ChunkLike], *, limit: int = 240) -> list[str]:
    """Build a compact structural topic index for broad multi-topic documents.

    Headings are strong document-level signals. Table bodies are much noisier: a single
    mailing list, example row or address table must not turn a place/person/value into a
    document topic. Table terms are therefore promoted only when they recur across multiple
    table chunks (or are already supported by a heading).
    """

    chunks = list(chunks)
    heading_counts: Counter[str] = Counter()
    table_chunk_frequency: Counter[str] = Counter()

    for chunk in chunks:
        for raw in chunk.section_path or []:
            for match in _TOPIC_TOKEN_RE.finditer(raw):
                token = match.group(0).casefold().strip("_-")
                if token not in _TOPIC_STOP:
                    heading_counts[token] += 4

        if str(chunk.content_kind).casefold() == "table":
            sample = (chunk.contextual_text or chunk.text or "")[:900]
            table_terms = {
                match.group(0).casefold().strip("_-")
                for match in _TOPIC_TOKEN_RE.finditer(sample)
                if match.group(0).casefold().strip("_-") not in _TOPIC_STOP
            }
            for token in table_terms:
                table_chunk_frequency[token] += 1

    counts = Counter(heading_counts)
    for token, frequency in table_chunk_frequency.items():
        if token in heading_counts or frequency >= 3:
            counts[token] += min(frequency, 8)

    return [token for token, _count in counts.most_common(max(1, limit))]


def _json(value) -> str:
    # JSON is valid YAML 1.2 and avoids an additional dependency.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
def _tags(document: Document, profile: dict) -> list[str]:
    values: list[str] = []
    for value in (document.source_role, document.family_key, profile.get("document_type")):
        if value:
            values.append(str(value))
    values.extend(str(v) for v in profile.get("primary_line_codes", [])[:4])
    values.extend(str(v) for v in profile.get("primary_rolling_stock", [])[:4])
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        folded = value.casefold()
        if value and folded not in seen:
            seen.add(folded)
            output.append(value)
    return output


def render_document_concept(document: Document, chunks: Iterable[ChunkLike], profile: dict) -> str:
    chunks = list(chunks)
    topic_terms = document_topic_terms(chunks)
    now = _iso_now()
    status = "deprecated" if document.lifecycle_status == "archived" else "stable"
    source = {
        "id": "source-pdf",
        "resource": f"ims://originals/{document.checksum_sha256}",
        "title": document.original_filename,
    }
    if document.authority:
        source["author"] = f"team:{document.authority.strip().replace(' ', '-').lower()}"
    description = (
        f"IMS knowledge representation for {document.title}; generated from the ingested source PDF "
        f"and its extracted document/chunk metadata."
    )
    lines = [
        "---",
        f"type: {_json('IMS Document Knowledge')}",
        f"title: {_json(document.title)}",
        f"description: {_json(description)}",
        f"resource: {_json(f'ims://documents/{document.id}')}",
        f"tags: {_json(_tags(document, profile))}",
        f"status: {_json(status)}",
        f"generated: {_json({'by': 'process:ims-ingestion', 'at': now})}",
        f"sources: {_json([source])}",
        f"ims_document_id: {_json(str(document.id))}",
        f"ims_revision: {_json(document.revision)}",
        f"ims_authority: {_json(document.authority)}",
        f"ims_family_key: {_json(document.family_key)}",
        f"ims_source_role: {_json(document.source_role)}",
        f"ims_effective_from: {_json(document.effective_from.isoformat() if document.effective_from else None)}",
        f"ims_effective_to: {_json(document.effective_to.isoformat() if document.effective_to else None)}",
        f"ims_operational_profile: {_json(profile)}",
        f"ims_topic_terms: {_json(topic_terms)}",
        "---",
        "",
        "# Source identity",
        "",
        f"- File: `{document.original_filename}`",
        f"- SHA-256: `{document.checksum_sha256}`",
        f"- Revision: `{document.revision or 'unspecified'}`",
        f"- Authority: `{document.authority or 'unspecified'}`",
        f"- Pages: `{document.page_count or 'unknown'}`",
        "",
        "# Operational applicability",
        "",
        f"- Primary lines: {', '.join(profile.get('primary_line_codes', [])) or 'unspecified'}",
        f"- Primary rolling stock: {', '.join(profile.get('primary_rolling_stock', [])) or 'unspecified'}",
        f"- Document type: {profile.get('document_type') or 'unknown'}",
        "",
        "# Indexed structure",
        "",
    ]
    section_rows = []
    for chunk in chunks:
        section = " / ".join(chunk.section_path or []) or "Unsectioned"
        page = str(chunk.page_from or "?")
        if chunk.page_to and chunk.page_to != chunk.page_from:
            page = f"{page}-{chunk.page_to}"
        section_rows.append((chunk.ordinal, page, chunk.content_kind, section))
    for ordinal, page, kind, section in section_rows[:200]:
        lines.append(f"- Chunk {ordinal}: page {page}; kind `{kind}`; section `{section}`")
    if len(section_rows) > 200:
        lines.append(f"- ... {len(section_rows) - 200} additional chunks omitted from this human-readable index")
    lines.extend([
        "",
        "# Notes",
        "",
        "This OKF concept is retrieval metadata, not a substitute for the authoritative source PDF. "
        "IMS should continue citing the underlying document passages for factual answers.",
        "",
    ])
    return "\n".join(lines)
def write_document_concept(
    document: Document,
    chunks: Iterable[ChunkLike],
    profile: dict,
    bundle_root: Path,
) -> Path:
    documents_dir = bundle_root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    index = bundle_root / "index.md"
    if not index.exists():
        index.write_text(
            f"---\nokf_version: \"{OKF_VERSION}\"\n---\n\n"
            "# IMS Open Knowledge Format bundle\n\n"
            "Machine-generated IMS document knowledge. Source PDFs remain authoritative.\n",
            encoding="utf-8",
        )

    path = documents_dir / f"{document.id}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_document_concept(document, chunks, profile), encoding="utf-8")
    tmp.replace(path)
    return path
