from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from ike.db.models import Document


class ChunkLike(Protocol):
    ordinal: int
    page_from: int | None
    page_to: int | None
    section_path: list[str]
    content_kind: str
    text: str
    contextual_text: str
    metadata: dict


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
            "---\nokf_version: \"0.2\"\n---\n\n"
            "# IMS Open Knowledge Format bundle\n\n"
            "Machine-generated IMS document knowledge. Source PDFs remain authoritative.\n",
            encoding="utf-8",
        )

    path = documents_dir / f"{document.id}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_document_concept(document, chunks, profile), encoding="utf-8")
    tmp.replace(path)
    return path
