from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import Document, User
from ike.retrieval.access import document_access_clause
from ike.retrieval.types import Evidence
from ike.schemas.visuals import VisualEvidence

_VISUAL_CUE_RE = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d*|diagram|schematic|flow\s*chart|drawing|layout|illustrat(?:ed|ion)|"
    r"shown\s+(?:below|above|in)|panel|switch\s+location|where\s+is\s+.+located)\b",
    re.IGNORECASE,
)
_TABLE_CUE_RE = re.compile(r"\b(table|matrix|rows?|columns?)\b", re.IGNORECASE)


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _extract_bbox(metadata: dict[str, Any], page: int) -> dict[str, Any] | None:
    """Best-effort extraction of Docling provenance bbox from existing chunk metadata."""
    for item in _walk_dicts(metadata):
        page_no = item.get("page_no") or item.get("page")
        if page_no is not None:
            try:
                if int(page_no) != int(page):
                    continue
            except (TypeError, ValueError):
                continue
        bbox = item.get("bbox")
        if isinstance(bbox, dict) and all(key in bbox for key in ("l", "t", "r", "b")):
            return {
                "l": float(bbox["l"]),
                "t": float(bbox["t"]),
                "r": float(bbox["r"]),
                "b": float(bbox["b"]),
                "coord_origin": str(bbox.get("coord_origin") or item.get("coord_origin") or "TOPLEFT"),
            }
    return None



def _bbox_from_provenance(value: Any, page: int) -> dict[str, Any] | None:
    for item in _walk_dicts(value):
        page_no = item.get("page_no") or item.get("page")
        if page_no is not None:
            try:
                if int(page_no) != int(page):
                    continue
            except (TypeError, ValueError):
                continue
        bbox = item.get("bbox")
        if isinstance(bbox, dict) and all(key in bbox for key in ("l", "t", "r", "b")):
            return {
                "l": float(bbox["l"]),
                "t": float(bbox["t"]),
                "r": float(bbox["r"]),
                "b": float(bbox["b"]),
                "coord_origin": str(bbox.get("coord_origin") or item.get("coord_origin") or "TOPLEFT"),
            }
    return None


def _canonical_visual_bbox(document: Document, page: int, kind: str, evidence_text: str) -> dict[str, Any] | None:
    """Best-effort Docling canonical-JSON visual provenance lookup.

    Existing 0.4.x ingestions already retain canonical JSON. This function reads it only
    for retrieved pages when a visual is useful; it does not extract/embed the full corpus.
    """
    if not document.parsed_path:
        return None
    path = Path(document.parsed_path)
    if not path.is_file():
        return None
    try:
        canonical = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    reference_match = re.search(r"\b(?:fig(?:ure)?\.?\s*)?([A-Za-z]?\d+(?:[.\-]\d+)*)\b", evidence_text, re.I)
    reference = reference_match.group(1).casefold() if reference_match else ""
    expected_labels = {"table"} if kind == "table" else {"picture", "figure", "diagram", "image"}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in _walk_dicts(canonical):
        label = str(item.get("label") or item.get("type") or item.get("name") or "").casefold()
        text_blob = json.dumps({k: v for k, v in item.items() if k in {"text", "caption", "captions", "name", "label"}}, ensure_ascii=False).casefold()
        label_score = 4 if any(token in label for token in expected_labels) else 0
        ref_score = 5 if reference and reference in text_blob else 0
        bbox = _bbox_from_provenance(item.get("prov") or item.get("provenance") or item, page)
        if bbox is None:
            continue
        # Prefer actual visual/table objects over arbitrary text boxes. If a figure
        # number/caption is explicitly referenced, that is even stronger.
        score = label_score + ref_score
        if score > 0:
            ranked.append((score, bbox))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[0][1]

def create_visual_token(*, document_id: UUID, page: int, bbox: dict[str, Any] | None, kind: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "typ": "ims-visual",
        "document_id": str(document_id),
        "page": int(page),
        "bbox": bbox,
        "kind": kind,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.visual_token_ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, settings.app_secret_key, algorithm="HS256")


def decode_visual_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    payload = jwt.decode(token, settings.app_secret_key, algorithms=["HS256"])
    if payload.get("typ") != "ims-visual":
        raise ValueError("Invalid visual token")
    return payload


def resolve_visual_evidence(
    db: Session,
    user: User,
    question: str,
    evidence: list[Evidence],
) -> list[VisualEvidence]:
    settings = get_settings()
    if not settings.visual_evidence_enabled or not evidence:
        return []

    explicit_visual_request = bool(_VISUAL_CUE_RE.search(question))
    explicit_table_request = bool(_TABLE_CUE_RE.search(question))
    resolved: list[VisualEvidence] = []
    seen: set[tuple[UUID, int, str]] = set()

    for item in evidence:
        candidate = item.candidate
        if candidate.page_from is None:
            continue
        text = f"{' / '.join(candidate.section_path)}\n{candidate.text}"
        is_table = candidate.content_kind.lower() == "table"
        # Text-first default: a source merely mentioning a panel/figure is not enough.
        # Auto-attach only when the user asked for a visual, or a cited table directly
        # answers an explicit table/schedule/limits-style request.
        if not (explicit_visual_request or (is_table and explicit_table_request)):
            continue

        kind = "table" if is_table else ("diagram" if re.search(r"diagram|schematic|flow\s*chart|layout", text, re.I) else "figure")
        pages = [candidate.page_from]
        # If source wording says the visual is below/next, inspect one nearby page lazily.
        if settings.visual_nearby_pages > 0 and re.search(r"shown\s+below|following\s+(?:figure|diagram|table)|next\s+page", text, re.I):
            pages.append(candidate.page_from + 1)

        for page in pages:
            key = (candidate.document_id, int(page), kind)
            if key in seen:
                continue
            document = db.scalar(
                select(Document).where(Document.id == candidate.document_id, document_access_clause(user))
            )
            if not document or not document.page_count or page < 1 or page > document.page_count:
                continue
            bbox = _extract_bbox(candidate.source_metadata, candidate.page_from) if page == candidate.page_from else None
            if bbox is None:
                bbox = _canonical_visual_bbox(document, page, kind, text)
            token = create_visual_token(
                document_id=candidate.document_id,
                page=page,
                bbox=bbox,
                kind=kind,
            )
            section = candidate.section_path[-1] if candidate.section_path else "Relevant source page"
            resolved.append(
                VisualEvidence(
                    document_id=candidate.document_id,
                    document_title=candidate.document_title,
                    page=page,
                    kind=kind,
                    caption=f"{section} · page {page}",
                    token=token,
                    evidence_id=item.evidence_id,
                )
            )
            seen.add(key)
            if len(resolved) >= settings.visual_max_items:
                return resolved
    return resolved


def render_visual_to_cache(db: Session, user: User, token: str) -> Path:
    """Render a signed, ACL-checked source page/crop to the deterministic cache."""
    payload = decode_visual_token(token)
    document_id = UUID(str(payload["document_id"]))
    page_number = int(payload["page"])
    document = db.scalar(select(Document).where(Document.id == document_id, document_access_clause(user)))
    if not document:
        raise PermissionError("Document is not accessible")
    if not document.page_count or page_number < 1 or page_number > document.page_count:
        raise ValueError("Invalid source page")

    source = Path(document.storage_path)
    if not source.is_file():
        raise FileNotFoundError("Original PDF is unavailable")

    settings = get_settings()
    settings.visual_cache_dir.mkdir(parents=True, exist_ok=True)
    bbox = payload.get("bbox") if isinstance(payload.get("bbox"), dict) else None
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "checksum": document.checksum_sha256,
                "page": page_number,
                "bbox": bbox,
                "dpi": settings.visual_render_dpi,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    target_dir = settings.visual_cache_dir / document.checksum_sha256
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"p{page_number}-{cache_key}.png"
    if target.exists():
        return target

    # Lazy import keeps ordinary API startup/tests independent of PDF rendering.
    import fitz  # type: ignore

    pdf = fitz.open(source)
    try:
        page = pdf.load_page(page_number - 1)
        clip = None
        if bbox:
            try:
                left, top, right, bottom = (float(bbox[k]) for k in ("l", "t", "r", "b"))
                origin = str(bbox.get("coord_origin") or "TOPLEFT").upper()
                if "BOTTOM" in origin:
                    y0 = page.rect.height - top
                    y1 = page.rect.height - bottom
                else:
                    y0, y1 = top, bottom
                clip = fitz.Rect(min(left, right), min(y0, y1), max(left, right), max(y0, y1))
                clip = clip & page.rect
                # Tiny/invalid boxes are less useful than a full source page.
                if clip.width < 40 or clip.height < 40:
                    clip = None
            except Exception:
                clip = None
        scale = max(1.0, settings.visual_render_dpi / 72.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pix.save(target)
    finally:
        pdf.close()
    return target
