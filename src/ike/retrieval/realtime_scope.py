from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ike.db.models import Chunk, Document
from ike.retrieval.access import document_access_clause


def _canon(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


@dataclass(slots=True)
class RealtimeScope:
    document_ids: list[UUID] = field(default_factory=list)
    exact_ids: list[UUID] = field(default_factory=list)
    rs_ids: list[UUID] = field(default_factory=list)
    line_ids: list[UUID] = field(default_factory=list)
    network_ids: list[UUID] = field(default_factory=list)
    context_used: dict = field(default_factory=dict)

    def as_trace(self) -> dict:
        return {
            "context_used": self.context_used,
            "document_count": len(self.document_ids),
            "exact_count": len(self.exact_ids),
            "rolling_stock_count": len(self.rs_ids),
            "line_count": len(self.line_ids),
            "network_count": len(self.network_ids),
        }


def resolve_realtime_scope(
    db: Session,
    user,
    context: dict | None,
    *,
    max_documents: int = 60,
) -> RealtimeScope:
    """Resolve a recall-safe physical operating scope from existing metadata."""
    context = context or {}
    line = _canon(context.get("line"))
    stock = _canon(context.get("rolling_stock"))
    if not line and not stock:
        return RealtimeScope(context_used={})

    rows = db.execute(
        select(Document.id, Document.extra_metadata, Document.source_role, Document.lifecycle_status)
        .where(document_access_clause(user), Document.ingestion_status == "ready")
    ).all()

    exact: list[UUID] = []
    rs_only: list[UUID] = []
    line_only: list[UUID] = []
    secondary: list[UUID] = []
    network: list[UUID] = []
    generic_types = {"rules", "emergency", "operating_manual", "troubleshooting"}

    for document_id, extra_metadata, source_role, lifecycle_status in rows:
        if lifecycle_status not in {None, "active"}:
            continue
        profile = (extra_metadata or {}).get("operational_profile") or {}
        primary_lines = {_canon(str(v)) for v in profile.get("primary_line_codes", []) if v}
        primary_stocks = {_canon(str(v)) for v in profile.get("primary_rolling_stock", []) if v}
        lines = {_canon(str(v)) for v in profile.get("line_codes", []) if v}
        stocks = {_canon(str(v)) for v in profile.get("rolling_stock", []) if v}
        primary_line_match = bool(line and line in primary_lines)
        primary_stock_match = bool(stock and stock in primary_stocks)
        line_match = bool(line and line in lines)
        stock_match = bool(stock and stock in stocks)
        known_line_mismatch = bool(line and primary_lines and not primary_line_match)
        known_stock_mismatch = bool(stock and primary_stocks and not primary_stock_match)
        known_selected = (bool(primary_lines) if line else False) or (bool(primary_stocks) if stock else False)
        all_known_match = not known_line_mismatch and not known_stock_mismatch

        if known_selected and all_known_match and (primary_line_match or primary_stock_match):
            exact.append(document_id)
            continue
        if primary_stock_match:
            rs_only.append(document_id)
            continue
        if primary_line_match:
            line_only.append(document_id)
            continue
        if stock_match or line_match:
            secondary.append(document_id)
            continue

        doc_type = str(profile.get("document_type") or "").casefold()
        if not lines and not stocks and (doc_type in generic_types or source_role == "governing_reference"):
            network.append(document_id)

    ordered: list[UUID] = []
    lane_limits = ((exact, 20), (rs_only, 12), (line_only, 12), (secondary, 12), (network, 16))
    for lane, lane_limit in lane_limits:
        for document_id in lane[:lane_limit]:
            if document_id not in ordered:
                ordered.append(document_id)
                if len(ordered) >= max_documents:
                    break
        if len(ordered) >= max_documents:
            break

    return RealtimeScope(
        document_ids=ordered,
        exact_ids=exact,
        rs_ids=rs_only,
        line_ids=line_only,
        network_ids=network,
        context_used={k: v for k, v in context.items() if v},
    )


def resolve_realtime_recovery_boost(
    db: Session,
    user,
    context: dict | None,
    queries: list[str],
    *,
    max_documents: int = 24,
) -> list[UUID]:
    """Find a bounded applicability-safe recovery scope from titles and corpus content.

    Title matching is excellent for named procedures (for example high-wind circulars) but
    weak for terse equipment faults whose authoritative manual title does not repeat the
    equipment acronym.  Strong technical tokens therefore also vote using FTS chunk hits.
    """
    context = context or {}
    line = _canon(context.get("line"))
    stock = _canon(context.get("rolling_stock"))
    stop = {
        "train", "procedure", "operating", "operation", "action", "manual", "role",
        "reporting", "steps", "corrective", "troubleshooting", "applicable", "technical",
        "open", "opened", "showing", "available", "green", "signal", "fault", "failure",
        "reset", "close", "closing", "inform", "instruction",
    }

    token_weights: dict[str, int] = {}
    for query in queries:
        for raw in re.findall(r"[A-Za-z0-9]+", query):
            token = raw.casefold()
            technical = (raw.isupper() and len(raw) >= 3) or any(ch.isdigit() for ch in raw)
            if token in stop:
                continue
            if len(token) >= 4 or technical:
                token_weights[token] = max(token_weights.get(token, 0), 3 if technical else 1)
            if len(token_weights) >= 10:
                break
        if len(token_weights) >= 10:
            break
    if not token_weights:
        return []

    rows = db.execute(
        select(Document.id, Document.title, Document.original_filename, Document.extra_metadata)
        .where(document_access_clause(user), Document.ingestion_status == "ready")
    ).all()

    eligible: dict[UUID, tuple[str, str]] = {}
    scores: dict[UUID, int] = {}
    title_hits: dict[UUID, int] = {}
    for document_id, title, filename, extra_metadata in rows:
        profile = (extra_metadata or {}).get("operational_profile") or {}
        primary_lines = {_canon(str(v)) for v in profile.get("primary_line_codes", []) if v}
        primary_stocks = {_canon(str(v)) for v in profile.get("primary_rolling_stock", []) if v}
        lines = {_canon(str(v)) for v in profile.get("line_codes", []) if v}
        stocks = {_canon(str(v)) for v in profile.get("rolling_stock", []) if v}
        if line and primary_lines and line not in primary_lines:
            continue
        if stock and primary_stocks and stock not in primary_stocks:
            continue

        title_value = title or ""
        filename_value = filename or ""
        eligible[document_id] = (title_value, filename_value)
        if stock and stock in primary_stocks:
            scores[document_id] = scores.get(document_id, 0) + 10
        elif stock and stock in stocks:
            scores[document_id] = scores.get(document_id, 0) + 5
        if line and line in primary_lines:
            scores[document_id] = scores.get(document_id, 0) + 8
        elif line and line in lines:
            scores[document_id] = scores.get(document_id, 0) + 4
        haystack = f"{title_value} {filename_value}".casefold()
        for token, weight in token_weights.items():
            if token in haystack:
                scores[document_id] = scores.get(document_id, 0) + (weight * 4)
                title_hits[document_id] = title_hits.get(document_id, 0) + 1

    eligible_ids = list(eligible)
    if eligible_ids:
        for token, weight in token_weights.items():
            tsquery = func.plainto_tsquery("simple", token)
            content_rows = db.execute(
                select(Chunk.document_id, func.count(Chunk.id))
                .where(
                    Chunk.document_id.in_(eligible_ids),
                    Chunk.search_vector.op("@@")(tsquery),
                )
                .group_by(Chunk.document_id)
            ).all()
            for document_id, count in content_rows:
                # Distinct technical concepts matter more than raw repetition, while several
                # corroborating chunks are useful evidence that a manual genuinely covers it.
                scores[document_id] = scores.get(document_id, 0) + weight * (2 + min(int(count), 6))

    best_title_hits = max(title_hits.values(), default=0)
    if best_title_hits >= 2:
        title_floor = max(2, best_title_hits - 1)
        scores = {
            document_id: score
            for document_id, score in scores.items()
            if title_hits.get(document_id, 0) >= title_floor
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            title_hits.get(item[0], 0),
        ),
        reverse=True,
    )
    if not ranked:
        return []

    best_score = ranked[0][1]
    # Keep only documents reasonably close to the strongest structural/content match.  This
    # avoids turning a generic acronym mention into another broad 20-document recovery.
    floor = max(4, int(best_score * 0.55))
    ranked = [item for item in ranked if item[1] >= floor]
    technical_present = any(weight >= 3 for weight in token_weights.values())
    effective_limit = min(max_documents, 8 if technical_present else max_documents)
    return [document_id for document_id, _score in ranked[:effective_limit]]
