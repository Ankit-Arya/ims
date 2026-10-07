from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import Chunk, Document, User
from ike.retrieval.access import document_access_clause
from ike.retrieval.corpus_intelligence import CorpusDiscovery, CorpusIntelligence
from ike.retrieval.search_plan import (
    relaxed_websearch_expression,
    token_overlap,
)
from ike.retrieval.types import Candidate
from ike.services.inference_client import InferenceClient


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SearchEngine:
    """Low-level access-safe retrieval primitives.

    This class deliberately contains no query planning, source selection, evidence
    sufficiency, coverage heuristics, recovery or answer policy. Those semantic decisions
    belong to the research agent.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.inference = InferenceClient.for_query()
        self.corpus_intelligence = CorpusIntelligence(db)
        self._hnsw_configured = False

    def discover_corpus(
        self,
        question: str,
        user: User,
        document_ids: list[UUID] | None = None,
    ) -> CorpusDiscovery:
        return self.corpus_intelligence.discover(
            question,
            user,
            document_ids,
        )

    @staticmethod
    def _base_filters(
        user: User,
        document_ids: list[UUID] | None,
    ) -> list:
        filters = [document_access_clause(user)]
        if document_ids:
            filters.append(Document.id.in_(document_ids))
        return filters

    @staticmethod
    def candidate(chunk: Chunk, document: Document) -> Candidate:
        return Candidate(
            chunk_id=chunk.id,
            document_id=document.id,
            ordinal=chunk.ordinal,
            page_from=chunk.page_from,
            page_to=chunk.page_to,
            section_path=chunk.section_path or [],
            content_kind=chunk.content_kind,
            text=chunk.text,
            contextual_text=chunk.contextual_text,
            document_title=document.title,
            filename=document.original_filename,
            revision=document.revision,
            authority=document.authority,
            family_key=document.family_key,
            source_metadata=chunk.source_metadata or {},
            document_profile=((document.extra_metadata or {}).get("operational_profile", {})),
        )

    def dense(
        self,
        vector: list[float],
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        if not self._hnsw_configured:
            self.db.execute(
                text("SELECT set_config('hnsw.iterative_scan', :value, true)"),
                {"value": self.settings.hnsw_iterative_scan_mode},
            )
            self.db.execute(
                text("SELECT set_config('hnsw.ef_search', :value, true)"),
                {"value": str(self.settings.hnsw_ef_search)},
            )
            self.db.execute(
                text("SELECT set_config('hnsw.max_scan_tuples', :value, true)"),
                {"value": str(self.settings.hnsw_max_scan_tuples)},
            )
            self._hnsw_configured = True

        distance = Chunk.embedding.cosine_distance(vector).label("distance")
        stmt = (
            select(Chunk, Document, distance)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._base_filters(user, document_ids))
            .order_by(distance)
            .limit(top_k)
        )
        return [self.candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def lexical(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        tsquery = func.websearch_to_tsquery("simple", query)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                Chunk.search_vector.op("@@")(tsquery),
            )
            .order_by(desc(rank))
            .limit(top_k)
        )
        return [self.candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def relaxed_lexical(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        if not self.settings.relaxed_lexical_enabled:
            return []
        expression = relaxed_websearch_expression(query)
        if not expression:
            return []

        tsquery = func.websearch_to_tsquery("simple", expression)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                Chunk.search_vector.op("@@")(tsquery),
            )
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(top_k)
        )
        candidates = [self.candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
        for candidate in candidates:
            candidate.sources.add("relaxed")
        return candidates

    def section_navigation(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        if not self.settings.section_navigation_enabled:
            return []
        expression = relaxed_websearch_expression(
            query,
            max_terms=12,
        )
        if not expression:
            return []

        tsquery = func.websearch_to_tsquery("simple", expression)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        scan_limit = max(
            top_k,
            top_k * max(1, self.settings.section_navigation_scan_multiplier),
        )
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                Chunk.search_vector.op("@@")(tsquery),
            )
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(scan_limit)
        )

        scored: list[tuple[float, Candidate]] = []
        for chunk, document, rank_value in self.db.execute(stmt).all():
            candidate = self.candidate(chunk, document)
            heading_parts = [
                item.strip() for item in (candidate.section_path or []) if item.strip()
            ]
            heading = " / ".join(heading_parts[-2:])
            heading_overlap = token_overlap(query, heading) if heading else 0.0
            local_text = candidate.text or candidate.contextual_text or ""
            local_overlap = token_overlap(
                query,
                f"{heading} {local_text[:1600]}",
            )
            score = float(rank_value or 0.0) + (1.25 * heading_overlap) + (0.35 * local_overlap)
            candidate.sources.add("section")
            scored.append((score, candidate))

        scored.sort(
            key=lambda item: (item[0], -item[1].ordinal),
            reverse=True,
        )
        selected: list[Candidate] = []
        seen_sections: set[tuple] = set()
        for _score, candidate in scored:
            path_key = tuple(
                item.strip().casefold() for item in (candidate.section_path or []) if item.strip()
            )
            key = (
                candidate.document_id,
                path_key or (f"chunk:{candidate.chunk_id}",),
            )
            if key in seen_sections:
                continue
            seen_sections.add(key)
            selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def table_lexical(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        tsquery = func.websearch_to_tsquery("simple", query)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                Chunk.content_kind == "table",
                Chunk.search_vector.op("@@")(tsquery),
            )
            .order_by(desc(rank))
            .limit(top_k)
        )
        candidates = [self.candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
        for candidate in candidates:
            candidate.sources.add("table")
        return candidates

    def exact(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        cleaned = re.sub(r"\s+", " ", query.strip())
        if len(cleaned) < 2 or len(cleaned) > 220:
            return []

        pattern = f"%{_escape_like(cleaned)}%"
        stmt = (
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                Chunk.contextual_text.ilike(
                    pattern,
                    escape="\\",
                ),
            )
            .order_by(Chunk.document_id, Chunk.ordinal)
            .limit(top_k)
        )
        return [self.candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
