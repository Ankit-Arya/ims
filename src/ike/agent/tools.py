from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import Chunk, Document, RetrievalNode, User
from ike.retrieval.access import document_access_clause
from ike.retrieval.evidence_selection import deduplicate_candidates
from ike.retrieval.fusion import reciprocal_rank_fusion
from ike.retrieval.search_engine import SearchEngine
from ike.retrieval.search_plan import content_tokens, token_overlap
from ike.retrieval.types import Candidate


@dataclass(slots=True)
class ToolResult:
    tool: str
    arguments: dict
    items: list[dict] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    note: str = ""

    def trace(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "items": self.items,
            "metadata": self.metadata,
            "elapsed_ms": self.elapsed_ms,
            "note": self.note,
        }


class CorpusTools:
    """Fast access-safe retrieval primitives.

    Semantic query planning, source choice, completeness and evidence relevance belong to
    the AI agents. These methods only execute requested retrieval operations.
    """

    def __init__(
        self,
        db: Session,
        user: User,
        *,
        request_id: str | None = None,
        allowed_document_ids: list[UUID] | None = None,
    ) -> None:
        self.db = db
        self.user = user
        self.request_id = request_id
        self.settings = get_settings()
        self.search_engine = SearchEngine(db)
        self.allowed_document_ids = list(allowed_document_ids or [])

    def _accessible_documents(self) -> list[Document]:
        stmt = select(Document).where(document_access_clause(self.user))
        if self.allowed_document_ids:
            stmt = stmt.where(Document.id.in_(self.allowed_document_ids))
        return list(self.db.scalars(stmt).all())

    def _scope(self, raw_ids: list[str] | None) -> tuple[list[UUID] | None, str]:
        requested = bool(raw_ids)
        values: list[UUID] = []
        for raw in raw_ids or []:
            try:
                values.append(UUID(str(raw)))
            except (TypeError, ValueError):
                return [], "invalid_document_id"
        values = list(dict.fromkeys(values))

        if self.allowed_document_ids:
            allowed = set(self.allowed_document_ids)
            if not requested:
                return list(self.allowed_document_ids), "ok"
            if any(value not in allowed for value in values):
                return [], "outside_requested_scope"
            return values, "ok"

        if not requested:
            return None, "ok"

        accessible = set(
            self.db.scalars(
                select(Document.id).where(
                    Document.id.in_(values),
                    document_access_clause(self.user),
                )
            ).all()
        )
        if accessible != set(values):
            return [], "inaccessible_document"
        return values, "ok"

    @staticmethod
    def _item(candidate: Candidate) -> dict:
        return {
            "chunk_id": str(candidate.chunk_id),
            "document_id": str(candidate.document_id),
            "document_title": candidate.document_title,
            "filename": candidate.filename,
            "page_from": candidate.page_from,
            "page_to": candidate.page_to,
            "section_path": list(candidate.section_path or []),
            "content_kind": candidate.content_kind,
            "score": round(
                float(
                    candidate.final_retrieval_score
                    or candidate.rerank_score
                    or candidate.fused_score
                    or 0.0
                ),
                6,
            ),
            "sources": sorted(candidate.sources),
            "snippet": " ".join(
                (candidate.text or candidate.contextual_text or "").split()
            )[:1800],
        }

    @staticmethod
    def _merge(pool: dict[UUID, Candidate], candidate: Candidate) -> None:
        existing = pool.get(candidate.chunk_id)
        if existing is None:
            pool[candidate.chunk_id] = candidate
            return
        existing.sources.update(candidate.sources)
        existing.fused_score = max(existing.fused_score, candidate.fused_score)
        existing.final_retrieval_score = max(
            existing.final_retrieval_score,
            candidate.final_retrieval_score,
        )

    @staticmethod
    def _title_score(query: str, document: Document) -> float:
        title = f"{document.title or ''} {document.original_filename or ''}".strip()
        if not title:
            return 0.0
        q = query.casefold().strip()
        t = title.casefold()
        score = token_overlap(query, title) * 10.0
        compact_q = re.sub(r"[^a-z0-9]", "", q)
        compact_t = re.sub(r"[^a-z0-9]", "", t)
        if q and q in t:
            score += 8.0
        if compact_q and compact_q in compact_t:
            score += 6.0
        tokens = [value for value in re.findall(r"[a-z0-9]+", q) if len(value) >= 2]
        if tokens and all(token in t for token in tokens):
            score += 4.0

        # Strongly preserve letter-number technical identifiers during source routing
        # (for example RS2, RS-10, L7, TS318) without hard-coding any product family.
        identifiers = re.findall(
            r"\b[a-z]{1,10}[\s-]?\d{1,5}[a-z0-9/-]*\b",
            q,
        )
        for identifier in identifiers:
            compact_identifier = re.sub(r"[^a-z0-9]", "", identifier)
            if compact_identifier and compact_identifier in compact_t:
                score += 10.0
        return score

    def search_documents(self, query: str) -> ToolResult:
        started = time.perf_counter()
        query = " ".join(str(query or "").split())
        documents = {document.id: document for document in self._accessible_documents()}
        scores: dict[UUID, float] = {}
        matched_sections: dict[UUID, list[str]] = {}

        for document in documents.values():
            title_score = self._title_score(query, document)
            if title_score > 0:
                scores[document.id] = title_score + min(
                    3.5,
                    0.6 * math.log1p(max(0, int(document.page_count or 0))),
                )

        discovery_available = False
        try:
            discovery = self.search_engine.discover_corpus(
                query,
                self.user,
                self.allowed_document_ids or None,
            )
            discovery_available = bool(discovery.available)
            for rank, hit in enumerate(discovery.hits[:40], start=1):
                if hit.document_id not in documents:
                    continue
                discovery_score = (
                    4.0 + float(hit.score or 0.0) + (1.0 / rank)
                )
                scores[hit.document_id] = (
                    scores.get(hit.document_id, 0.0)
                    + discovery_score
                )
                section = " > ".join(hit.section_path) or hit.label
                if section:
                    matched_sections.setdefault(hit.document_id, [])
                    if section not in matched_sections[hit.document_id]:
                        matched_sections[hit.document_id].append(section)
        except Exception:
            discovery_available = False

        ranked = sorted(
            (
                (score, documents[document_id])
                for document_id, score in scores.items()
                if document_id in documents
            ),
            key=lambda item: (item[0], item[1].updated_at),
            reverse=True,
        )

        limit = self.settings.agent_document_search_limit
        items = [
            {
                "document_id": str(document.id),
                "title": document.title,
                "filename": document.original_filename,
                "page_count": document.page_count,
                "revision": document.revision,
                "authority": document.authority,
                "source_role": document.source_role,
                "score": round(score, 6),
                "matched_sections": matched_sections.get(document.id, [])[:6],
            }
            for score, document in ranked[:limit]
        ]
        return ToolResult(
            tool="source_lookup",
            arguments={"query": query},
            items=items,
            metadata={
                "candidate_count": len(items),
                "truncated": len(ranked) > limit,
                "corpus_discovery_used": discovery_available,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note=(
                "Document routing candidates from title plus corpus-level section hints. "
                "The AI chooses applicability."
            ),
        )

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        exact_terms: list[str] | None = None,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        query = " ".join(str(query or "").split())
        scope, scope_status = self._scope(document_ids)
        if scope_status != "ok":
            return ToolResult(
                tool="search",
                arguments={
                    "query": query,
                    "mode": mode,
                    "document_ids": list(document_ids or []),
                },
                metadata={"scope_status": scope_status},
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="Requested scope was rejected.",
            )

        mode = mode if mode in {"lexical", "semantic", "hybrid"} else "hybrid"
        top_k = min(
            max(4, int(top_k or self.settings.agent_search_top_k)),
            self.settings.agent_search_max_top_k,
        )
        per_lane_k = max(top_k, self.settings.agent_search_prefilter_k)
        lanes: list[tuple[str, list[Candidate], float]] = []
        lane_status: dict[str, str] = {}

        def run_lane(name: str, weight: float, fn) -> None:
            try:
                values = fn()
                lane_status[name] = "ok"
            except Exception:
                values = []
                lane_status[name] = "failed"
            for candidate in values:
                candidate.sources.add(f"agent:{name}")
            lanes.append((name, values, weight))

        if query and mode in {"lexical", "hybrid"}:
            run_lane(
                "lexical",
                1.0,
                lambda: self.search_engine.lexical(
                    query,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )
            run_lane(
                "exact_query",
                1.15,
                lambda: self.search_engine.exact(
                    query,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )

        cleaned_terms: list[str] = []
        seen_terms: set[str] = set()
        for raw in exact_terms or []:
            term = " ".join(str(raw or "").split())
            key = term.casefold()
            if not term or key in seen_terms:
                continue
            seen_terms.add(key)
            cleaned_terms.append(term)
            if len(cleaned_terms) >= 8:
                break
        for index, term in enumerate(cleaned_terms):
            run_lane(
                f"exact_term_{index + 1}",
                1.25,
                lambda term=term: self.search_engine.exact(
                    term,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )

        if query and mode in {"semantic", "hybrid"}:
            def dense_values() -> list[Candidate]:
                vector = self.search_engine.inference.embed_query(query)
                return self.search_engine.dense(
                    vector,
                    self.user,
                    scope,
                    per_lane_k,
                )

            run_lane("dense", 1.0, dense_values)

        pool: dict[UUID, Candidate] = {}
        ranked_lists: list[tuple[str, list[UUID], float]] = []
        for name, values, weight in lanes:
            if not values:
                continue
            ranked_lists.append(
                (name, [candidate.chunk_id for candidate in values], weight)
            )
            for candidate in values:
                self._merge(pool, candidate)

        if not ranked_lists:
            return ToolResult(
                tool="search",
                arguments={
                    "query": query,
                    "mode": mode,
                    "exact_terms": cleaned_terms,
                    "document_ids": [str(value) for value in (scope or [])],
                    "top_k": top_k,
                },
                metadata={
                    "scope_status": "ok",
                    "lane_status": lane_status,
                    "candidate_count": 0,
                },
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="No evidence candidates found.",
            )

        fused = reciprocal_rank_fusion(ranked_lists)
        ranked: list[Candidate] = []
        for chunk_id, (score, sources) in fused.items():
            candidate = pool[chunk_id]
            candidate.fused_score = max(candidate.fused_score, score)
            candidate.final_retrieval_score = max(
                candidate.final_retrieval_score,
                score,
            )
            candidate.rank_method = "agent_fast_rrf"
            candidate.sources.update(f"agent:{source}" for source in sources)
            ranked.append(candidate)

        ranked.sort(
            key=lambda candidate: (
                candidate.final_retrieval_score,
                -candidate.ordinal,
            ),
            reverse=True,
        )
        ranked = deduplicate_candidates(ranked)[:top_k]
        return ToolResult(
            tool="search",
            arguments={
                "query": query,
                "mode": mode,
                "exact_terms": cleaned_terms,
                "document_ids": [str(value) for value in (scope or [])],
                "top_k": top_k,
            },
            items=[self._item(candidate) for candidate in ranked],
            candidates=ranked,
            metadata={
                "scope_status": "ok",
                "lane_status": lane_status,
                "candidate_count": len(ranked),
                "document_ids_returned": sorted(
                    {str(candidate.document_id) for candidate in ranked}
                ),
                "reranked": False,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Fast retrieval bundle; semantic relevance is decided by the AI.",
        )

    def enumerate(
        self,
        query: str,
        *,
        mode: str = "lexical",
        exact_terms: list[str] | None = None,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        """Broader candidate retrieval for AI-requested exhaustive/list questions."""

        started = time.perf_counter()
        query = " ".join(str(query or "").split())
        scope, scope_status = self._scope(document_ids)
        if scope_status != "ok":
            return ToolResult(
                tool="enumerate",
                arguments={
                    "query": query,
                    "mode": mode,
                    "document_ids": list(document_ids or []),
                },
                metadata={"scope_status": scope_status},
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="Requested scope was rejected.",
            )

        mode = mode if mode in {"lexical", "semantic", "hybrid"} else "lexical"
        top_k = min(
            max(
                self.settings.agent_enumeration_top_k,
                int(top_k or self.settings.agent_enumeration_top_k),
            ),
            self.settings.agent_enumeration_max_top_k,
        )
        per_lane_k = max(top_k, self.settings.agent_enumeration_prefilter_k)
        lanes: list[tuple[str, list[Candidate], float]] = []
        lane_status: dict[str, str] = {}

        def run_lane(name: str, weight: float, fn) -> None:
            try:
                values = fn()
                lane_status[name] = "ok"
            except Exception:
                values = []
                lane_status[name] = "failed"
            for candidate in values:
                candidate.sources.add(f"agent:enumerate:{name}")
            lanes.append((name, values, weight))

        if query and mode in {"lexical", "hybrid"}:
            run_lane(
                "lexical",
                1.0,
                lambda: self.search_engine.lexical(
                    query,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )
            run_lane(
                "relaxed",
                0.9,
                lambda: self.search_engine.relaxed_lexical(
                    query,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )

        cleaned_terms: list[str] = []
        seen_terms: set[str] = set()
        for raw in exact_terms or []:
            term = " ".join(str(raw or "").split())
            key = term.casefold()
            if not term or key in seen_terms:
                continue
            seen_terms.add(key)
            cleaned_terms.append(term)
            if len(cleaned_terms) >= 8:
                break

        for index, term in enumerate(cleaned_terms):
            run_lane(
                f"exact_term_{index + 1}",
                1.2,
                lambda term=term: self.search_engine.exact(
                    term,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )

        if query and mode in {"semantic", "hybrid"}:
            def dense_values() -> list[Candidate]:
                vector = self.search_engine.inference.embed_query(query)
                return self.search_engine.dense(
                    vector,
                    self.user,
                    scope,
                    min(per_lane_k, self.settings.agent_enumeration_dense_top_k),
                )

            run_lane("dense", 0.85, dense_values)

        pool: dict[UUID, Candidate] = {}
        ranked_lists: list[tuple[str, list[UUID], float]] = []
        for name, values, weight in lanes:
            if not values:
                continue
            ranked_lists.append(
                (name, [candidate.chunk_id for candidate in values], weight)
            )
            for candidate in values:
                self._merge(pool, candidate)

        if not ranked_lists:
            return ToolResult(
                tool="enumerate",
                arguments={
                    "query": query,
                    "mode": mode,
                    "exact_terms": cleaned_terms,
                    "document_ids": [str(value) for value in (scope or [])],
                    "top_k": top_k,
                },
                metadata={
                    "scope_status": "ok",
                    "lane_status": lane_status,
                    "candidate_count": 0,
                },
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="No enumeration candidates found.",
            )

        fused = reciprocal_rank_fusion(ranked_lists)
        ranked: list[Candidate] = []
        for chunk_id, (score, sources) in fused.items():
            candidate = pool[chunk_id]
            candidate.fused_score = max(candidate.fused_score, score)
            candidate.final_retrieval_score = max(
                candidate.final_retrieval_score,
                score,
            )
            candidate.rank_method = "agent_enumeration_rrf"
            candidate.sources.update(
                f"agent:enumerate:{source}" for source in sources
            )
            ranked.append(candidate)

        ranked.sort(
            key=lambda candidate: (
                candidate.final_retrieval_score,
                -candidate.ordinal,
            ),
            reverse=True,
        )
        ranked = deduplicate_candidates(ranked)[:top_k]
        return ToolResult(
            tool="enumerate",
            arguments={
                "query": query,
                "mode": mode,
                "exact_terms": cleaned_terms,
                "document_ids": [str(value) for value in (scope or [])],
                "top_k": top_k,
            },
            items=[self._item(candidate) for candidate in ranked],
            candidates=ranked,
            metadata={
                "scope_status": "ok",
                "lane_status": lane_status,
                "candidate_count": len(ranked),
                "document_ids_returned": sorted(
                    {str(candidate.document_id) for candidate in ranked}
                ),
                "exhaustive_candidate_mode": True,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note=(
                "Broad candidate retrieval requested by the AI for a bounded/exhaustive "
                "question. The AI still decides which items belong in the final set."
            ),
        )

    def inspect_structure(
        self,
        document_id: str,
        *,
        query: str = "",
    ) -> ToolResult:
        started = time.perf_counter()
        scope, status = self._scope([document_id])
        if status != "ok" or not scope:
            return ToolResult(
                tool="structure",
                arguments={"document_id": document_id, "query": query},
                metadata={"scope_status": status},
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="Document scope was rejected.",
            )
        doc_id = scope[0]
        document = self.db.scalar(
            select(Document).where(
                Document.id == doc_id,
                document_access_clause(self.user),
            )
        )
        if document is None:
            return ToolResult(
                tool="structure",
                arguments={"document_id": document_id, "query": query},
                note="Document is not accessible.",
            )

        nodes = list(
            self.db.scalars(
                select(RetrievalNode)
                .where(
                    RetrievalNode.document_id == doc_id,
                    RetrievalNode.node_type == "section",
                )
                .order_by(
                    RetrievalNode.ordinal_from,
                    RetrievalNode.page_from,
                )
                .limit(5000)
            ).all()
        )
        structure_query = " ".join(str(query or "").split())
        query_tokens = {
            token.casefold()
            for token in content_tokens(structure_query, max_terms=12)
        }

        unique: list[RetrievalNode] = []
        seen_paths: set[tuple[str, ...]] = set()
        for node in nodes:
            path = tuple(
                str(value).strip()
                for value in (node.section_path or [])
                if str(value).strip()
            )
            if not path:
                continue
            haystack = " ".join((node.label or "", *path)).casefold()
            if structure_query:
                exact = structure_query.casefold() in haystack
                hay_tokens = {
                    token.casefold()
                    for token in content_tokens(haystack, max_terms=80)
                }
                if not exact and not (query_tokens & hay_tokens):
                    continue
            key = tuple(value.casefold() for value in path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            unique.append(node)

        limit = self.settings.agent_structure_max_nodes
        returned = unique[:limit]
        candidates: list[Candidate] = []
        items: list[dict] = []
        seen_chunks: set[UUID] = set()

        for node in returned:
            path = [
                str(value).strip()
                for value in (node.section_path or [])
                if str(value).strip()
            ]
            start = node.ordinal_from if node.ordinal_from is not None else 0
            end = node.ordinal_to if node.ordinal_to is not None else start
            chunk = self.db.scalar(
                select(Chunk)
                .where(
                    Chunk.document_id == doc_id,
                    Chunk.ordinal >= start,
                    Chunk.ordinal <= end,
                )
                .order_by(Chunk.ordinal)
                .limit(1)
            )
            representative_chunk_id = str(chunk.id) if chunk is not None else None
            if chunk is not None and chunk.id not in seen_chunks:
                seen_chunks.add(chunk.id)
                candidate = self.search_engine.candidate(chunk, document)
                candidate.evidence_lane = "document_structure"
                candidate.rank_method = "agent_structure"
                candidate.final_retrieval_score = 0.9
                candidate.sources.add("agent:structure")
                candidates.append(candidate)
            items.append(
                {
                    "node_id": str(node.id),
                    "label": node.label,
                    "section_path": path,
                    "depth": len(path),
                    "page_from": node.page_from,
                    "page_to": node.page_to,
                    "ordinal_from": node.ordinal_from,
                    "ordinal_to": node.ordinal_to,
                    "representative_chunk_id": representative_chunk_id,
                }
            )

        return ToolResult(
            tool="structure",
            arguments={"document_id": str(doc_id), "query": structure_query},
            items=items,
            candidates=candidates,
            metadata={
                "document_id": str(doc_id),
                "document_title": document.title,
                "matched_node_count": len(unique),
                "returned_node_count": len(returned),
                "truncated": len(unique) > limit,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Raw indexed hierarchy; the AI interprets the structure.",
        )

    def inspect_context(
        self,
        chunk_id: str,
        *,
        before: int = 3,
        after: int = 5,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            target_id = UUID(str(chunk_id))
        except (TypeError, ValueError):
            return ToolResult(
                tool="context",
                arguments={"chunk_id": chunk_id},
                note="Invalid chunk id.",
            )

        row = self.db.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.id == target_id,
                document_access_clause(self.user),
            )
        ).first()
        if row is None:
            return ToolResult(
                tool="context",
                arguments={"chunk_id": chunk_id},
                note="Chunk is not accessible.",
            )
        target, document = row
        if self.allowed_document_ids and target.document_id not in set(
            self.allowed_document_ids
        ):
            return ToolResult(
                tool="context",
                arguments={"chunk_id": chunk_id},
                note="Chunk is outside the selected document scope.",
            )

        before = min(max(0, int(before)), 6)
        after = min(max(0, int(after)), 8)
        rows = list(
            self.db.scalars(
                select(Chunk)
                .where(
                    Chunk.document_id == target.document_id,
                    Chunk.ordinal >= max(0, target.ordinal - before),
                    Chunk.ordinal <= target.ordinal + after,
                )
                .order_by(Chunk.ordinal)
            ).all()
        )
        candidates: list[Candidate] = []
        for chunk in rows:
            candidate = self.search_engine.candidate(chunk, document)
            candidate.evidence_lane = "context"
            candidate.rank_method = "agent_context"
            candidate.final_retrieval_score = 1.0 if chunk.id == target.id else 0.8
            candidate.sources.add("agent:context")
            candidates.append(candidate)

        return ToolResult(
            tool="context",
            arguments={
                "chunk_id": str(target_id),
                "before": before,
                "after": after,
            },
            items=[self._item(candidate) for candidate in candidates],
            candidates=candidates,
            metadata={
                "document_id": str(target.document_id),
                "document_title": document.title,
                "target_ordinal": target.ordinal,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Original neighboring chunks in source order.",
        )
