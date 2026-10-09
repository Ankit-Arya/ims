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
from ike.retrieval.search_plan import (
    content_tokens,
    table_retrieval_relevant,
    token_overlap,
)
from ike.retrieval.table_context import retrieval_text, table_query_affinity
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
            "direct_score": round(
                float(candidate.judge_details.get("direct_evidence_score", 0.0) or 0.0),
                6,
            ),
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
    def _rerank_text(candidate: Candidate) -> str:
        section = " > ".join(
            str(value).strip()
            for value in (candidate.section_path or [])[-2:]
            if str(value).strip()
        )
        prefix = [
            f"Document: {candidate.document_title}",
            f"File: {candidate.filename}",
        ]
        if section:
            prefix.append(f"Section: {section}")
        return "\n".join(prefix + [retrieval_text(candidate)])

    def _rerank_candidates(
        self,
        query: str,
        ranked: list[Candidate],
        *,
        top_k: int,
        pool_limit: int,
    ) -> tuple[list[Candidate], dict]:
        if not ranked:
            return [], {"status": "empty"}
        pool = deduplicate_candidates(ranked)[: max(top_k, pool_limit)]
        if not self.settings.agent_rerank_enabled or len(pool) <= 1:
            return pool[:top_k], {
                "status": "disabled",
                "candidate_count": len(pool),
            }
        try:
            results, timing = self.search_engine.inference.rerank(
                query,
                [self._rerank_text(candidate) for candidate in pool],
                top_k=min(top_k, len(pool)),
                request_id=self.request_id,
            )
        except Exception as exc:
            return pool[:top_k], {
                "status": "failed",
                "candidate_count": len(pool),
                "error_type": type(exc).__name__,
            }

        selected: list[Candidate] = []
        for result in results:
            if result.index < 0 or result.index >= len(pool):
                continue
            candidate = pool[result.index]
            candidate.rerank_score = float(result.score)
            candidate.final_retrieval_score = (
                float(result.score)
                + table_query_affinity(query, candidate)
            )
            candidate.rank_method = "agent_local_cross_encoder"
            candidate.sources.add("agent:local_rerank")
            selected.append(candidate)
        return selected, {
            "status": "ok",
            "candidate_count": len(pool),
            "returned_count": len(selected),
            "queue_wait_ms": timing.queue_wait_ms,
            "execution_ms": timing.execution_ms,
            "details": timing.details,
        }

    @classmethod
    def _direct_evidence_score(
        cls,
        query: str,
        exact_terms: list[str],
        candidate: Candidate,
    ) -> float:
        """Cheap query-to-chunk affinity before any optional cross-encoder rerank."""

        text = cls._rerank_text(candidate)
        if not text:
            return 0.0
        score = 0.10 * token_overlap(query, text)
        score += 0.025 * token_overlap(query, candidate.document_title or "")

        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        matched_terms = 0
        literal_hits = 0
        for term in exact_terms:
            normalized_term = re.sub(
                r"[^a-z0-9]+",
                " ",
                str(term or "").casefold(),
            ).strip()
            if not normalized_term:
                continue
            term_tokens = normalized_term.split()
            if all(token in normalized_text.split() for token in term_tokens):
                matched_terms += 1
            if len(normalized_term) >= 4 and normalized_term in normalized_text:
                literal_hits += 1
        if exact_terms:
            score += 0.08 * (matched_terms / max(1, len(exact_terms)))
        score += min(0.06, 0.02 * literal_hits)
        score += 0.30 * table_query_affinity(query, candidate)
        return score

    @staticmethod
    def _table_section_prior_ids(
        lanes: list[tuple[str, list[Candidate], float]],
    ) -> list[UUID]:
        """Fuse exact table-term hits that occur across chunks of one table section."""

        section_terms: dict[tuple[UUID, tuple[str, ...]], set[str]] = {}
        chunk_sections: dict[UUID, tuple[UUID, tuple[str, ...]]] = {}
        best_rank: dict[UUID, int] = {}
        chunk_lane_hits: dict[UUID, int] = {}

        for lane_name, values, _weight in lanes:
            if not lane_name.startswith("table_exact_term_"):
                continue
            for rank, candidate in enumerate(values):
                path = tuple(
                    str(value).strip().casefold()
                    for value in (candidate.section_path or [])
                    if str(value).strip()
                )
                if not path:
                    continue
                section_key = (candidate.document_id, path)
                section_terms.setdefault(section_key, set()).add(lane_name)
                chunk_sections[candidate.chunk_id] = section_key
                best_rank[candidate.chunk_id] = min(
                    rank,
                    best_rank.get(candidate.chunk_id, rank),
                )
                chunk_lane_hits[candidate.chunk_id] = (
                    chunk_lane_hits.get(candidate.chunk_id, 0) + 1
                )

        qualified = {
            key: len(term_lanes)
            for key, term_lanes in section_terms.items()
            if len(term_lanes) >= 2
        }
        if not qualified:
            return []

        chunk_ids = [
            chunk_id
            for chunk_id, section_key in chunk_sections.items()
            if section_key in qualified
        ]
        chunk_ids.sort(
            key=lambda chunk_id: (
                -qualified[chunk_sections[chunk_id]],
                -chunk_lane_hits.get(chunk_id, 0),
                best_rank.get(chunk_id, 10**9),
            )
        )
        return chunk_ids

    @staticmethod
    def _source_prior_ids(
        lanes: list[tuple[str, list[Candidate], float]],
        boost_document_ids: list[UUID] | None,
    ) -> list[UUID]:
        if not boost_document_ids:
            return []
        source_rank = {
            document_id: index
            for index, document_id in enumerate(boost_document_ids)
        }
        best_lane_rank: dict[UUID, int] = {}
        candidates: dict[UUID, Candidate] = {}
        for _name, values, _weight in lanes:
            for rank, candidate in enumerate(values):
                candidates[candidate.chunk_id] = candidate
                best_lane_rank[candidate.chunk_id] = min(
                    rank,
                    best_lane_rank.get(candidate.chunk_id, rank),
                )
        boosted = [
            candidate
            for candidate in candidates.values()
            if candidate.document_id in source_rank
        ]
        boosted.sort(
            key=lambda candidate: (
                source_rank[candidate.document_id],
                best_lane_rank.get(candidate.chunk_id, 10**9),
                candidate.ordinal,
            )
        )
        return [candidate.chunk_id for candidate in boosted]

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

        # Preserve canonical short document identities/acronyms when the query literally
        # names them. This is generic source fidelity, not a domain-specific alias table.
        aliases = {
            str(document.title or "").strip(),
            re.sub(
                r"\.[a-z0-9]{1,6}$",
                "",
                str(document.original_filename or "").strip(),
                flags=re.IGNORECASE,
            ),
        }
        for alias in aliases:
            alias_folded = alias.casefold().strip()
            if not alias_folded or len(alias_folded) > 48:
                continue
            alias_tokens = re.findall(r"[a-z0-9]+", alias_folded)
            if not alias_tokens or len(alias_tokens) > 5:
                continue
            pattern = r"(?<![a-z0-9])" + r"[\s._/-]*".join(
                re.escape(token) for token in alias_tokens
            ) + r"(?![a-z0-9])"
            if re.search(pattern, q):
                score += 24.0

        # Also preserve acronym-like identity tokens embedded in longer filenames/titles,
        # e.g. a numbered file whose canonical identity is one uppercase token.
        raw_identity = f"{document.title or ''} {document.original_filename or ''}"
        for acronym in set(re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,12}(?![A-Za-z0-9])", raw_identity)):
            acronym_folded = acronym.casefold()
            if re.search(
                r"(?<![a-z0-9])" + re.escape(acronym_folded) + r"(?![a-z0-9])",
                q,
            ):
                score += 36.0

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

    @staticmethod
    def _metadata_routing_hints(document: Document) -> dict:
        metadata = dict(document.extra_metadata or {})
        okf = dict(metadata.get("okf") or {})
        profile = dict(metadata.get("operational_profile") or {})

        def values(name: str, *, limit: int = 12) -> list[str]:
            raw = profile.get(name) or []
            if isinstance(raw, str):
                raw = [raw]
            return [
                str(value).strip()
                for value in raw
                if str(value).strip()
            ][:limit]

        return {
            "topic_terms": [
                str(value).strip()
                for value in (okf.get("topic_terms") or [])
                if str(value).strip()
            ][:20],
            "rolling_stock": values("rolling_stock"),
            "primary_rolling_stock": values("primary_rolling_stock"),
            "line_codes": values("line_codes"),
            "primary_line_codes": values("primary_line_codes"),
            "document_type": str(profile.get("document_type") or "").strip(),
        }

    @classmethod
    def _metadata_score(cls, query: str, document: Document) -> tuple[float, dict]:
        hints = cls._metadata_routing_hints(document)
        metadata_values: list[str] = []
        for key in (
            "topic_terms",
            "rolling_stock",
            "primary_rolling_stock",
            "line_codes",
            "primary_line_codes",
        ):
            metadata_values.extend(hints.get(key) or [])
        if hints.get("document_type"):
            metadata_values.append(str(hints["document_type"]))
        if document.family_key:
            metadata_values.append(str(document.family_key))
        if document.source_role:
            metadata_values.append(str(document.source_role))

        metadata_text = " ".join(metadata_values).strip()
        if not metadata_text:
            return 0.0, hints

        query_folded = query.casefold()
        metadata_folded = metadata_text.casefold()
        score = token_overlap(query, metadata_text) * 8.0

        query_tokens = {
            value
            for value in re.findall(r"[a-z0-9]+", query_folded)
            if len(value) >= 3
        }
        topic_tokens = {
            str(value).casefold()
            for value in (hints.get("topic_terms") or [])
            if len(str(value).strip()) >= 3
        }
        score += min(15.0, 3.0 * len(query_tokens & topic_tokens))

        compact_metadata = re.sub(r"[^a-z0-9]", "", metadata_folded)
        identifiers = re.findall(
            r"\b[a-z]{1,10}[\s-]?\d{1,5}[a-z0-9/-]*\b",
            query_folded,
        )
        for identifier in identifiers:
            compact_identifier = re.sub(r"[^a-z0-9]", "", identifier)
            if compact_identifier and compact_identifier in compact_metadata:
                score += 10.0

        return score, hints

    def search_documents(self, query: str) -> ToolResult:
        started = time.perf_counter()
        query = " ".join(str(query or "").split())
        documents = {document.id: document for document in self._accessible_documents()}
        scores: dict[UUID, float] = {}
        matched_sections: dict[UUID, list[str]] = {}
        metadata_hints: dict[UUID, dict] = {}

        for document in documents.values():
            title_score = self._title_score(query, document)
            metadata_score, hints = self._metadata_score(query, document)
            metadata_hints[document.id] = hints
            base_score = title_score + metadata_score
            if base_score > 0:
                scores[document.id] = base_score + min(
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
                "metadata_hints": metadata_hints.get(document.id, {}),
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
                "Document routing candidates from title, metadata and corpus-level section "
                "hints. Inferred candidates are soft ranking hints; only explicit document "
                "scope can restrict chunk retrieval."
            ),
        )

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        exact_terms: list[str] | None = None,
        document_ids: list[str] | None = None,
        boost_document_ids: list[str] | None = None,
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
                    "boost_document_ids": list(boost_document_ids or []),
                },
                metadata={"scope_status": scope_status},
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="Requested hard document scope was rejected.",
            )

        boost_scope: list[UUID] = []
        boost_status = "not_requested"
        if boost_document_ids:
            resolved_boost, boost_status = self._scope(boost_document_ids)
            if boost_status == "ok":
                boost_scope = list(resolved_boost or [])

        mode = mode if mode in {"lexical", "semantic", "hybrid"} else "hybrid"
        top_k = min(
            max(
                self.settings.agent_search_top_k,
                int(top_k or self.settings.agent_search_top_k),
            ),
            self.settings.agent_search_max_top_k,
        )
        per_lane_k = max(
            top_k,
            self.settings.agent_search_prefilter_k,
        )
        lanes: list[tuple[str, list[Candidate], float]] = []
        lane_status: dict[str, str] = {}
        table_relevant = table_retrieval_relevant(
            goal_kinds=[],
            facets=[],
            question=query,
        )

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
            if self.settings.relaxed_lexical_enabled:
                run_lane(
                    "relaxed",
                    0.8,
                    lambda: self.search_engine.relaxed_lexical(
                        query,
                        self.user,
                        scope,
                        per_lane_k,
                    ),
                )
            if boost_scope:
                run_lane(
                    "source_lexical",
                    0.9,
                    lambda: self.search_engine.lexical(
                        query,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )
                if self.settings.relaxed_lexical_enabled:
                    run_lane(
                        "source_relaxed",
                        0.8,
                        lambda: self.search_engine.relaxed_lexical(
                            query,
                            self.user,
                            boost_scope,
                            per_lane_k,
                        ),
                    )
            if table_relevant:
                run_lane(
                    "table_lexical",
                    1.05,
                    lambda: self.search_engine.table_lexical(
                        query,
                        self.user,
                        scope,
                        per_lane_k,
                    ),
                )
                run_lane(
                    "table_relaxed",
                    1.0,
                    lambda: self.search_engine.table_relaxed_lexical(
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
            if boost_scope:
                run_lane(
                    f"source_exact_term_{index + 1}",
                    1.0,
                    lambda term=term: self.search_engine.exact(
                        term,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )
            if table_relevant:
                run_lane(
                    f"table_exact_term_{index + 1}",
                    1.35,
                    lambda term=term: self.search_engine.table_exact(
                        term,
                        self.user,
                        scope,
                        per_lane_k,
                    ),
                )

        if query and mode in {"semantic", "hybrid"}:
            vector = self.search_engine.inference.embed_query(query)
            run_lane(
                "dense",
                1.0,
                lambda: self.search_engine.dense(
                    vector,
                    self.user,
                    scope,
                    per_lane_k,
                ),
            )
            if boost_scope:
                run_lane(
                    "source_dense",
                    0.9,
                    lambda: self.search_engine.dense(
                        vector,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )

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

        source_prior_ids = self._source_prior_ids(lanes, boost_scope)
        if source_prior_ids:
            ranked_lists.append(("source_prior", source_prior_ids, 0.45))
        table_section_prior_ids = self._table_section_prior_ids(lanes)
        if table_section_prior_ids:
            ranked_lists.append(
                ("table_section_prior", table_section_prior_ids, 1.10)
            )

        if not ranked_lists:
            return ToolResult(
                tool="search",
                arguments={
                    "query": query,
                    "mode": mode,
                    "exact_terms": cleaned_terms,
                    "document_ids": [str(value) for value in (scope or [])],
                    "boost_document_ids": [str(value) for value in boost_scope],
                    "top_k": top_k,
                },
                metadata={
                    "scope_status": "ok",
                    "boost_status": boost_status,
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
            direct_score = self._direct_evidence_score(
                query,
                cleaned_terms,
                candidate,
            )
            candidate.judge_details["direct_evidence_score"] = direct_score
            candidate.fused_score = max(candidate.fused_score, score)
            candidate.final_retrieval_score = score + direct_score
            candidate.rank_method = "agent_global_rrf_direct"
            candidate.sources.update(f"agent:{source}" for source in sources)
            if direct_score > 0:
                candidate.sources.add("agent:direct_match")
            ranked.append(candidate)

        ranked.sort(
            key=lambda candidate: (
                candidate.final_retrieval_score,
                -candidate.ordinal,
            ),
            reverse=True,
        )
        ranked, rerank_metadata = self._rerank_candidates(
            query,
            ranked,
            top_k=top_k,
            pool_limit=self.settings.agent_search_rerank_candidates,
        )
        return ToolResult(
            tool="search",
            arguments={
                "query": query,
                "mode": mode,
                "exact_terms": cleaned_terms,
                "document_ids": [str(value) for value in (scope or [])],
                "boost_document_ids": [str(value) for value in boost_scope],
                "top_k": top_k,
            },
            items=[self._item(candidate) for candidate in ranked],
            candidates=ranked,
            metadata={
                "scope_status": "ok",
                "boost_status": boost_status,
                "lane_status": lane_status,
                "candidate_count": len(ranked),
                "document_ids_returned": sorted(
                    {str(candidate.document_id) for candidate in ranked}
                ),
                "source_prior_candidate_count": len(source_prior_ids),
                "table_section_prior_candidate_count": len(table_section_prior_ids),
                "reranked": rerank_metadata.get("status") == "ok",
                "rerank": rerank_metadata,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note=(
                "Global chunk retrieval with lexical, exact, semantic and applicable "
                "structured-table lanes. AI-inferred source candidates are soft ranking "
                "hints only. "
                + (
                    "A local cross-encoder reranked the evidence pool."
                    if rerank_metadata.get("status") == "ok"
                    else "Deterministic fusion/direct-match scoring ranked the evidence pool."
                )
            ),
        )

    def enumerate(
        self,
        query: str,
        *,
        mode: str = "lexical",
        exact_terms: list[str] | None = None,
        document_ids: list[str] | None = None,
        boost_document_ids: list[str] | None = None,
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
                    "boost_document_ids": list(boost_document_ids or []),
                },
                metadata={"scope_status": scope_status},
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="Requested hard document scope was rejected.",
            )

        boost_scope: list[UUID] = []
        boost_status = "not_requested"
        if boost_document_ids:
            resolved_boost, boost_status = self._scope(boost_document_ids)
            if boost_status == "ok":
                boost_scope = list(resolved_boost or [])

        mode = mode if mode in {"lexical", "semantic", "hybrid"} else "lexical"
        top_k = min(
            max(
                16,
                int(top_k or self.settings.agent_enumeration_top_k),
            ),
            self.settings.agent_enumeration_max_top_k,
        )
        per_lane_k = max(top_k, self.settings.agent_enumeration_prefilter_k)
        lanes: list[tuple[str, list[Candidate], float]] = []
        lane_status: dict[str, str] = {}
        table_relevant = table_retrieval_relevant(
            goal_kinds=["enumeration"],
            facets=[],
            question=query,
        )

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
            if boost_scope:
                run_lane(
                    "source_lexical",
                    0.9,
                    lambda: self.search_engine.lexical(
                        query,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )
                run_lane(
                    "source_relaxed",
                    0.8,
                    lambda: self.search_engine.relaxed_lexical(
                        query,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )
            if table_relevant:
                run_lane(
                    "table_lexical",
                    1.05,
                    lambda: self.search_engine.table_lexical(
                        query,
                        self.user,
                        scope,
                        per_lane_k,
                    ),
                )
                run_lane(
                    "table_relaxed",
                    1.0,
                    lambda: self.search_engine.table_relaxed_lexical(
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
            if boost_scope:
                run_lane(
                    f"source_exact_term_{index + 1}",
                    1.0,
                    lambda term=term: self.search_engine.exact(
                        term,
                        self.user,
                        boost_scope,
                        per_lane_k,
                    ),
                )
            if table_relevant:
                run_lane(
                    f"table_exact_term_{index + 1}",
                    1.3,
                    lambda term=term: self.search_engine.table_exact(
                        term,
                        self.user,
                        scope,
                        per_lane_k,
                    ),
                )

        if query and mode in {"semantic", "hybrid"}:
            vector = self.search_engine.inference.embed_query(query)
            dense_k = min(
                per_lane_k,
                self.settings.agent_enumeration_dense_top_k,
            )
            run_lane(
                "dense",
                0.85,
                lambda: self.search_engine.dense(
                    vector,
                    self.user,
                    scope,
                    dense_k,
                ),
            )
            if boost_scope:
                run_lane(
                    "source_dense",
                    0.8,
                    lambda: self.search_engine.dense(
                        vector,
                        self.user,
                        boost_scope,
                        dense_k,
                    ),
                )

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

        source_prior_ids = self._source_prior_ids(lanes, boost_scope)
        if source_prior_ids:
            ranked_lists.append(("source_prior", source_prior_ids, 0.35))
        table_section_prior_ids = self._table_section_prior_ids(lanes)
        if table_section_prior_ids:
            ranked_lists.append(
                ("table_section_prior", table_section_prior_ids, 1.05)
            )

        if not ranked_lists:
            return ToolResult(
                tool="enumerate",
                arguments={
                    "query": query,
                    "mode": mode,
                    "exact_terms": cleaned_terms,
                    "document_ids": [str(value) for value in (scope or [])],
                    "boost_document_ids": [str(value) for value in boost_scope],
                    "top_k": top_k,
                },
                metadata={
                    "scope_status": "ok",
                    "boost_status": boost_status,
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
            direct_score = self._direct_evidence_score(
                query,
                cleaned_terms,
                candidate,
            )
            candidate.judge_details["direct_evidence_score"] = direct_score
            candidate.fused_score = max(candidate.fused_score, score)
            candidate.final_retrieval_score = score + direct_score
            candidate.rank_method = "agent_enumeration_rrf_direct"
            candidate.sources.update(
                f"agent:enumerate:{source}" for source in sources
            )
            if direct_score > 0:
                candidate.sources.add("agent:enumerate:direct_match")
            ranked.append(candidate)

        ranked.sort(
            key=lambda candidate: (
                candidate.final_retrieval_score,
                -candidate.ordinal,
            ),
            reverse=True,
        )
        ranked, rerank_metadata = self._rerank_candidates(
            query,
            ranked,
            top_k=top_k,
            pool_limit=self.settings.agent_enumeration_rerank_candidates,
        )
        return ToolResult(
            tool="enumerate",
            arguments={
                "query": query,
                "mode": mode,
                "exact_terms": cleaned_terms,
                "document_ids": [str(value) for value in (scope or [])],
                "boost_document_ids": [str(value) for value in boost_scope],
                "top_k": top_k,
            },
            items=[self._item(candidate) for candidate in ranked],
            candidates=ranked,
            metadata={
                "scope_status": "ok",
                "boost_status": boost_status,
                "lane_status": lane_status,
                "candidate_count": len(ranked),
                "document_ids_returned": sorted(
                    {str(candidate.document_id) for candidate in ranked}
                ),
                "source_prior_candidate_count": len(source_prior_ids),
                "table_section_prior_candidate_count": len(table_section_prior_ids),
                "exhaustive_candidate_mode": True,
                "reranked": rerank_metadata.get("status") == "ok",
                "rerank": rerank_metadata,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note=(
                "Broad global candidate retrieval for a bounded/exhaustive question, "
                "including structured-table lanes. AI-inferred source candidates are soft "
                "ranking hints only. "
                + (
                    "A local cross-encoder reranked the broad evidence set."
                    if rerank_metadata.get("status") == "ok"
                    else "Deterministic fusion/direct-match scoring ranked the broad evidence set."
                )
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
