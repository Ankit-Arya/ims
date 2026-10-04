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
from ike.retrieval.evidence_shape import enumeration_shape_score
from ike.retrieval.fusion import reciprocal_rank_fusion
from ike.retrieval.search_plan import token_overlap
from ike.retrieval.table_context import retrieval_text
from ike.retrieval.types import Candidate
from ike.retrieval.engine import RetrievalEngine
from ike.retrieval.query_plan import QueryPlan


_GENERIC_QUERY_WORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "what", "which", "who", "when", "where", "why", "how", "should", "would",
    "could", "can", "do", "does", "did", "case", "please", "show", "give", "tell",
    "list", "complete", "all", "name", "names", "location", "locations", "provide",
}


def _normalized_terms(value: str, *, long_only: bool = False) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", value or ""):
        token = raw.casefold()
        if token in _GENERIC_QUERY_WORDS:
            continue
        if long_only and len(token) < 4:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is")):
            token = token[:-1]
        if len(token) >= 2:
            terms.add(token)
    return terms


def _normalized_overlap(query: str, candidate: str, *, long_only: bool = False) -> float:
    query_terms = _normalized_terms(query, long_only=long_only)
    candidate_terms = _normalized_terms(candidate, long_only=long_only)
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / len(query_terms)


def _clean_list_subject(value: str) -> str:
    cleaned = " ".join(str(value or "").split())
    cleaned = re.sub(
        r"^(?:(?:complete|full|exhaustive)\s+)?"
        r"(?:list|names?|locations?)\s+(?:of\s+)?(?:all\s+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"^all\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _subject_anchor_terms(value: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", value or ""):
        token = raw.casefold()
        if token in _GENERIC_QUERY_WORDS or len(token) < 3:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is")):
            token = token[:-1]
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    if len(ordered) > 1:
        # The final token is often a generic head noun (station, control, charge, rule).
        # The modifier is what distinguishes the requested set from arbitrary rows.
        return ordered[:-1]
    return ordered


def _list_structural_key_signal(subject: str, candidate: Candidate) -> float:
    anchors = _subject_anchor_terms(subject)
    if not anchors:
        return 0.0
    text_value = retrieval_text(candidate).casefold()
    section_value = " > ".join(candidate.section_path or []).casefold()
    hits = 0
    for anchor in anchors:
        stem = re.escape(anchor)
        in_section = re.search(rf"\b{stem}[a-z0-9]*\b", section_value) is not None
        key_left = re.search(
            rf"\b{stem}[a-z0-9]*\b[^=|;,.]{{0,32}}=",
            text_value,
        ) is not None
        key_right = re.search(
            rf"=\s*[^|;,.]{{0,24}}\b{stem}[a-z0-9]*\b",
            text_value,
        ) is not None
        if in_section or key_left or key_right:
            hits += 1
    return hits / len(anchors)


def _list_member_density_signal(subject: str, candidate: Candidate) -> float:
    if _list_structural_key_signal(subject, candidate) <= 0.0:
        return 0.0
    text_value = retrieval_text(candidate)
    row_markers = len(re.findall(r"(?:^|[|.;]\s*)\d{1,3}\s*[,.)|]", text_value))
    equals = text_value.count("=")
    comma_runs = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9 -]{0,30},\s*[A-Za-z0-9]", text_value))
    if row_markers >= 4 or equals >= 8 or comma_runs >= 3:
        return 1.0
    if row_markers >= 2 or equals >= 4 or comma_runs >= 1:
        return 0.6
    return 0.2


@dataclass(slots=True)
class ToolExecution:
    tool: str
    arguments: dict
    items: list[dict] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    query_groups: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0
    note: str = ""

    def trace(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "elapsed_ms": self.elapsed_ms,
            "note": self.note,
            "items": self.items,
            "query_groups": self.query_groups,
        }


class CorpusMCPServer:
    """Read-only corpus tool boundary used by the research agent.

    The interface is MCP-shaped: named tools receive JSON-like arguments and return
    JSON-like observations. Arbitrary SQL is never exposed to the model.
    """

    TOOL_NAMES = {
        "search_documents",
        "search_chunks",
        "search_many",
        "search_lists",
        "search_sections",
        "get_document_structure",
        "get_section",
        "search_tables",
        "exact_lookup",
    }

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
        self.retrieval = RetrievalEngine(db)
        self.allowed_document_ids = list(allowed_document_ids or [])

    @classmethod
    def tool_catalog(cls) -> list[dict]:
        return [
            {
                "name": "search_documents",
                "description": "Find likely source documents by title, filename and corpus index. Use for a named or suspected source.",
                "arguments": {"query": "string"},
            },
            {
                "name": "search_chunks",
                "description": "Hybrid semantic+lexical chunk search. Default tool for one factual or procedural need.",
                "arguments": {"query": "string", "document_ids": ["uuid"]},
            },
            {
                "name": "search_many",
                "description": "Run independent hybrid searches for multiple distinct requested parts. Provide one semantic query per part; results remain grouped so one category cannot crowd out another.",
                "arguments": {"queries": ["string"], "document_ids": ["uuid"]},
            },
            {
                "name": "search_lists",
                "description": "Find set-bearing list, directory or table evidence for one or more requested categories. Provide one semantic subject per requested set. Use for list/all/names/locations enumeration requests rather than ordinary mention search.",
                "arguments": {"subjects": ["string"], "document_ids": ["uuid"]},
            },
            {
                "name": "search_sections",
                "description": "Find section, chapter or topic anchors in the document hierarchy.",
                "arguments": {"query": "string", "document_ids": ["uuid"]},
            },
            {
                "name": "get_document_structure",
                "description": "Return top-level headings for one document with representative evidence chunks.",
                "arguments": {"document_id": "uuid", "query": "optional structural selector"},
            },
            {
                "name": "get_section",
                "description": "Fetch chunks belonging to the best matching named section in one document.",
                "arguments": {"document_id": "uuid", "section_selector": "string"},
            },
            {
                "name": "search_tables",
                "description": "Search table chunks when the answer is likely in structured rows or columns.",
                "arguments": {"query": "string", "document_ids": ["uuid"]},
            },
            {
                "name": "exact_lookup",
                "description": "Exact lookup for identifiers, quotations, rule numbers, codes or precise phrases.",
                "arguments": {"query": "string", "document_ids": ["uuid"]},
            },
        ]

    def _accessible_documents(self) -> list[Document]:
        stmt = select(Document).where(document_access_clause(self.user))
        if self.allowed_document_ids:
            stmt = stmt.where(Document.id.in_(self.allowed_document_ids))
        return list(self.db.scalars(stmt).all())

    def _normalize_scope(self, raw_ids: list[str] | None) -> list[UUID] | None:
        ids: list[UUID] = []
        for raw in raw_ids or []:
            try:
                ids.append(UUID(str(raw)))
            except (TypeError, ValueError):
                continue
        ids = list(dict.fromkeys(ids))
        if self.allowed_document_ids:
            allowed = set(self.allowed_document_ids)
            ids = [item for item in ids if item in allowed]
            return ids or list(self.allowed_document_ids)
        if not ids:
            return None
        accessible = set(
            self.db.scalars(
                select(Document.id).where(
                    Document.id.in_(ids),
                    document_access_clause(self.user),
                )
            ).all()
        )
        return [item for item in ids if item in accessible] or None

    @staticmethod
    def _item(candidate: Candidate, *, source: str | None = None) -> dict:
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
            "source": source or candidate.rank_method,
            "snippet": " ".join(
                (candidate.text or candidate.contextual_text or "").split()
            )[:1200],
        }

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
        q_tokens = [
            value for value in re.findall(r"[a-z0-9]+", q)
            if len(value) >= 2
        ]
        if q_tokens and all(token in t for token in q_tokens):
            score += 4.0
        return score

    def search_documents(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> ToolExecution:
        started = time.perf_counter()
        top_k = min(
            max(1, top_k or self.settings.agent_max_documents),
            self.settings.agent_max_documents,
        )
        documents = self._accessible_documents()
        corpus_scores: dict[UUID, float] = {}
        try:
            discovery = self.retrieval.discover_corpus(
                query, self.user, self.allowed_document_ids or None
            )
            for rank, hit in enumerate(discovery.hits[:40], start=1):
                corpus_scores[hit.document_id] = max(
                    corpus_scores.get(hit.document_id, 0.0),
                    float(hit.score or 0.0) + (1.0 / rank),
                )
        except Exception:
            pass

        scored: list[tuple[float, Document]] = []
        for document in documents:
            title_score = self._title_score(query, document)
            score = title_score + corpus_scores.get(document.id, 0.0)
            if score > 0:
                # Apply richness only among documents whose own title/filename matched the
                # requested source. Corpus-only related documents must not jump ahead merely
                # because they are long.
                if title_score > 0:
                    score += min(
                        3.5,
                        0.6 * math.log1p(max(0, int(document.page_count or 0))),
                    )
                scored.append((score, document))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)

        items = []
        for score, document in scored[:top_k]:
            items.append(
                {
                    "document_id": str(document.id),
                    "title": document.title,
                    "filename": document.original_filename,
                    "revision": document.revision,
                    "authority": document.authority,
                    "source_role": document.source_role,
                    "authority_level": document.authority_level,
                    "page_count": document.page_count,
                    "score": round(score, 6),
                }
            )
        return ToolExecution(
            tool="search_documents",
            arguments={"query": query, "top_k": top_k},
            items=items,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Document candidates are routing hints, not answer evidence.",
        )

    def _hybrid_search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        tables_only: bool = False,
        exact_only: bool = False,
    ) -> ToolExecution:
        started = time.perf_counter()
        scope = self._normalize_scope(document_ids)
        top_k = min(
            max(1, top_k or self.settings.agent_search_top_k),
            self.settings.agent_search_top_k,
        )
        prefilter = max(top_k, self.settings.agent_search_prefilter_k)
        lanes: list[tuple[str, list[Candidate], float]] = []

        if exact_only:
            exact = self.retrieval._exact(query, self.user, scope, prefilter)
            for candidate in exact:
                candidate.sources.add("agent:exact")
            lanes.append(("exact", exact, 1.5))
        elif tables_only:
            tables = self.retrieval._table_lexical(
                query, self.user, scope, prefilter
            )
            lanes.append(("table", tables, 1.2))
            try:
                vector = self.retrieval.inference.embed_query(query)
                dense = [
                    candidate
                    for candidate in self.retrieval._dense(
                        vector, self.user, scope, prefilter
                    )
                    if candidate.content_kind == "table"
                ]
                for candidate in dense:
                    candidate.sources.add("agent:dense_table")
                lanes.append(("dense_table", dense, 1.0))
            except Exception:
                pass
        else:
            try:
                vector = self.retrieval.inference.embed_query(query)
                dense = self.retrieval._dense(
                    vector, self.user, scope, prefilter
                )
                for candidate in dense:
                    candidate.sources.add("agent:dense")
                lanes.append(("dense", dense, 1.0))
            except Exception:
                pass

            for name, weight, fn in (
                ("lexical", 1.25, self.retrieval._lexical),
                ("relaxed", 0.72, self.retrieval._relaxed_lexical),
                ("section", 0.95, self.retrieval._section_navigation_candidates),
                ("table", 0.65, self.retrieval._table_lexical),
                ("exact", 1.35, self.retrieval._exact),
            ):
                try:
                    values = fn(query, self.user, scope, prefilter)
                except Exception:
                    values = []
                for candidate in values:
                    candidate.sources.add(f"agent:{name}")
                lanes.append((name, values, weight))

        pool: dict[UUID, Candidate] = {}
        ranked_lists: list[tuple[str, list[UUID], float]] = []
        for name, values, weight in lanes:
            if not values:
                continue
            ranked_lists.append(
                (name, [candidate.chunk_id for candidate in values], weight)
            )
            for candidate in values:
                current = pool.get(candidate.chunk_id)
                if current is None:
                    pool[candidate.chunk_id] = candidate
                else:
                    current.sources.update(candidate.sources)

        if not pool:
            return ToolExecution(
                tool=(
                    "search_tables"
                    if tables_only
                    else ("exact_lookup" if exact_only else "search_chunks")
                ),
                arguments={
                    "query": query,
                    "document_ids": [str(value) for value in (scope or [])],
                },
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                note="No accessible candidates found.",
            )

        fused = reciprocal_rank_fusion(ranked_lists)
        ranked: list[Candidate] = []
        for chunk_id, (score, sources) in fused.items():
            candidate = pool[chunk_id]
            candidate.fused_score = score
            section_text = " > ".join(candidate.section_path or [])
            local_text = (candidate.text or candidate.contextual_text or "")[:1800]
            title_overlap = _normalized_overlap(
                query, candidate.document_title, long_only=True
            )
            section_overlap = _normalized_overlap(
                query, section_text, long_only=True
            )
            local_overlap = _normalized_overlap(query, local_text)
            # Long topic words in a document title are a stronger applicability signal than
            # short role acronyms or generic index matches. Section/local overlap remain
            # useful bounded bonuses. All signals are derived from query + corpus text.
            relevance_bonus = (
                (0.10 * title_overlap)
                + (0.04 * section_overlap)
                + (0.015 * local_overlap)
            )
            candidate.final_retrieval_score = score + relevance_bonus
            candidate.rank_method = "agent_rrf"
            candidate.sources.update(f"agent:{source}" for source in sources)
            ranked.append(candidate)
        ranked.sort(key=lambda item: item.final_retrieval_score, reverse=True)
        ranked = deduplicate_candidates(ranked)[:prefilter]

        # Cross-encoder reranking is optional on the agent path. It is disabled by
        # default because a CPU reranker can dominate latency; the agent can instead issue
        # a second targeted search after inspecting fast hybrid results.
        if self.settings.agent_rerank_enabled and len(ranked) > 1:
            try:
                results, _timing = self.retrieval.inference.rerank(
                    query,
                    [retrieval_text(candidate)[: self.settings.agent_rerank_text_chars] for candidate in ranked],
                    top_k=min(top_k, len(ranked)),
                    request_id=self.request_id,
                )
                reranked: list[Candidate] = []
                for result in results:
                    candidate = ranked[result.index]
                    candidate.rerank_score = result.score
                    candidate.final_retrieval_score = result.score
                    candidate.rank_method = "agent_cross_encoder"
                    candidate.sources.add("agent:reranked")
                    reranked.append(candidate)
                ranked = reranked
            except Exception:
                ranked = ranked[:top_k]
        else:
            ranked = ranked[:top_k]

        ranked = ranked[:top_k]
        tool_name = (
            "search_tables"
            if tables_only
            else ("exact_lookup" if exact_only else "search_chunks")
        )
        return ToolExecution(
            tool=tool_name,
            arguments={
                "query": query,
                "document_ids": [str(value) for value in (scope or [])],
                "top_k": top_k,
            },
            candidates=ranked,
            items=[self._item(candidate) for candidate in ranked],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def search_chunks(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        return self._hybrid_search(
            query, document_ids=document_ids, top_k=top_k
        )


    def search_many(
        self,
        queries: list[str],
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        """Run independent hybrid searches while preserving one result group per need.

        This is intentionally not one concatenated query. Multi-part user requests can have
        evidence in different documents, and each semantic subquery gets its own retrieval/rerank
        pass so a strong result for one part cannot crowd out another part before the agent sees it.
        """
        started = time.perf_counter()
        cleaned: list[str] = []
        seen_queries: set[str] = set()
        for raw in queries or []:
            query = " ".join(str(raw or "").split())
            key = query.casefold()
            if not query or key in seen_queries:
                continue
            seen_queries.add(key)
            cleaned.append(query)
            if len(cleaned) >= 4:
                break

        if not cleaned:
            return ToolExecution(
                tool="search_many",
                arguments={"queries": [], "document_ids": document_ids or []},
                note="No usable subqueries supplied.",
            )

        per_query_top_k = min(
            max(1, top_k or min(6, self.settings.agent_search_top_k)),
            self.settings.agent_search_top_k,
        )
        groups: list[dict] = []
        items: list[dict] = []
        candidates: list[Candidate] = []
        seen_chunks: set[UUID] = set()

        for index, query in enumerate(cleaned, start=1):
            execution = self._hybrid_search(
                query,
                document_ids=document_ids,
                top_k=per_query_top_k,
            )
            group_items: list[dict] = []
            for item in execution.items:
                enriched = dict(item)
                enriched["query"] = query
                group_items.append(enriched)
                items.append(enriched)
            groups.append(
                {
                    "query": query,
                    "elapsed_ms": execution.elapsed_ms,
                    "items": group_items,
                }
            )
            for candidate in execution.candidates:
                candidate.sources.add(f"agent:search_many:{index}")
                if candidate.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(candidate.chunk_id)
                candidates.append(candidate)

        return ToolExecution(
            tool="search_many",
            arguments={
                "queries": cleaned,
                "document_ids": list(document_ids or []),
                "top_k_per_query": per_query_top_k,
            },
            items=items,
            candidates=candidates,
            query_groups=groups,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Independent retrieval groups; each query represents a distinct requested part.",
        )


    def search_lists(
        self,
        subjects: list[str],
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        """Retrieve set-bearing evidence independently for each semantic subject.

        The model supplies the subjects. The tool combines several domain-neutral signals:
        list/table shape, matching section headings, lexical matches and a final semantic
        rerank that asks whether the candidate actually enumerates members of the requested set.
        """
        started = time.perf_counter()
        scope = self._normalize_scope(document_ids)
        cleaned: list[str] = []
        seen_subjects: set[str] = set()
        for raw in subjects or []:
            subject = _clean_list_subject(str(raw or ""))
            key = subject.casefold()
            if not subject or key in seen_subjects:
                continue
            seen_subjects.add(key)
            cleaned.append(subject)
            if len(cleaned) >= 4:
                break

        if not cleaned:
            return ToolExecution(
                tool="search_lists",
                arguments={"subjects": [], "document_ids": [str(value) for value in (scope or [])]},
                note="No usable list subjects supplied.",
            )

        per_subject_limit = min(
            max(1, top_k or 6),
            min(8, self.settings.agent_search_top_k),
        )
        prefilter_limit = max(18, per_subject_limit * 3)
        subject_pools: list[list[Candidate]] = []
        subject_stats: list[dict] = []

        for subject in cleaned:
            plan = QueryPlan(
                original=subject,
                normalized=subject,
                intent="enumeration",
                topic=subject,
                answer_type="structured_list",
                entity_term=subject,
                entity_terms=[subject],
                coverage_kind="entity_attribute",
                facets=["enumeration"],
            )
            structural, stats = self.retrieval._entity_attribute_coverage_candidates(
                plan,
                self.user,
                scope,
                max_documents=min(
                    16,
                    max(
                        prefilter_limit,
                        getattr(self.settings, "coverage_max_documents", 16),
                    ),
                ),
                scan_top_k=getattr(self.settings, "coverage_scan_top_k", 800),
            )
            try:
                section_candidates = self.retrieval._section_navigation_candidates(
                    subject, self.user, scope, min(24, prefilter_limit)
                )
            except Exception:
                section_candidates = []
            try:
                table_candidates = self.retrieval._table_lexical(
                    subject, self.user, scope, min(24, prefilter_limit)
                )
            except Exception:
                table_candidates = []
            try:
                lexical_candidates = self.retrieval._lexical(
                    subject, self.user, scope, min(24, prefilter_limit)
                )
            except Exception:
                lexical_candidates = []

            pool: dict[UUID, Candidate] = {}
            for source_name, values in (
                ("coverage", structural),
                ("section", section_candidates),
                ("table", table_candidates),
                ("lexical", lexical_candidates),
            ):
                for candidate in values:
                    existing = pool.get(candidate.chunk_id)
                    if existing is None:
                        pool[candidate.chunk_id] = candidate
                        existing = candidate
                    existing.sources.add(f"agent:list:{source_name}")

            ranked_pool: list[tuple[float, Candidate]] = []
            for candidate in pool.values():
                shape = enumeration_shape_score(subject, candidate)
                local_heading = " > ".join(candidate.section_path or [])
                heading_overlap = _normalized_overlap(subject, local_heading)
                title_overlap = _normalized_overlap(subject, candidate.document_title or "")
                key_signal = _list_structural_key_signal(subject, candidate)
                member_density = _list_member_density_signal(subject, candidate)
                table_bonus = 0.10 if candidate.content_kind == "table" else 0.0
                source_bonus = (
                    0.07 if "agent:list:section" in candidate.sources else 0.0
                ) + (
                    0.05 if "agent:list:coverage" in candidate.sources else 0.0
                )
                cheap_score = (
                    (0.38 * shape)
                    + (0.16 * heading_overlap)
                    + (0.05 * title_overlap)
                    + (0.22 * key_signal)
                    + (0.14 * member_density)
                    + table_bonus
                    + source_bonus
                )
                if len(_subject_anchor_terms(subject)) > 0 and key_signal <= 0.0:
                    cheap_score -= 0.18
                candidate.final_retrieval_score = max(
                    candidate.final_retrieval_score, cheap_score
                )
                ranked_pool.append((cheap_score, candidate))
            ranked_pool.sort(
                key=lambda item: (
                    item[0],
                    item[1].final_retrieval_score,
                    -item[1].ordinal,
                ),
                reverse=True,
            )
            subject_pools.append(
                [candidate for _score, candidate in ranked_pool[:prefilter_limit]]
            )
            entity_stats = (
                ((stats.get("entities") or [{}])[0])
                if isinstance(stats, dict)
                else {}
            )
            subject_stats.append(entity_stats)

        groups: list[dict] = []
        all_items: list[dict] = []
        all_candidates: list[Candidate] = []
        seen_chunks: set[UUID] = set()

        for subject_index, (subject, pool, stats) in enumerate(
            zip(cleaned, subject_pools, subject_stats, strict=True),
            start=1,
        ):
            # The research-controller LLM will inspect these concrete list-bearing snippets.
            # Do not add another expensive semantic inference pass here. When a top result
            # names the requested list in its section hierarchy, materialize adjacent chunks
            # from that exact section so split tables arrive as one bounded evidence unit.
            anchors = pool[:per_subject_limit]
            expanded: list[Candidate] = []
            expanded_ids: set[UUID] = set()
            for rank, candidate in enumerate(anchors):
                local_heading = " > ".join(candidate.section_path or [])
                heading_overlap = _normalized_overlap(subject, local_heading)
                expanded_section = False
                if (
                    rank < 3
                    and candidate.section_path
                    and heading_overlap >= 0.8
                ):
                    try:
                        rows = self.db.execute(
                            select(Chunk, Document)
                            .join(Document, Document.id == Chunk.document_id)
                            .where(
                                *self.retrieval._base_filters(
                                    self.user, [candidate.document_id]
                                ),
                                Chunk.section_path == list(candidate.section_path),
                            )
                            .order_by(Chunk.ordinal)
                            .limit(self.settings.agent_max_section_chunks)
                        ).all()
                    except Exception:
                        rows = []
                    if len(rows) > 1:
                        for chunk, document in rows:
                            sibling = self.retrieval._candidate(chunk, document)
                            if sibling.chunk_id in expanded_ids:
                                continue
                            sibling.evidence_lane = "enumeration"
                            sibling.rank_method = "agent_list_section_expansion"
                            sibling.final_retrieval_score = candidate.final_retrieval_score
                            sibling.sources.update(candidate.sources)
                            sibling.sources.add("agent:list_section_expansion")
                            expanded.append(sibling)
                            expanded_ids.add(sibling.chunk_id)
                        expanded_section = True
                if not expanded_section and candidate.chunk_id not in expanded_ids:
                    candidate.rank_method = "agent_list_shape"
                    expanded.append(candidate)
                    expanded_ids.add(candidate.chunk_id)

            selected = expanded[: max(per_subject_limit, 10)]
            for candidate in selected:
                if candidate.rank_method == "unranked":
                    candidate.rank_method = "agent_list_shape"

            group_items: list[dict] = []
            for candidate in selected:
                candidate.evidence_lane = "enumeration"
                candidate.sources.add("agent:structured_list")
                candidate.sources.add(f"agent:list_subject:{subject_index}")
                item = self._item(candidate, source=candidate.rank_method)
                item["query"] = subject
                item["list_subject"] = subject
                group_items.append(item)
                all_items.append(item)
                if candidate.chunk_id not in seen_chunks:
                    seen_chunks.add(candidate.chunk_id)
                    all_candidates.append(candidate)

            groups.append(
                {
                    "query": subject,
                    "subject": subject,
                    "stats": stats,
                    "items": group_items,
                }
            )

        return ToolExecution(
            tool="search_lists",
            arguments={
                "subjects": cleaned,
                "document_ids": [str(value) for value in (scope or [])],
                "top_k_per_subject": per_subject_limit,
            },
            items=all_items,
            candidates=all_candidates,
            query_groups=groups,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note=(
                "Each semantic subject is independently shape-ranked for set-bearing "
                "evidence; incidental mentions are intentionally demoted."
            ),
        )

    def search_tables(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        return self._hybrid_search(
            query,
            document_ids=document_ids,
            top_k=top_k,
            tables_only=True,
        )

    def exact_lookup(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        return self._hybrid_search(
            query,
            document_ids=document_ids,
            top_k=top_k,
            exact_only=True,
        )

    def search_sections(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolExecution:
        started = time.perf_counter()
        scope = self._normalize_scope(document_ids)
        top_k = min(
            max(1, top_k or self.settings.agent_search_top_k),
            self.settings.agent_search_top_k,
        )
        candidates = self.retrieval._section_navigation_candidates(
            query, self.user, scope, max(top_k * 2, top_k)
        )
        candidates = deduplicate_candidates(candidates)[:top_k]
        for candidate in candidates:
            candidate.evidence_lane = "section_navigation"
            candidate.rank_method = "agent_section_navigation"
            candidate.final_retrieval_score = max(
                candidate.final_retrieval_score, 0.7
            )
            candidate.sources.add("agent:section_navigation")
        return ToolExecution(
            tool="search_sections",
            arguments={
                "query": query,
                "document_ids": [str(value) for value in (scope or [])],
                "top_k": top_k,
            },
            candidates=candidates,
            items=[self._item(candidate) for candidate in candidates],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def get_document_structure(self, document_id: str, query: str = "") -> ToolExecution:
        started = time.perf_counter()
        try:
            doc_id = UUID(str(document_id))
        except (TypeError, ValueError):
            return ToolExecution(
                tool="get_document_structure",
                arguments={"document_id": document_id, "query": query},
                note="Invalid document id.",
            )
        scope = self._normalize_scope([str(doc_id)])
        if not scope or doc_id not in scope:
            return ToolExecution(
                tool="get_document_structure",
                arguments={"document_id": document_id, "query": query},
                note="Document is not accessible in the current query scope.",
            )

        document = self.db.scalar(
            select(Document).where(
                Document.id == doc_id, document_access_clause(self.user)
            )
        )
        if document is None:
            return ToolExecution(
                tool="get_document_structure",
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
                    RetrievalNode.ordinal_from, RetrievalNode.page_from
                )
                .limit(5000)
            ).all()
        )
        top: dict[str, RetrievalNode] = {}
        child_headings: dict[str, list[str]] = {}
        for node in nodes:
            path = [
                str(value).strip()
                for value in (node.section_path or [])
                if str(value).strip()
            ]
            if not path:
                continue
            if len(path) > 1:
                bucket = child_headings.setdefault(path[0], [])
                if path[1] not in bucket and len(bucket) < 6:
                    bucket.append(path[1])
            current = top.get(path[0])
            if current is None:
                top[path[0]] = node
                continue
            current_ordinal = (
                current.ordinal_from
                if current.ordinal_from is not None
                else 10**9
            )
            node_ordinal = (
                node.ordinal_from
                if node.ordinal_from is not None
                else 10**9
            )
            if node_ordinal < current_ordinal:
                top[path[0]] = node

        candidates: list[Candidate] = []
        items: list[dict] = []
        structure_query = " ".join(str(query or "").split())
        ordered = sorted(
            top.items(),
            key=lambda item: (
                item[1].ordinal_from
                if item[1].ordinal_from is not None
                else 10**9,
                item[1].page_from
                if item[1].page_from is not None
                else 10**9,
            ),
        )
        if structure_query:
            query_cf = structure_query.casefold()
            filtered = [
                item for item in ordered
                if query_cf in item[0].casefold()
                or _normalized_overlap(structure_query, item[0]) > 0
            ]
            if filtered:
                ordered = filtered
        for label, node in ordered[:160]:
            start = node.ordinal_from if node.ordinal_from is not None else 0
            end = (
                node.ordinal_to
                if node.ordinal_to is not None
                else start + 2
            )
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
            item = {
                "document_id": str(doc_id),
                "document_title": document.title,
                "label": label,
                "section_path": list(node.section_path or []),
                "page_from": node.page_from,
                "page_to": node.page_to,
                "ordinal_from": node.ordinal_from,
                "ordinal_to": node.ordinal_to,
                "child_headings": child_headings.get(label, []),
            }
            if chunk is not None:
                candidate = self.retrieval._candidate(chunk, document)
                candidate.evidence_lane = "document_structure"
                candidate.rank_method = "agent_document_structure"
                candidate.final_retrieval_score = 0.95
                candidate.sources.add("agent:document_structure")
                candidates.append(candidate)
                item["chunk_id"] = str(chunk.id)
                item["snippet"] = " ".join((chunk.text or "").split())[:700]
            items.append(item)

        return ToolExecution(
            tool="get_document_structure",
            arguments={"document_id": str(doc_id), "query": structure_query},
            items=items,
            candidates=candidates,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Top-level document hierarchy.",
        )

    def get_section(
        self,
        document_id: str,
        section_selector: str,
    ) -> ToolExecution:
        started = time.perf_counter()
        try:
            doc_id = UUID(str(document_id))
        except (TypeError, ValueError):
            return ToolExecution(
                tool="get_section",
                arguments={
                    "document_id": document_id,
                    "section_selector": section_selector,
                },
                note="Invalid document id.",
            )
        scope = self._normalize_scope([str(doc_id)])
        if not scope or doc_id not in scope:
            return ToolExecution(
                tool="get_section",
                arguments={
                    "document_id": document_id,
                    "section_selector": section_selector,
                },
                note="Document is not accessible in the current query scope.",
            )

        document = self.db.scalar(
            select(Document).where(
                Document.id == doc_id, document_access_clause(self.user)
            )
        )
        if document is None:
            return ToolExecution(
                tool="get_section",
                arguments={
                    "document_id": document_id,
                    "section_selector": section_selector,
                },
                note="Document is not accessible.",
            )

        nodes = list(
            self.db.scalars(
                select(RetrievalNode)
                .where(
                    RetrievalNode.document_id == doc_id,
                    RetrievalNode.node_type == "section",
                )
                .order_by(RetrievalNode.ordinal_from)
                .limit(5000)
            ).all()
        )
        # Build logical prefix ranges from stored section paths. Some ingesters create
        # nodes only for leaf paths (for example "Chapter > Rule") rather than a standalone
        # parent node. Aggregating prefixes lets a request for the parent retrieve all of its
        # bounded child chunks without teaching the code words such as "chapter" or "rule".
        prefix_ranges: dict[tuple[str, ...], dict] = {}
        for node in nodes:
            path = tuple(
                str(value).strip()
                for value in (node.section_path or [])
                if str(value).strip()
            )
            if not path:
                continue
            for depth in range(1, len(path) + 1):
                prefix = path[:depth]
                bucket = prefix_ranges.setdefault(
                    prefix,
                    {
                        "ordinal_from": None,
                        "ordinal_to": None,
                        "page_from": None,
                        "page_to": None,
                    },
                )
                for key, value, fn in (
                    ("ordinal_from", node.ordinal_from, min),
                    ("page_from", node.page_from, min),
                ):
                    if value is not None:
                        bucket[key] = value if bucket[key] is None else fn(bucket[key], value)
                for key, value, fn in (
                    ("ordinal_to", node.ordinal_to, max),
                    ("page_to", node.page_to, max),
                ):
                    if value is not None:
                        bucket[key] = value if bucket[key] is None else fn(bucket[key], value)

        selector_cf = " ".join(section_selector.casefold().split())
        selector_compact = re.sub(r"[^a-z0-9]", "", selector_cf)
        scored_prefixes: list[tuple[float, tuple[str, ...], dict]] = []
        for prefix, bounds in prefix_ranges.items():
            label = " > ".join(prefix)
            label_cf = " ".join(label.casefold().split())
            label_compact = re.sub(r"[^a-z0-9]", "", label_cf)
            score = 10.0 * _normalized_overlap(section_selector, label)
            if selector_cf and selector_cf in label_cf:
                score += 8.0
            if selector_compact and selector_compact == label_compact:
                score += 12.0
            # Prefer the shortest matching logical prefix when the selector names a parent;
            # a more specific child can still win when its own words uniquely match.
            score -= max(0, len(prefix) - 1) * 0.35
            if score > 0:
                scored_prefixes.append((score, prefix, bounds))

        scored_prefixes.sort(
            key=lambda item: (
                item[0],
                -(
                    item[2]["ordinal_from"]
                    if item[2]["ordinal_from"] is not None
                    else 10**9
                ),
            ),
            reverse=True,
        )
        if not scored_prefixes:
            return ToolExecution(
                tool="get_section",
                arguments={
                    "document_id": str(doc_id),
                    "section_selector": section_selector,
                },
                note="No matching section node found.",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        best_score, best_prefix, bounds = scored_prefixes[0]
        candidates: list[Candidate] = []
        seen_chunks: set[UUID] = set()

        # For a logical parent section, return one representative chunk per immediate
        # child first. This gives the answer model broad coverage of the whole chapter/part
        # instead of exhausting the budget on continuations of the earliest child.
        child_ranges = [
            (prefix, child_bounds)
            for prefix, child_bounds in prefix_ranges.items()
            if len(prefix) == len(best_prefix) + 1
            and prefix[: len(best_prefix)] == best_prefix
        ]
        child_ranges.sort(
            key=lambda item: (
                item[1]["ordinal_from"]
                if item[1]["ordinal_from"] is not None
                else 10**9
            )
        )

        if child_ranges:
            # Preserve content that belongs directly to the requested parent before its
            # child sections. Without this, a parent request can silently skip its opening
            # clauses/preamble and return only deeper children.
            parent_start = (
                bounds["ordinal_from"]
                if bounds["ordinal_from"] is not None
                else 0
            )
            first_child_start = min(
                (
                    child_bounds["ordinal_from"]
                    for _child_prefix, child_bounds in child_ranges
                    if child_bounds["ordinal_from"] is not None
                ),
                default=parent_start,
            )
            parent_end = max(parent_start, first_child_start - 1)
            direct_parent_chunks = list(
                self.db.scalars(
                    select(Chunk)
                    .where(
                        Chunk.document_id == doc_id,
                        Chunk.ordinal >= parent_start,
                        Chunk.ordinal <= parent_end,
                    )
                    .order_by(Chunk.ordinal)
                    .limit(min(3, self.settings.agent_max_section_chunks))
                ).all()
            )
            for chunk in direct_parent_chunks:
                if chunk.id in seen_chunks:
                    continue
                seen_chunks.add(chunk.id)
                candidate = self.retrieval._candidate(chunk, document)
                candidate.evidence_lane = "section_navigation"
                candidate.rank_method = "agent_get_section_parent"
                candidate.final_retrieval_score = max(
                    0.92, min(1.0, best_score / 20.0)
                )
                candidate.sources.add("agent:get_section")
                candidate.sources.add("agent:section_parent")
                candidates.append(candidate)

            remaining = max(
                0,
                self.settings.agent_max_section_chunks - len(candidates),
            )
            for _child_prefix, child_bounds in child_ranges[:remaining]:
                child_start = (
                    child_bounds["ordinal_from"]
                    if child_bounds["ordinal_from"] is not None
                    else 0
                )
                child_end = (
                    child_bounds["ordinal_to"]
                    if child_bounds["ordinal_to"] is not None
                    else child_start
                )
                chunk = self.db.scalar(
                    select(Chunk)
                    .where(
                        Chunk.document_id == doc_id,
                        Chunk.ordinal >= child_start,
                        Chunk.ordinal <= child_end,
                    )
                    .order_by(Chunk.ordinal)
                    .limit(1)
                )
                if chunk is None or chunk.id in seen_chunks:
                    continue
                seen_chunks.add(chunk.id)
                candidate = self.retrieval._candidate(chunk, document)
                candidate.evidence_lane = "section_navigation"
                candidate.rank_method = "agent_get_section_child"
                candidate.final_retrieval_score = max(
                    0.88, min(1.0, best_score / 20.0)
                )
                candidate.sources.add("agent:get_section")
                candidate.sources.add("agent:section_child")
                candidates.append(candidate)
        else:
            start = bounds["ordinal_from"] if bounds["ordinal_from"] is not None else 0
            end = (
                bounds["ordinal_to"]
                if bounds["ordinal_to"] is not None
                else start + self.settings.agent_max_section_chunks - 1
            )
            chunks = list(
                self.db.scalars(
                    select(Chunk)
                    .where(
                        Chunk.document_id == doc_id,
                        Chunk.ordinal >= start,
                        Chunk.ordinal <= end,
                    )
                    .order_by(Chunk.ordinal)
                    .limit(self.settings.agent_max_section_chunks)
                ).all()
            )
            for chunk in chunks:
                if chunk.id in seen_chunks:
                    continue
                seen_chunks.add(chunk.id)
                candidate = self.retrieval._candidate(chunk, document)
                candidate.evidence_lane = "section_navigation"
                candidate.rank_method = "agent_get_section"
                candidate.final_retrieval_score = max(
                    0.85, min(1.0, best_score / 20.0)
                )
                candidate.sources.add("agent:get_section")
                candidates.append(candidate)

        return ToolExecution(
            tool="get_section",
            arguments={
                "document_id": str(doc_id),
                "section_selector": section_selector,
            },
            candidates=candidates,
            items=[self._item(candidate) for candidate in candidates],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            note="Best matching indexed section and bounded original chunks.",
        )

    def execute(self, tool: str, arguments: dict) -> ToolExecution:
        if tool not in self.TOOL_NAMES:
            return ToolExecution(
                tool=tool, arguments=arguments, note="Unknown tool."
            )
        if tool == "search_documents":
            return self.search_documents(
                str(arguments.get("query") or "")
            )
        if tool == "search_chunks":
            return self.search_chunks(
                str(arguments.get("query") or ""),
                document_ids=arguments.get("document_ids") or [],
            )
        if tool == "search_many":
            return self.search_many(
                [str(value) for value in (arguments.get("queries") or [])],
                document_ids=arguments.get("document_ids") or [],
            )
        if tool == "search_lists":
            return self.search_lists(
                [str(value) for value in (arguments.get("subjects") or [])],
                document_ids=arguments.get("document_ids") or [],
            )
        if tool == "search_sections":
            return self.search_sections(
                str(arguments.get("query") or ""),
                document_ids=arguments.get("document_ids") or [],
            )
        if tool == "get_document_structure":
            return self.get_document_structure(
                str(arguments.get("document_id") or ""),
                str(arguments.get("query") or ""),
            )
        if tool == "get_section":
            return self.get_section(
                str(arguments.get("document_id") or ""),
                str(arguments.get("section_selector") or ""),
            )
        if tool == "search_tables":
            return self.search_tables(
                str(arguments.get("query") or ""),
                document_ids=arguments.get("document_ids") or [],
            )
        return self.exact_lookup(
            str(arguments.get("query") or ""),
            document_ids=arguments.get("document_ids") or [],
        )
