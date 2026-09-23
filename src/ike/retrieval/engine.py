import re
import time
from collections import defaultdict
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import desc, func, or_, select, text
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import Chunk, Document, User
from ike.retrieval.access import document_access_clause
from ike.retrieval.applicability import applicability_score
from ike.retrieval.query_plan import QueryPlan, build_query_plan
from ike.retrieval.cache import TTLCache
from ike.retrieval.corpus_intelligence import CorpusDiscovery, CorpusIntelligence
from ike.retrieval.fusion import reciprocal_rank_fusion
from ike.retrieval.lookup import definition_score
from ike.retrieval.evidence_selection import deduplicate_candidates
from ike.retrieval.hierarchical import same_logical_section
from ike.retrieval.normalization import lexical_form_variants, technical_identifier_variants
from ike.retrieval.overview import overview_name_matches
from ike.retrieval.search_plan import (
    content_tokens,
    relaxed_websearch_expression,
    table_retrieval_relevant,
    token_overlap,
)
from ike.retrieval.source_policy import SourcePolicy, source_name_matches
from ike.retrieval.table_context import retrieval_text, table_query_affinity
from ike.retrieval.role import (
    extract_role_aliases,
    is_short_role_identifier,
    role_candidate_score,
    role_section_key,
)
from ike.retrieval.types import Candidate, Evidence
from ike.services.inference_client import InferenceClient
from ike.workflows.routing import (
    coverage_search_query,
    extract_lookup_term,
    extract_role_subject,
    is_coverage_question,
    is_role_coverage_question,
)
from ike.workflows.evidence_planning import (
    EvidencePlan,
    lookup_terms_by_goal,
    plan_query_specs,
)

ProgressFn = Callable[[str, str, str, int], None]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class RetrievalEngine:
    _role_alias_cache: TTLCache | None = None

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.inference = InferenceClient.for_query()
        self.corpus_intelligence = CorpusIntelligence(db)
        self._hnsw_configured = False
        if RetrievalEngine._role_alias_cache is None:
            RetrievalEngine._role_alias_cache = TTLCache(
                max_items=self.settings.role_alias_cache_max_items,
                ttl_seconds=self.settings.role_alias_cache_ttl_seconds,
            )

    def discover_corpus(
        self,
        question: str,
        user: User,
        document_ids: list[UUID] | None = None,
    ) -> CorpusDiscovery:
        return self.corpus_intelligence.discover(question, user, document_ids)

    @staticmethod
    def _emit(progress: ProgressFn | None, stage: str, label: str, detail: str, percent: int) -> None:
        if progress:
            progress(stage, label, detail, percent)

    def _base_filters(self, user: User, document_ids: list[UUID] | None) -> list:
        filters = [document_access_clause(user)]
        if document_ids:
            filters.append(Document.id.in_(document_ids))
        return filters

    def resolve_named_source_scope(
        self,
        source_scope: str | None,
        user: User,
        *,
        max_documents: int = 24,
    ) -> list[UUID]:
        """Resolve an explicit documentary source phrase to accessible ready documents.

        This is intentionally title/filename based. A strong user phrase such as MRGR, ADM,
        HR Compendium or train operation handbook is a document selector, not answer text.
        Weak/non-document qualifiers should already have been rejected by routing.
        """
        cleaned = re.sub(r"\s+", " ", str(source_scope or "").strip())
        if not cleaned:
            return []
        compact = re.sub(r"[^a-z0-9]", "", cleaned.casefold())
        tokens = [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", cleaned)
            if len(token) >= 3
        ]
        if not compact:
            return []

        rows = self.db.execute(
            select(
                Document.id,
                Document.title,
                Document.original_filename,
                Document.ingestion_status,
                Document.lifecycle_status,
            ).where(document_access_clause(user), Document.ingestion_status == "ready")
        ).all()
        ranked: list[tuple[int, UUID]] = []
        for document_id, title, filename, _status, lifecycle_status in rows:
            if lifecycle_status not in {None, "active"}:
                continue
            name = f"{title or ''} {filename or ''}"
            name_compact = re.sub(r"[^a-z0-9]", "", name.casefold())
            score = 0
            if compact in name_compact:
                score += 20
            token_hits = sum(1 for token in tokens if token in name.casefold())
            score += token_hits * 4
            if score:
                ranked.append((score, document_id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return []
        best = ranked[0][0]
        # Named-source scoping must be high precision. Keep close title aliases/copies only.
        floor = max(12, best - 8)
        return [document_id for score, document_id in ranked if score >= floor][:max_documents]

    @staticmethod
    def _governing_lane_relevant(query_plan: QueryPlan) -> bool:
        """Use the configured governing-reference lane only when it fits the question."""
        if query_plan.source_scope:
            return True
        if query_plan.intent in {"procedure", "troubleshooting", "responsibilities"}:
            return True
        if query_plan.line or query_plan.rolling_stock or query_plan.system or query_plan.subsystem:
            return True
        text_value = query_plan.original
        return bool(
            re.search(
                r"(?:SC|SI|TI|ATP|ATO|UTO|TCMS|TIMS|OCC|TO|SC|RA|ETO|signal|train|"
                r"station|depot|track|traction|OHE|pantograph|door|brake|coupling)",
                text_value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _candidate(chunk: Chunk, document: Document) -> Candidate:
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
            document_profile=(document.extra_metadata or {}).get("operational_profile", {}),
        )

    def _dense(
        self,
        vector: list[float],
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        # pgvector HNSW filtering happens after the initial approximate scan. Iterative scans
        # protect recall when ACL/document filters are selective. These settings are local to
        # the current transaction and require pgvector >= 0.8.0 (the Compose image is pinned).
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
        return [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def _lexical(
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
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank))
            .limit(top_k)
        )
        return [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def _relaxed_lexical(
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
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(top_k)
        )
        candidates = [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
        for candidate in candidates:
            candidate.sources.add("relaxed")
        return candidates

    def _section_navigation_candidates(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        """Find section anchors using the existing indexed chunk corpus.

        No reprocessing or schema change is required: contextual_text already includes
        headings and section_path is stored separately.  We first use the indexed relaxed
        lexical query to obtain a bounded scan, then promote candidates whose hierarchy
        matches more of the information need and keep one anchor per logical section.
        """

        if not self.settings.section_navigation_enabled:
            return []
        expression = relaxed_websearch_expression(query, max_terms=12)
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
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(scan_limit)
        )
        scored: list[tuple[float, Candidate]] = []
        for chunk, document, rank_value in self.db.execute(stmt).all():
            candidate = self._candidate(chunk, document)
            heading = " / ".join(candidate.section_path or [])
            heading_overlap = token_overlap(query, heading) if heading else 0.0
            local_overlap = token_overlap(query, (candidate.contextual_text or candidate.text)[:1600])
            # FTS rank remains the primary signal; hierarchy overlap is a bounded bonus.
            score = float(rank_value or 0.0) + (1.25 * heading_overlap) + (0.35 * local_overlap)
            candidate.sources.add("section")
            scored.append((score, candidate))

        scored.sort(key=lambda item: (item[0], -item[1].ordinal), reverse=True)
        selected: list[Candidate] = []
        seen_sections: set[tuple] = set()
        for _score, candidate in scored:
            key = (
                candidate.document_id,
                tuple(item.strip().casefold() for item in (candidate.section_path or []) if item.strip()),
            )
            if not key[1]:
                key = (candidate.document_id, (f"chunk:{candidate.chunk_id}",))
            if key in seen_sections:
                continue
            seen_sections.add(key)
            selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def _table_lexical(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        """Dedicated table lane over already-contextualized table chunks.

        Docling's HybridChunker was configured with repeat_table_header=True at ingestion,
        so search_vector/contextual_text already carry row header context for the existing
        corpus. This lane protects terse table facts from verbose-paragraph crowd-out.
        """
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
        candidates = [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
        for candidate in candidates:
            candidate.sources.add("table")
        return candidates

    def _exact(
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
            .where(*self._base_filters(user, document_ids), Chunk.contextual_text.ilike(pattern, escape="\\"))
            .order_by(Chunk.document_id, Chunk.ordinal)
            .limit(top_k)
        )
        return [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def _definition_candidates(
        self, term: str, user: User, document_ids: list[UUID] | None, top_k: int, source_scope: str | None = None
    ) -> list[Candidate]:
        """Find explicit definition-bearing chunks before broad semantic retrieval.

        Common institutional terms may occur in hundreds of procedural passages.  Querying
        those mentions first makes the result depend on SQL row order.  Search explicit
        definitional constructions first, then use a bounded fallback only when needed.
        """
        cleaned = re.sub(r"\s+", " ", term.strip())
        if not cleaned:
            return []

        base = list(self._base_filters(user, document_ids))
        if source_scope:
            scope = f"%{_escape_like(source_scope.strip())}%"
            base.append(
                Document.title.ilike(scope, escape="\\")
                | Document.original_filename.ilike(scope, escape="\\")
            )

        escaped = _escape_like(cleaned)
        structural_patterns = [
            f"%{escaped}% means %",
            f"%{escaped}%' means %",
            f'%{escaped}%" means %',
            f"%{escaped}% shall mean %",
            f"%{escaped}%' shall mean %",
            f"%{escaped}% defined as %",
            f"%{escaped}% is defined as %",
            f"%{escaped}% refers to %",
        ]
        term_tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9]+", cleaned)]
        flexible_term = r"\s+".join(term_tokens)
        definition_regex = (
            flexible_term
            + r".{0,40}(?:means|shall\s+mean|defined\s+as|is\s+defined\s+as|refers\s+to)"
            if flexible_term
            else r"(?!x)x"
        )
        structural_filter = or_(
            *[Chunk.contextual_text.ilike(pattern, escape="\\") for pattern in structural_patterns],
            Chunk.contextual_text.op("~*")(definition_regex),
        )
        stmt = (
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(*base, structural_filter)
            .limit(max(80, top_k * 12))
        )
        rows = self.db.execute(stmt).all()

        # Definitions are also frequently represented as a table/glossary row or under a
        # Definitions heading without the literal word "means". Add a bounded fallback only
        # after the explicit-pattern query.
        if len(rows) < top_k:
            term_pattern = f"%{escaped}%"
            fallback = (
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.document_id)
                .where(*base, Chunk.contextual_text.ilike(term_pattern, escape="\\"))
                .limit(1200)
            )
            seen = {row[0].id for row in rows}
            rows.extend(row for row in self.db.execute(fallback).all() if row[0].id not in seen)

        candidates = [self._candidate(row[0], row[1]) for row in rows]
        ranked: list[tuple[int, Candidate]] = []
        for candidate in candidates:
            score = definition_score(cleaned, candidate)
            if score <= 0:
                continue
            candidate.sources.update({"definition_structural", "lookup"})
            ranked.append((score, candidate))
        ranked.sort(
            key=lambda item: (
                item[0],
                "definition" in " ".join(item[1].section_path or []).casefold(),
                -item[1].ordinal,
            ),
            reverse=True,
        )
        return [candidate for _, candidate in ranked[:top_k]]

    def _identifier_candidates(
        self,
        term: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        """Resolve a bare technical identifier from titles/indexes before semantics."""
        cleaned = re.sub(r"\s+", " ", term.strip())
        compact_term = re.sub(r"[^a-z0-9]", "", cleaned.casefold())
        if len(compact_term) < 2:
            return []

        pattern = f"%{_escape_like(cleaned)}%"
        stmt = (
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                *self._base_filters(user, document_ids),
                (
                    Document.title.ilike(pattern, escape="\\")
                    | Document.original_filename.ilike(pattern, escape="\\")
                    | Chunk.contextual_text.ilike(pattern, escape="\\")
                ),
            )
            .limit(900)
        )
        ranked: list[tuple[int, Candidate]] = []
        for chunk, document in self.db.execute(stmt).all():
            candidate = self._candidate(chunk, document)
            text_value = candidate.contextual_text or candidate.text or ""
            title = f"{candidate.document_title} {candidate.filename}"
            section = " ".join(candidate.section_path or [])
            compact_title = re.sub(r"[^a-z0-9]", "", title.casefold())
            compact_text = re.sub(r"[^a-z0-9]", "", text_value.casefold())
            compact_section = re.sub(r"[^a-z0-9]", "", section.casefold())
            score = 0
            if compact_term in compact_title:
                score += 20
            if "index" in title.casefold() and compact_term in compact_text:
                score += 14
            if compact_term in compact_section:
                score += 8
            if compact_term in compact_text:
                score += 5
            if candidate.content_kind == "table":
                score += 2
            if score <= 0:
                continue
            candidate.sources.update({"identifier_structural", "lookup"})
            ranked.append((score, candidate))

        ranked.sort(
            key=lambda item: (
                item[0],
                "index" in item[1].document_title.casefold(),
                -item[1].ordinal,
            ),
            reverse=True,
        )
        selected: list[Candidate] = []
        seen_docs: set[UUID] = set()
        for _score, candidate in ranked:
            if candidate.document_id in seen_docs and len(selected) >= max(2, top_k // 2):
                continue
            selected.append(candidate)
            seen_docs.add(candidate.document_id)
            if len(selected) >= top_k:
                break
        return selected

    def _lookup_candidates(
        self,
        term: str,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        """Retrieve a broad, document-diverse pool for short acronym/definition lookups.

        This avoids the common RAG failure where repeated hits from one manual consume the
        candidate budget and hide a different expansion in another manual.
        """

        tsquery = func.plainto_tsquery("simple", term)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        scan_limit = min(max(top_k * 4, top_k), 1200)
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(scan_limit)
        )
        candidates = [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]
        for candidate in candidates:
            candidate.sources.add("lookup")

        ranked = sorted(
            candidates,
            key=lambda item: (definition_score(term, item), -item.ordinal),
            reverse=True,
        )
        # First pass guarantees breadth across documents; second pass fills remaining slots.
        selected: list[Candidate] = []
        seen_ids: set[UUID] = set()
        seen_documents: set[UUID] = set()
        for candidate in ranked:
            if candidate.document_id in seen_documents:
                continue
            selected.append(candidate)
            seen_ids.add(candidate.chunk_id)
            seen_documents.add(candidate.document_id)
            if len(selected) >= top_k:
                return selected
        for candidate in ranked:
            if candidate.chunk_id in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate.chunk_id)
            if len(selected) >= top_k:
                break
        return selected

    def _coverage_candidates(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
        *,
        max_documents: int,
        scan_top_k: int,
    ) -> list[Candidate]:
        """Return one strong lexical/structural hit per matching document.

        Coverage-sensitive questions such as procedures must not let one manual consume
        the entire global top-k. This pass deliberately discovers documents first and
        reserves one representative passage from each before neighbour expansion.
        """

        topic_query = coverage_search_query(query)
        coverage_forms = technical_identifier_variants(
            topic_query, max_variants=self.settings.identifier_variant_max_queries
        ) if self.settings.identifier_variant_expansion else [topic_query]
        coverage_forms.extend(lexical_form_variants(topic_query))
        unique_forms: list[str] = []
        seen_forms: set[str] = set()
        for form in coverage_forms:
            key = form.casefold()
            if key not in seen_forms:
                seen_forms.add(key)
                unique_forms.append(form)
        # websearch_to_tsquery understands OR and keeps this a single indexed SQL query.
        coverage_expression = " OR ".join(f'"{form}"' for form in unique_forms[:6])
        tsquery = func.websearch_to_tsquery("simple", coverage_expression or topic_query)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(max(100, scan_top_k))
        )
        rows = self.db.execute(stmt).all()
        selected: list[Candidate] = []
        seen_documents: set[UUID] = set()
        # Ask for one extra document so callers can detect truncation at the configured cap.
        target = max(1, max_documents) + 1
        for chunk, document, _rank in rows:
            if document.id in seen_documents:
                continue
            candidate = self._candidate(chunk, document)
            candidate.sources.add("coverage")
            selected.append(candidate)
            seen_documents.add(document.id)
            if len(selected) >= target:
                break
        return selected

    def _entity_attribute_coverage_candidates(
        self,
        plan: QueryPlan,
        user: User,
        document_ids: list[UUID] | None,
        *,
        max_documents: int,
        scan_top_k: int,
    ) -> tuple[list[Candidate], dict]:
        """Return balanced document-diverse evidence for one or more entity targets.

        Each coordinated category is searched independently. This is crucial for questions
        such as ``List of depots and crew controls``: requiring the combined phrase would
        only retrieve chunks where both categories happen to co-occur and systematically
        miss authoritative lists for either category. Attribute words remain ranking hints,
        never mandatory terms, so terse table rows remain discoverable.
        """

        entities = [
            value.strip()
            for value in (plan.entity_terms or ([plan.entity_term] if plan.entity_term else []))
            if value and value.strip()
        ]
        if not entities and plan.lookup_term:
            entities = [plan.lookup_term.strip()]
        if not entities:
            return [], {"entities": [], "truncated": False, "sql_queries": 0}

        # Bound coordinated-set fan-out without collapsing the categories back into one
        # phrase. The parser itself already caps extraction at five list members.
        entities = entities[:5]
        attribute_terms = [term.casefold() for term in plan.attribute_terms if term.strip()]
        facets = set(plan.facets or [])

        if "enumeration" in facets:
            structural_terms = (
                "general information", "directory", "master list", "list of",
                "number of", "no. of", "total", "name",
            )
        elif "location" in facets:
            structural_terms = (
                "general information", "location", "address", "site", "room",
                "office", "station", "depot", "directory",
            )
        elif "contact" in facets:
            structural_terms = ("contact", "phone", "mobile", "extension", "telephone", "directory")
        else:
            structural_terms = ("general information", "definition", "details")

        # Precompute surface forms for sibling co-location scoring. A table/section covering
        # more than one requested category is usually a stronger structured-list source.
        forms_by_entity: list[list[str]] = []
        for entity in entities:
            forms = (
                technical_identifier_variants(
                    entity, max_variants=self.settings.identifier_variant_max_queries
                )
                if self.settings.identifier_variant_expansion
                else [entity]
            )
            forms.extend(lexical_form_variants(entity, max_variants=6))
            unique_forms: list[str] = []
            seen_forms: set[str] = set()
            for form in forms:
                cleaned = re.sub(r"\s+", " ", form.strip()).replace('"', " ")
                key = cleaned.casefold()
                if cleaned and key not in seen_forms:
                    seen_forms.add(key)
                    unique_forms.append(cleaned)
            forms_by_entity.append(unique_forms or [entity])

        selected_by_entity: list[list[Candidate]] = []
        per_entity_stats: list[dict] = []
        sql_queries = 0
        any_truncated = False

        for entity_index, (entity, unique_forms) in enumerate(zip(entities, forms_by_entity, strict=True)):
            expression = " OR ".join(f'"{form}"' for form in unique_forms[:6])
            tsquery = func.websearch_to_tsquery("simple", expression or entity)
            rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
            stmt = (
                select(Chunk, Document, rank)
                .join(Document, Document.id == Chunk.document_id)
                .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
                .order_by(desc(rank), Document.id, Chunk.ordinal)
                .limit(max(100, scan_top_k))
            )
            rows = self.db.execute(stmt).all()
            sql_queries += 1

            entity_forms = [form.casefold() for form in unique_forms]
            sibling_forms = [
                [form.casefold() for form in sibling]
                for sibling_index, sibling in enumerate(forms_by_entity)
                if sibling_index != entity_index
            ]
            best_by_document: dict[UUID, tuple[float, float, Candidate]] = {}
            for chunk, document, rank_value in rows:
                candidate = self._candidate(chunk, document)
                candidate.sources.add("entity_coverage")
                candidate.sources.add(f"entity_coverage:{entity_index}")
                haystack = retrieval_text(candidate).casefold()
                section = " / ".join(candidate.section_path or []).casefold()
                title = f"{candidate.document_title} {candidate.filename}".casefold()

                exact_entity = any(form and form in haystack for form in entity_forms)
                attribute_hits = sum(1 for term in attribute_terms if term and term in haystack)
                structural_hits = sum(
                    1 for term in structural_terms if term in section or term in haystack[:1400]
                )
                sibling_hits = sum(
                    1
                    for sibling in sibling_forms
                    if any(form and form in haystack for form in sibling)
                )
                title_hit = any(form and form in title for form in entity_forms)
                table_bonus = 10.0 if candidate.content_kind == "table" else 0.0
                exact_bonus = 10.0 if exact_entity else 0.0
                compact_table_bonus = (
                    2.0
                    if candidate.content_kind == "table" and len(candidate.text or "") <= 1800
                    else 0.0
                )
                # Sibling co-location is deliberately strong for coordinated list queries:
                # one structured source that contains several requested categories is often
                # a better list/index than an incidental mention of only one category.
                score = (
                    exact_bonus
                    + table_bonus
                    + compact_table_bonus
                    + min(8.0, attribute_hits * 2.0)
                    + min(6.0, structural_hits * 1.5)
                    + min(12.0, sibling_hits * 6.0)
                    + (2.0 if title_hit else 0.0)
                )
                rank_float = float(rank_value or 0.0)
                current = best_by_document.get(candidate.document_id)
                value = (score, rank_float, candidate)
                if current is None or (score, rank_float, -candidate.ordinal) > (
                    current[0], current[1], -current[2].ordinal
                ):
                    best_by_document[candidate.document_id] = value

            ranked = sorted(
                best_by_document.values(),
                key=lambda item: (item[0], item[1], -item[2].ordinal),
                reverse=True,
            )

            # Protect only candidates that look like structured/attribute-bearing evidence,
            # not one arbitrary mention from every matching document. The 0.6.1 version
            # protected up to sixteen documents even when the tail was merely incidental,
            # which is how broad SWO/control-matrix material displaced the requested lists.
            best_score = ranked[0][0] if ranked else 0.0
            if "enumeration" in facets:
                quality_floor = 18.0
                relative_window = 5.0
            elif facets.intersection({"location", "contact", "constraint"}):
                quality_floor = 10.0
                relative_window = 7.0
            else:
                quality_floor = 8.0
                relative_window = 8.0

            weak = bool(ranked) and best_score < quality_floor
            if weak:
                # Preserve a small fallback rather than fabricating completeness from many
                # low-signal mentions. Marking it truncated/incomplete keeps generation and
                # verification from making exhaustive claims.
                qualifying = ranked[: min(4, max(1, max_documents))]
            else:
                threshold = max(quality_floor, best_score - relative_window)
                qualifying = [item for item in ranked if item[0] >= threshold]

            selected = [item[2] for item in qualifying[: max(1, max_documents)]]
            missing = not selected
            truncated = weak or missing or len(qualifying) > max_documents
            any_truncated = any_truncated or truncated
            selected_by_entity.append(selected)
            per_entity_stats.append(
                {
                    "term": entity,
                    "documents_discovered": len(ranked),
                    "documents_qualifying": len(qualifying),
                    "documents_selected": len(selected),
                    "best_structural_score": round(best_score, 4),
                    "weak": weak,
                    "missing": missing,
                    "truncated": truncated,
                }
            )

        # Round-robin across categories so the protected evidence budget cannot be consumed
        # by the first category before later categories are represented.
        balanced: list[Candidate] = []
        seen_chunks: set[UUID] = set()
        max_depth = max((len(items) for items in selected_by_entity), default=0)
        for depth in range(max_depth):
            for items in selected_by_entity:
                if depth >= len(items):
                    continue
                candidate = items[depth]
                if candidate.chunk_id in seen_chunks:
                    continue
                balanced.append(candidate)
                seen_chunks.add(candidate.chunk_id)

        return balanced, {
            "entities": per_entity_stats,
            "truncated": any_truncated,
            "sql_queries": sql_queries,
            "documents_selected": len({candidate.document_id for candidate in balanced}),
            "candidates_selected": len(balanced),
        }

    def _role_alias_source_candidates(
        self,
        user: User,
        document_ids: list[UUID] | None,
        top_k: int,
    ) -> list[Candidate]:
        """Scan structural role headings without assuming a particular acronym expansion."""

        tsquery = func.to_tsquery(
            "simple",
            "responsibility | responsibilities | duty | duties | function | functions | obligation | obligations",
        )
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(max(100, top_k))
        )
        return [self._candidate(row[0], row[1]) for row in self.db.execute(stmt).all()]

    def _acl_corpus_revision(
        self, user: User, document_ids: list[UUID] | None
    ) -> tuple[int, str]:
        """Cheap ACL-aware revision fingerprint for safe metadata caching.

        The document table is small relative to chunks. Including the user-specific ACL
        clause, explicit document scope, count and latest update prevents alias metadata
        from crossing authorization/corpus revisions.
        """
        stmt = select(func.count(Document.id), func.max(Document.updated_at)).where(
            *self._base_filters(user, document_ids)
        )
        count, latest = self.db.execute(stmt).one()
        return int(count or 0), latest.isoformat() if latest else ""

    def _resolve_role_aliases(
        self,
        subject: str,
        user: User,
        document_ids: list[UUID] | None,
    ) -> tuple[list[str], dict[str, int | bool]]:
        """Resolve a short role identifier from corpus evidence, never from a hard-coded map."""

        if not is_short_role_identifier(subject):
            return [], {"identifier": False, "lookup_candidates": 0, "structural_candidates": 0}

        lookup = self._lookup_candidates(
            subject,
            user,
            document_ids,
            min(self.settings.lookup_scan_top_k, self.settings.role_alias_scan_top_k),
        )
        lookup_texts: list[str] = []
        for candidate in lookup:
            lookup_texts.append(candidate.contextual_text or candidate.text)
            if candidate.section_path:
                lookup_texts.append(" / ".join(candidate.section_path))
        aliases = extract_role_aliases(subject, lookup_texts)[: self.settings.role_coverage_max_aliases]
        if aliases:
            return aliases, {
                "identifier": True,
                "lookup_candidates": len(lookup),
                "structural_candidates": 0,
                "fallback_used": False,
            }

        # Only if subject-specific evidence cannot resolve the identifier do we inspect
        # generic role headings. The fallback is deliberately bounded: the 0.4.x path
        # scanned 1,000+ responsibility chunks on every short-role query.
        structural = self._role_alias_source_candidates(
            user, document_ids, self.settings.role_alias_fallback_scan_top_k
        )
        structural_texts: list[str] = []
        for candidate in structural:
            structural_texts.append(candidate.contextual_text or candidate.text)
            if candidate.section_path:
                structural_texts.append(" / ".join(candidate.section_path))
        aliases = extract_role_aliases(subject, structural_texts)[: self.settings.role_coverage_max_aliases]
        return aliases, {
            "identifier": True,
            "lookup_candidates": len(lookup),
            "structural_candidates": len(structural),
            "fallback_used": True,
        }

    def _role_coverage_candidates(
        self,
        subject: str,
        aliases: list[str],
        user: User,
        document_ids: list[UUID] | None,
    ) -> tuple[list[Candidate], dict[str, int]]:
        """Discover section-diverse role duties with one structural FTS query.

        The 0.4.x implementation executed exact + lexical SQL separately for as many as
        sixteen role formulations. On a large corpus that created dozens of database
        round-trips before reranking. Here we compile the evidence-derived subject/aliases
        into one PostgreSQL tsquery and require a generic responsibility/obligation cue.

        This remains corpus-generic: no role, acronym, document or manual name is hard-coded.
        Multiple sections from the same governing document are still deliberately preserved.
        """

        names: list[str] = []
        seen_names: set[str] = set()
        for raw_name in [*aliases, subject]:
            cleaned_name = re.sub(r"\s+", " ", raw_name).strip()
            if not cleaned_name:
                continue
            key = cleaned_name.casefold()
            if key in seen_names:
                continue
            names.append(cleaned_name)
            seen_names.add(key)

        name_expressions: list[str] = []
        for name in names:
            # to_tsquery syntax is assembled only from alphanumeric lexemes extracted
            # locally, not raw user SQL. Terms inside a role name are ANDed; supported
            # aliases are ORed so both full names and short identifiers can match.
            lexemes = re.findall(r"[A-Za-z0-9]+", name.casefold())[:8]
            if lexemes:
                name_expressions.append("(" + " & ".join(lexemes) + ")")

        if not name_expressions:
            return [], {
                "sections_discovered": 0,
                "documents_discovered": 0,
                "sections_selected": 0,
                "documents_selected": 0,
                "sql_queries": 0,
            }

        role_terms = (
            "responsibility | responsibilities | duty | duties | function | functions | "
            "obligation | obligations | role | roles | shall | must"
        )
        tsquery_text = f"({' | '.join(name_expressions)}) & ({role_terms})"
        tsquery = func.to_tsquery("simple", tsquery_text)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        scan_limit = max(
            self.settings.role_coverage_scan_top_k,
            self.settings.role_coverage_max_sections * 8,
        )
        stmt = (
            select(Chunk, Document, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._base_filters(user, document_ids), Chunk.search_vector.op("@@")(tsquery))
            .order_by(desc(rank), Document.id, Chunk.ordinal)
            .limit(scan_limit)
        )
        rows = self.db.execute(stmt).all()
        pool: dict[UUID, Candidate] = {}
        scores: dict[UUID, int] = {}
        for rank_index, (chunk, document, _rank) in enumerate(rows):
            candidate = self._candidate(chunk, document)
            candidate.sources.add("role_structural")
            pool[candidate.chunk_id] = candidate
            structural_score = role_candidate_score(names, candidate)
            rank_bonus = max(0, 8 - min(rank_index // 8, 8))
            scores[candidate.chunk_id] = structural_score + rank_bonus

        ranked = sorted(
            pool.values(),
            key=lambda item: (scores.get(item.chunk_id, 0), -item.ordinal),
            reverse=True,
        )

        # De-duplicate adjacent chunks from the same structural section before applying
        # document/section caps.
        unique_sections: list[Candidate] = []
        seen_sections: set[tuple[str, str]] = set()
        for candidate in ranked:
            key = role_section_key(candidate)
            if key in seen_sections:
                continue
            candidate.sources.add("role_coverage")
            unique_sections.append(candidate)
            seen_sections.add(key)

        discovered_documents = {candidate.document_id for candidate in unique_sections}
        selected: list[Candidate] = []
        selected_sections: set[tuple[str, str]] = set()
        selected_documents: set[UUID] = set()
        per_document: defaultdict[UUID, int] = defaultdict(int)
        max_documents = max(1, self.settings.role_coverage_max_documents)
        per_doc_limit = max(1, self.settings.role_coverage_sections_per_document)
        max_sections = max(1, self.settings.role_coverage_max_sections)

        def add(candidate: Candidate) -> bool:
            key = role_section_key(candidate)
            if key in selected_sections:
                return False
            if candidate.document_id not in selected_documents and len(selected_documents) >= max_documents:
                return False
            if per_document[candidate.document_id] >= per_doc_limit:
                return False
            selected.append(candidate)
            selected_sections.add(key)
            selected_documents.add(candidate.document_id)
            per_document[candidate.document_id] += 1
            return True

        # Canonical role headings/clauses first.
        for candidate in unique_sections:
            if scores.get(candidate.chunk_id, 0) < 30:
                continue
            add(candidate)
            if len(selected) >= max_sections:
                break

        # Then guarantee document breadth before filling additional sections from strong
        # governing/source documents.
        if len(selected) < max_sections:
            for candidate in unique_sections:
                if candidate.document_id in selected_documents:
                    continue
                add(candidate)
                if len(selected) >= max_sections:
                    break

        if len(selected) < max_sections:
            for candidate in unique_sections:
                add(candidate)
                if len(selected) >= max_sections:
                    break

        return selected, {
            "sections_discovered": len(unique_sections),
            "documents_discovered": len(discovered_documents),
            "sections_selected": len(selected),
            "documents_selected": len(selected_documents),
            "sql_queries": 1,
        }

    def resolve_source_policy(
        self, user: User, document_ids: list[UUID] | None
    ) -> SourcePolicy:
        """Describe request scope for tracing without introducing a retrieval gate.

        Explicit document selection remains the only hard scope. Administrator-configured
        governing-source patterns are reported for observability, while actual governing
        evidence is retrieved additively by :meth:`_governing_document_ids`.
        """

        del user  # Scope authorization is enforced inside every retrieval lane.
        if document_ids:
            return SourcePolicy(
                priority_document_ids=list(document_ids),
                priority_patterns=[],
                hard_scope=True,
                reason="explicit_document_scope",
            )
        return SourcePolicy(
            priority_patterns=self.settings.governing_reference_pattern_list,
            hard_scope=False,
            reason="governing_reference_additive",
        )

    def _governing_document_ids(
        self, user: User, document_ids: list[UUID] | None
    ) -> list[UUID]:
        """Resolve mandatory governing-reference sources under the same ACL/scope.

        Governing references are an always-considered evidence lane.  They never become a
        gate and never suppress the global corpus search.  Patterns remain administrator
        configuration rather than domain-specific application logic.
        """

        if not self.settings.governing_reference_enabled:
            return []
        patterns = self.settings.governing_reference_pattern_list
        stmt = select(
            Document.id, Document.title, Document.original_filename, Document.source_role
        ).where(*self._base_filters(user, document_ids))
        result: list[UUID] = []
        for document_id, title, filename, source_role in self.db.execute(stmt).all():
            explicit_role = str(source_role or "").casefold() in {
                "governing_reference",
                "governing_rule",
            }
            configured_match = bool(
                patterns
                and any(source_name_matches(title, filename, pattern) for pattern in patterns)
            )
            if explicit_role or configured_match:
                result.append(document_id)
        return result

    # Compatibility alias for code/tests that still use the 0.9 terminology.
    def _overview_document_ids(
        self, user: User, document_ids: list[UUID] | None
    ) -> list[UUID]:
        return self._governing_document_ids(user, document_ids)

    @staticmethod
    def _without_documents(candidates: list[Candidate], excluded: set[UUID]) -> list[Candidate]:
        if not excluded:
            return candidates
        return [candidate for candidate in candidates if candidate.document_id not in excluded]

    def _section_expansion(
        self,
        anchors: list[Candidate],
        user: User,
        document_ids: list[UUID] | None,
    ) -> list[Candidate]:
        if not self.settings.hierarchical_retrieval_enabled:
            return []
        expanded: list[Candidate] = []
        seen: set[UUID] = set()
        radius = max(1, self.settings.hierarchical_window)
        per_anchor = max(1, self.settings.hierarchical_max_chunks_per_anchor)
        for anchor in anchors:
            low = max(0, anchor.ordinal - radius)
            high = anchor.ordinal + radius
            stmt = (
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    *self._base_filters(user, document_ids),
                    Chunk.document_id == anchor.document_id,
                    Chunk.ordinal.between(low, high),
                )
                .order_by(Chunk.ordinal)
            )
            local = []
            for chunk, document in self.db.execute(stmt).all():
                candidate = self._candidate(chunk, document)
                if not same_logical_section(anchor, candidate):
                    continue
                candidate.sources.add("section_context")
                candidate.evidence_lane = anchor.evidence_lane
                # Neighbouring chunks inherit the anchor's goal attribution so the semantic
                # evidence gate may use continuation text from the same logical section.
                for source in anchor.sources:
                    if source.startswith("goal:") or source == "priority_source":
                        candidate.sources.add(source)
                candidate.goal_rerank_scores.update(anchor.goal_rerank_scores)
                candidate.rerank_score = max(0.0, anchor.rerank_score - 0.005)
                local.append(candidate)
            # Preserve logical order, but keep the expansion bounded.
            if len(local) > per_anchor:
                local = sorted(local, key=lambda c: abs(c.ordinal - anchor.ordinal))[:per_anchor]
                local.sort(key=lambda c: c.ordinal)
            for candidate in local:
                if candidate.chunk_id in seen or candidate.chunk_id == anchor.chunk_id:
                    continue
                expanded.append(candidate)
                seen.add(candidate.chunk_id)
        return expanded

    def _limits(self, profile: str, lookup_term: str | None) -> dict[str, int]:
        if profile == "priority_probe":
            limits = {
                "dense": self.settings.priority_probe_dense_top_k,
                "lexical": self.settings.priority_probe_lexical_top_k,
                "exact": self.settings.priority_probe_exact_top_k,
                "fused": self.settings.priority_probe_fused_top_k,
                "rerank": self.settings.priority_probe_rerank_top_k,
                "evidence": self.settings.priority_probe_evidence_k,
            }
        elif profile == "priority":
            limits = {
                "dense": self.settings.priority_dense_top_k,
                "lexical": self.settings.priority_lexical_top_k,
                "exact": self.settings.priority_exact_top_k,
                "fused": self.settings.priority_fused_top_k,
                "rerank": self.settings.priority_rerank_top_k,
                "evidence": self.settings.priority_evidence_k,
            }
        elif profile == "realtime":
            limits = {
                "dense": 6,
                "lexical": 8,
                "exact": 8,
                "fused": 10,
                "rerank": 6,
                "evidence": 5,
            }
        elif profile == "qa_fast":
            limits = {
                "dense": 10,
                "lexical": 16,
                "exact": 16,
                "fused": 16,
                "rerank": 8,
                "evidence": 8,
            }
        elif profile == "qa_focused":
            limits = {
                "dense": 16,
                "lexical": 24,
                "exact": 24,
                "fused": 24,
                "rerank": 12,
                "evidence": 12,
            }
        elif profile == "qa_research":
            limits = {
                "dense": 24,
                "lexical": 36,
                "exact": 36,
                "fused": 32,
                "rerank": 18,
                "evidence": 18,
            }
        elif profile == "direct":
            limits = {
                "dense": self.settings.direct_dense_top_k,
                "lexical": self.settings.direct_lexical_top_k,
                "exact": self.settings.direct_exact_top_k,
                "fused": self.settings.direct_fused_top_k,
                "rerank": self.settings.direct_rerank_top_k,
                "evidence": self.settings.direct_evidence_k,
            }
        else:
            limits = {
                "dense": self.settings.dense_top_k,
                "lexical": self.settings.lexical_top_k,
                "exact": self.settings.exact_top_k,
                "fused": self.settings.fused_top_k,
                "rerank": self.settings.rerank_top_k,
                "evidence": self.settings.answer_evidence_k,
            }
        if lookup_term and profile not in {"realtime", "qa_fast", "qa_focused", "qa_research"}:
            # The large lookup budget is for terse acronyms/identifiers with genuinely
            # ambiguous expansions. Multi-word entities already carry strong lexical
            # specificity and should not force a much larger CPU reranker pool.
            lookup_tokens = re.findall(r"[A-Za-z0-9_./-]+", lookup_term)
            if len(lookup_tokens) == 1 and len(lookup_term) <= 16:
                limits["fused"] = max(limits["fused"], self.settings.lookup_rerank_top_k * 2)
                limits["rerank"] = max(limits["rerank"], self.settings.lookup_rerank_top_k)
                limits["evidence"] = max(limits["evidence"], self.settings.lookup_evidence_k)
                limits["exact"] = max(limits["exact"], self.settings.lookup_scan_top_k // 2)
            else:
                limits["evidence"] = max(limits["evidence"], min(12, self.settings.lookup_evidence_k))
        return limits

    def retrieve(
        self,
        question: str,
        user: User,
        document_ids: list[UUID] | None = None,
        expansions: list[str] | None = None,
        *,
        profile: str = "research",
        query_plan: QueryPlan | None = None,
        evidence_plan: EvidencePlan | None = None,
        goal_ids: set[str] | None = None,
        goal_expansions: dict[str, list[str]] | None = None,
        include_base_query: bool = True,
        source_stage: str = "all",
        boost_document_ids: list[UUID] | None = None,
        request_id: str | None = None,
        progress: ProgressFn | None = None,
        progress_points: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[Evidence], dict]:
        started = time.perf_counter()
        query_plan = query_plan or build_query_plan(question)
        lookup_term = query_plan.lookup_term if include_base_query else None
        coverage_descriptor = query_plan.coverage_kind or ""
        coverage_profiles = {"research", "priority", "priority_probe", "qa_research", "qa_focused", "qa_fast"}
        procedure_sensitive = include_base_query and profile in coverage_profiles and "procedure" in coverage_descriptor
        role_sensitive = include_base_query and profile in coverage_profiles and "role" in coverage_descriptor
        entity_sensitive = include_base_query and profile in coverage_profiles and "entity_attribute" in coverage_descriptor
        role_subject = query_plan.role_subject if role_sensitive else None
        role_aliases: list[str] = []
        role_resolution: dict[str, int | bool] = {
            "identifier": False,
            "lookup_candidates": 0,
            "structural_candidates": 0,
        }
        role_alias_started = time.perf_counter()
        if role_sensitive and role_subject:
            corpus_revision = self._acl_corpus_revision(user, document_ids)
            cache_key = (
                str(user.id),
                str(user.role),
                str(user.department or ""),
                tuple(sorted(str(item) for item in (document_ids or []))),
                role_subject.casefold(),
                corpus_revision,
            )
            cache = RetrievalEngine._role_alias_cache
            cached = cache.get(cache_key) if cache else None
            if cached is not None:
                role_aliases, role_resolution = cached
                role_resolution["cache_hit"] = True
            else:
                role_aliases, role_resolution = self._resolve_role_aliases(
                    role_subject, user, document_ids
                )
                role_resolution["cache_hit"] = False
                if cache:
                    cache.set(cache_key, (role_aliases, role_resolution))
            role_resolution["corpus_revision"] = f"{corpus_revision[0]}:{corpus_revision[1]}"
            role_resolution["total_ms"] = int((time.perf_counter() - role_alias_started) * 1000)

        limits = self._limits(profile, lookup_term)

        # Structural definition resolution is an early retrieval lane, not merely a reranker
        # hint.  When an explicit definition is present, return it before embeddings, broad
        # candidate fusion, or the CPU cross-encoder are invoked.  Explicit Research mode
        # still uses the full pipeline so users can request broader context intentionally.
        if (
            include_base_query
            and profile != "research"
            and query_plan.intent == "definition"
            and query_plan.lookup_term
        ):
            definition_started = time.perf_counter()
            structural_defs = self._definition_candidates(
                query_plan.lookup_term,
                user,
                document_ids,
                min(8, max(4, limits["evidence"])),
                query_plan.source_scope,
            )
            top_definition_score = (
                float(definition_score(query_plan.lookup_term, structural_defs[0]))
                if structural_defs
                else 0.0
            )
            if structural_defs and top_definition_score >= 16.0:
                active_goals = [
                    goal
                    for goal in (evidence_plan.goals if evidence_plan is not None else [])
                    if goal_ids is None or goal.id in goal_ids
                ]
                definition_goals = [goal for goal in active_goals if goal.kind == "definition"]
                for candidate in structural_defs:
                    candidate.sources.update({"structural_definition", "definition_structural", "lookup"})
                    candidate.final_retrieval_score = min(
                        1.0, float(definition_score(query_plan.lookup_term, candidate)) / 25.0
                    )
                    candidate.rank_method = "structural_definition"
                    for goal in definition_goals:
                        candidate.sources.update({f"goal:{goal.id}", f"goal:{goal.id}:lookup"})

                evidence = self._build_evidence(
                    structural_defs,
                    user,
                    document_ids,
                    limit=min(limits["evidence"], 6),
                    protected=structural_defs,
                    use_neighbors=True,
                )
                goal_stats: dict[str, dict] = {}
                for goal in active_goals:
                    attributed = [
                        item for item in evidence if f"goal:{goal.id}" in item.candidate.sources
                    ]
                    lexical_like = [
                        item
                        for item in attributed
                        if f"goal:{goal.id}:lookup" in item.candidate.sources
                    ]
                    goal_stats[goal.id] = {
                        "kind": goal.kind,
                        "required": goal.required,
                        "question": goal.question,
                        "entity_terms": goal.entity_terms,
                        "evidence_count": len(attributed),
                        "lexical_like_evidence_count": len(lexical_like),
                        "max_rerank_score": 0.0,
                        "evidence_ids": [item.evidence_id for item in attributed],
                        "candidate_count": len(structural_defs),
                        "source_counts": {"lookup": len(attributed)},
                    }
                required_goals = [goal for goal in active_goals if goal.required]
                goal_complete = all(
                    int(goal_stats.get(goal.id, {}).get("evidence_count", 0)) > 0
                    for goal in required_goals
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                definition_ms = int((time.perf_counter() - definition_started) * 1000)
                trace = {
                    "queries": [query_plan.lookup_term],
                    "query_specs": [
                        {
                            "text": query_plan.lookup_term,
                            "goal_ids": [goal.id for goal in definition_goals],
                            "goal_kinds": ["definition"] if definition_goals else [],
                            "weight": 1.0,
                            "origins": ["structural_definition"],
                        }
                    ],
                    "profile": profile,
                    "source_stage": source_stage,
                    "compositional": False,
                    "evidence_plan": evidence_plan.as_dict() if evidence_plan is not None else None,
                    "goal_ids_requested": sorted(goal_ids) if goal_ids else None,
                    "goal_stats": goal_stats,
                    "goal_retrieval_complete": goal_complete,
                    "goal_source_counts": {
                        goal.id: {"lookup": int(goal_stats.get(goal.id, {}).get("evidence_count", 0))}
                        for goal in active_goals
                    },
                    "lookup_term": query_plan.lookup_term,
                    "entity_term": query_plan.entity_term,
                    "entity_terms": query_plan.entity_terms,
                    "query_facets": query_plan.facets,
                    "attribute_terms": query_plan.attribute_terms,
                    "lexical_queries": query_plan.lexical_queries,
                    "exact_terms": query_plan.exact_terms,
                    "source_counts": {"structural_definition": len(structural_defs)},
                    "coverage_sensitive": False,
                    "coverage_kind": query_plan.coverage_kind,
                    "coverage_complete": True,
                    "top_rerank_score": 0.0,
                    "definition_fast_path_satisfied": True,
                    "definition_scores": {
                        str(candidate.chunk_id): float(
                            definition_score(query_plan.lookup_term, candidate)
                        )
                        for candidate in structural_defs
                    },
                    "definition_source_scope": query_plan.source_scope,
                    "rerank_details": {
                        "mode": "structural_definition_early_exit",
                        "cross_encoder_called": False,
                        "top_definition_score": top_definition_score,
                        "source_scope": query_plan.source_scope,
                    },
                    "timings_ms": {
                        "definition_structural_search": definition_ms,
                        "rerank": 0,
                        "rerank_queue_wait": 0,
                        "rerank_execution": 0,
                    },
                    "total_ms": elapsed_ms,
                    "evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "chunk_id": str(item.candidate.chunk_id),
                            "document_id": str(item.candidate.document_id),
                            "document_title": item.candidate.document_title,
                            "family_key": item.candidate.family_key,
                            "page": item.candidate.page_from,
                            "section": (
                                item.candidate.section_path[-1]
                                if item.candidate.section_path
                                else None
                            ),
                            "rerank_score": item.candidate.rerank_score,
                            "final_retrieval_score": item.candidate.final_retrieval_score,
                            "rank_method": item.candidate.rank_method,
                            "sources": sorted(item.candidate.sources),
                        }
                        for item in evidence
                    ],
                }
                return evidence, trace

        bare_identifier = bool(
            include_base_query
            and profile != "research"
            and query_plan.lookup_term
            and re.fullmatch(
                r"[A-Za-z]{1,5}[\s_-]*0*\d{1,5}[A-Za-z]?",
                question.strip().strip("?.! :"),
                re.IGNORECASE,
            )
        )
        if bare_identifier:
            identifier_started = time.perf_counter()
            identifier_candidates = self._identifier_candidates(
                query_plan.lookup_term,
                user,
                document_ids,
                min(10, max(5, limits["evidence"])),
            )
            if identifier_candidates:
                top_candidate = identifier_candidates[0]
                title = f"{top_candidate.document_title} {top_candidate.filename}".casefold()
                compact_term = re.sub(r"[^a-z0-9]", "", query_plan.lookup_term.casefold())
                compact_title = re.sub(r"[^a-z0-9]", "", title)
                top_structural_score = (
                    20.0
                    if compact_term and compact_term in compact_title
                    else 14.0
                    if "index" in title
                    else 0.0
                )
            else:
                top_structural_score = 0.0

            if identifier_candidates and top_structural_score >= 14.0:
                active_goals = [
                    goal
                    for goal in (evidence_plan.goals if evidence_plan is not None else [])
                    if goal_ids is None or goal.id in goal_ids
                ]
                for candidate in identifier_candidates:
                    candidate.sources.update({"structural_identifier", "lookup"})
                    for goal in active_goals:
                        candidate.sources.update({f"goal:{goal.id}", f"goal:{goal.id}:lookup"})

                evidence = self._build_evidence(
                    identifier_candidates,
                    user,
                    document_ids,
                    limit=min(limits["evidence"], 8),
                    protected=identifier_candidates,
                    use_neighbors=True,
                )
                goal_stats: dict[str, dict] = {}
                for goal in active_goals:
                    attributed = [
                        item for item in evidence if f"goal:{goal.id}" in item.candidate.sources
                    ]
                    lexical_like = [
                        item
                        for item in attributed
                        if f"goal:{goal.id}:lookup" in item.candidate.sources
                    ]
                    goal_stats[goal.id] = {
                        "kind": goal.kind,
                        "required": goal.required,
                        "question": goal.question,
                        "entity_terms": goal.entity_terms,
                        "evidence_count": len(attributed),
                        "lexical_like_evidence_count": len(lexical_like),
                        "max_rerank_score": 0.0,
                        "evidence_ids": [item.evidence_id for item in attributed],
                        "candidate_count": len(identifier_candidates),
                        "source_counts": {"lookup": len(attributed)},
                    }
                required_goals = [goal for goal in active_goals if goal.required]
                goal_complete = all(
                    int(goal_stats.get(goal.id, {}).get("evidence_count", 0)) > 0
                    for goal in required_goals
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                identifier_ms = int((time.perf_counter() - identifier_started) * 1000)
                trace = {
                    "queries": [query_plan.lookup_term],
                    "query_specs": [
                        {
                            "text": query_plan.lookup_term,
                            "goal_ids": [goal.id for goal in active_goals],
                            "goal_kinds": [goal.kind for goal in active_goals],
                            "weight": 1.0,
                            "origins": ["structural_identifier"],
                        }
                    ],
                    "profile": profile,
                    "source_stage": source_stage,
                    "compositional": False,
                    "evidence_plan": evidence_plan.as_dict() if evidence_plan is not None else None,
                    "goal_ids_requested": sorted(goal_ids) if goal_ids else None,
                    "goal_stats": goal_stats,
                    "goal_retrieval_complete": goal_complete,
                    "goal_source_counts": {
                        goal.id: {"lookup": int(goal_stats.get(goal.id, {}).get("evidence_count", 0))}
                        for goal in active_goals
                    },
                    "lookup_term": query_plan.lookup_term,
                    "entity_term": query_plan.entity_term,
                    "entity_terms": query_plan.entity_terms,
                    "query_facets": query_plan.facets,
                    "attribute_terms": query_plan.attribute_terms,
                    "lexical_queries": query_plan.lexical_queries,
                    "exact_terms": query_plan.exact_terms,
                    "source_counts": {"structural_identifier": len(identifier_candidates)},
                    "coverage_sensitive": False,
                    "coverage_kind": query_plan.coverage_kind,
                    "coverage_complete": True,
                    "top_rerank_score": 0.0,
                    "definition_fast_path_satisfied": False,
                    "identifier_fast_path_satisfied": True,
                    "rerank_details": {
                        "mode": "structural_identifier_early_exit",
                        "cross_encoder_called": False,
                        "top_structural_score": top_structural_score,
                    },
                    "timings_ms": {
                        "identifier_structural_search": identifier_ms,
                        "rerank": 0,
                        "rerank_queue_wait": 0,
                        "rerank_execution": 0,
                    },
                    "total_ms": elapsed_ms,
                    "evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "chunk_id": str(item.candidate.chunk_id),
                            "document_id": str(item.candidate.document_id),
                            "document_title": item.candidate.document_title,
                            "family_key": item.candidate.family_key,
                            "page": item.candidate.page_from,
                            "section": (
                                item.candidate.section_path[-1]
                                if item.candidate.section_path
                                else None
                            ),
                            "rerank_score": item.candidate.rerank_score,
                            "sources": sorted(item.candidate.sources),
                        }
                        for item in evidence
                    ],
                }
                return evidence, trace

        if role_sensitive:
            # Structural role discovery protects recall, so the CPU cross-encoder does not
            # need to score the full generic Research pool. This is deliberately scoped to
            # role-coverage questions and leaves ordinary Research behavior unchanged.
            limits["fused"] = min(
                limits["fused"], max(16, self.settings.role_coverage_rerank_pool)
            )
            limits["rerank"] = min(
                limits["fused"],
                max(limits["rerank"], self.settings.role_coverage_rerank_top_k),
            )

        if entity_sensitive and not procedure_sensitive and not role_sensitive:
            # Entity-attribute coverage is protected separately from global reranking. Keep
            # the expensive CPU cross-encoder focused on the strongest semantic candidates.
            limits["fused"] = min(limits["fused"], 28)
            limits["rerank"] = min(limits["rerank"], 16)

        if evidence_plan is not None and evidence_plan.requires_decomposition:
            # Atomic goal retrieval already preserves branch recall. A smaller global
            # cross-encoder pool prevents compositional questions from multiplying CPU cost.
            limits["fused"] = min(limits["fused"], max(12, self.settings.compositional_rerank_pool))
            limits["rerank"] = min(
                limits["fused"],
                max(8, self.settings.compositional_rerank_top_k),
            )

        # 0.7.0 builds retrieval formulations around atomic evidence goals first.
        # The whole natural-language question remains valuable for dense retrieval, but it
        # must not crowd out independent lookups such as A / B / C or relationship goals.
        max_query_count = (
            3
            if profile in {"realtime", "qa_fast"}
            else 4
            if profile == "qa_focused"
            else 6
            if profile == "qa_research"
            else self.settings.priority_probe_max_queries
            if profile == "priority_probe"
            else self.settings.priority_max_queries
            if source_stage == "priority"
            else (
                self.settings.compositional_max_queries
                if evidence_plan is not None and evidence_plan.requires_decomposition
                else 8
            )
        )
        query_meta_by_key: dict[str, dict] = {}
        raw_queries: list[str] = []

        def add_raw_query(
            value: str,
            *,
            goal_ids_for_query: list[str] | None = None,
            goal_kinds_for_query: list[str] | None = None,
            weight: float = 1.0,
            origin: str = "base",
        ) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "").strip())
            if not cleaned:
                return
            key = cleaned.casefold()
            metadata = query_meta_by_key.setdefault(
                key,
                {"goal_ids": [], "goal_kinds": [], "weight": float(weight), "origins": []},
            )
            for goal_id in goal_ids_for_query or []:
                if goal_id not in metadata["goal_ids"]:
                    metadata["goal_ids"].append(goal_id)
            for kind in goal_kinds_for_query or []:
                if kind not in metadata["goal_kinds"]:
                    metadata["goal_kinds"].append(kind)
            metadata["weight"] = max(float(metadata.get("weight", 1.0)), float(weight))
            if origin not in metadata["origins"]:
                metadata["origins"].append(origin)
            if key not in {item.casefold() for item in raw_queries}:
                raw_queries.append(cleaned)

        if evidence_plan is not None:
            goal_specs = plan_query_specs(
                evidence_plan,
                goal_ids=goal_ids,
                max_queries=max_query_count,
            )

            def add_goal_repair_queries() -> None:
                # A targeted recovery pass exists specifically because the seed formulations
                # were insufficient.  When the caller suppresses the base query, admit the
                # materially new repair probes before seed goal queries so the bounded query
                # budget cannot truncate away the only changed retrieval strategy.
                for recovery_goal_id, recovery_queries in (goal_expansions or {}).items():
                    if goal_ids is not None and recovery_goal_id not in goal_ids:
                        continue
                    goal_kind = next(
                        (goal.kind for goal in evidence_plan.goals if goal.id == recovery_goal_id),
                        "fact",
                    )
                    for recovery_query in recovery_queries[: self.settings.compositional_queries_per_goal]:
                        add_raw_query(
                            recovery_query,
                            goal_ids_for_query=[recovery_goal_id],
                            goal_kinds_for_query=[goal_kind],
                            weight=1.15,
                            origin="goal_recovery",
                        )

            if goal_expansions and not include_base_query:
                add_goal_repair_queries()

            for spec in goal_specs:
                add_raw_query(
                    spec["text"],
                    goal_ids_for_query=list(spec.get("goal_ids") or []),
                    goal_kinds_for_query=list(spec.get("goal_kinds") or []),
                    weight=float(spec.get("weight") or 1.0),
                    origin="evidence_goal",
                )

            if goal_expansions and include_base_query:
                add_goal_repair_queries()

        if include_base_query:
            # Q0 is an architectural invariant: generated/canonical formulations expand
            # retrieval but never replace the employee's exact wording.
            add_raw_query(question, origin="original")
            for value in query_plan.semantic_queries:
                add_raw_query(value, origin="semantic")
            if role_sensitive and role_subject:
                names_for_dense = role_aliases[:2] or [role_subject]
                for name in names_for_dense:
                    add_raw_query(f"{name} duties responsibilities", origin="role")
            # Planner-derived lexical and exact forms are explicit retrieval hypotheses.
            for value in query_plan.lexical_queries[:4]:
                add_raw_query(value, origin="lexical_plan")
            for value in query_plan.exact_terms[:4]:
                add_raw_query(value, origin="exact_plan")
            for value in expansions or []:
                add_raw_query(value, weight=0.90, origin="expansion")
            if query_plan.line or query_plan.rolling_stock:
                scoped = " ".join(
                    value for value in [question, query_plan.line, query_plan.rolling_stock] if value
                )
                add_raw_query(scoped, origin="scope")

        if self.settings.identifier_variant_expansion:
            for raw in list(raw_queries):
                meta = query_meta_by_key.get(raw.casefold(), {})
                variants = technical_identifier_variants(
                    raw, max_variants=self.settings.identifier_variant_max_queries
                )
                variants.extend(lexical_form_variants(raw))
                for variant in variants:
                    add_raw_query(
                        variant,
                        goal_ids_for_query=list(meta.get("goal_ids") or []),
                        goal_kinds_for_query=list(meta.get("goal_kinds") or []),
                        weight=float(meta.get("weight") or 1.0),
                        origin="identifier_variant",
                    )

        queries: list[str] = []
        query_metadata: list[dict] = []
        seen_queries: set[str] = set()
        for query in raw_queries:
            cleaned_query = query.strip()
            if not cleaned_query:
                continue
            key = cleaned_query.casefold()
            if key in seen_queries:
                continue
            queries.append(cleaned_query)
            query_metadata.append(query_meta_by_key.get(key, {"goal_ids": [], "goal_kinds": [], "weight": 1.0, "origins": ["base"]}))
            seen_queries.add(key)
            if len(queries) >= max_query_count:
                break

        if not queries:
            queries = [question]
            query_metadata = [{"goal_ids": [], "goal_kinds": [], "weight": 1.0, "origins": ["fallback"]}]

        ranked_lists: list[tuple[str, list[UUID], float]] = []
        pool: dict[UUID, Candidate] = {}
        source_counts: dict[str, int] = {}
        timings_ms: dict[str, int] = {}
        protected_lookup: list[Candidate] = []
        protected_coverage: list[Candidate] = []
        protected_role: list[Candidate] = []
        protected_entity: list[Candidate] = []
        protected_goal: list[Candidate] = []
        goal_candidate_ids: dict[str, set[UUID]] = defaultdict(set)
        goal_source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Governing references are a bounded parallel lane only when the question is
        # operational/regulatory. Injecting MRGR into HR/finance/policy lookups polluted top
        # evidence and made unrelated answers look authoritative.
        overview_ids = (
            self._governing_document_ids(user, document_ids)
            if self._governing_lane_relevant(query_plan)
            else []
        )
        overview_id_set = set(overview_ids)
        overview_candidates: list[Candidate] = []
        overview_candidate_ids: set[UUID] = set()
        overview_goal_searched: set[str] = set()
        routed_boost_ids = [
            item for item in (boost_document_ids or []) if item not in overview_id_set
        ] if self.settings.routed_document_boost_enabled else []
        procedure_truncated = False
        role_truncated = False
        entity_truncated = False
        entity_stats: dict = {"entities": [], "truncated": False, "sql_queries": 0}
        role_stats = {
            "sections_discovered": 0,
            "documents_discovered": 0,
            "sections_selected": 0,
            "documents_selected": 0,
            "sql_queries": 0,
        }

        goal_coverage_stats: dict[str, dict] = {}

        if procedure_sensitive:
            coverage_started = time.perf_counter()
            coverage = self._coverage_candidates(
                question,
                user,
                document_ids,
                max_documents=self.settings.coverage_max_documents,
                scan_top_k=self.settings.coverage_scan_top_k,
            )
            timings_ms["coverage_discovery"] = int((time.perf_counter() - coverage_started) * 1000)
            procedure_truncated = len(coverage) > self.settings.coverage_max_documents
            protected_coverage = coverage[: self.settings.coverage_max_documents]
            if protected_coverage:
                required_evidence = len(protected_coverage) * max(
                    1, self.settings.coverage_evidence_per_document
                )
                limits["evidence"] = max(
                    limits["evidence"],
                    min(self.settings.coverage_max_evidence_k, required_evidence),
                )
                limits["fused"] = max(limits["fused"], min(160, len(protected_coverage) * 6))
                limits["rerank"] = max(
                    limits["rerank"], min(48, len(protected_coverage) * 2)
                )
                source_counts["coverage"] = len(protected_coverage)
                ranked_lists.append(
                    ("coverage", [c.chunk_id for c in protected_coverage], 1.65)
                )
                for candidate in protected_coverage:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add("coverage")

        if entity_sensitive:
            entity_coverage_started = time.perf_counter()
            entity_coverage, entity_stats = self._entity_attribute_coverage_candidates(
                query_plan,
                user,
                document_ids,
                max_documents=self.settings.coverage_max_documents,
                scan_top_k=self.settings.coverage_scan_top_k,
            )
            timings_ms["entity_coverage_discovery"] = int((time.perf_counter() - entity_coverage_started) * 1000)
            entity_truncated = bool(entity_stats.get("truncated"))
            if len(entity_coverage) > self.settings.coverage_max_evidence_k:
                entity_truncated = True
            # The helper returns round-robin category ordering, so the evidence cap remains
            # balanced instead of silently dropping later coordinated categories.
            protected_entity = entity_coverage[: self.settings.coverage_max_evidence_k]
            if protected_entity:
                limits["evidence"] = max(
                    limits["evidence"],
                    min(
                        self.settings.coverage_max_evidence_k,
                        len(protected_entity) + max(2, len(query_plan.entity_terms or []) * 2),
                    ),
                )
                source_counts["entity_coverage"] = len(protected_entity)
                ranked_lists.append(
                    ("entity_coverage", [c.chunk_id for c in protected_entity], 1.85)
                )
                for candidate in protected_entity:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add("entity_coverage")

        if role_sensitive and role_subject:
            role_coverage_started = time.perf_counter()
            protected_role, role_stats = self._role_coverage_candidates(
                role_subject, role_aliases, user, document_ids
            )
            timings_ms["role_coverage_discovery"] = int((time.perf_counter() - role_coverage_started) * 1000)
            role_truncated = (
                role_stats["sections_discovered"] > role_stats["sections_selected"]
                or role_stats["documents_discovered"] > role_stats["documents_selected"]
            )
            if len(protected_role) > self.settings.role_coverage_max_evidence_k:
                role_truncated = True
                protected_role = protected_role[: self.settings.role_coverage_max_evidence_k]
            if protected_role:
                limits["evidence"] = max(
                    limits["evidence"],
                    min(
                        self.settings.role_coverage_max_evidence_k,
                        max(16, len(protected_role)),
                    ),
                )
                source_counts["role_coverage"] = len(protected_role)
                ranked_lists.append(
                    ("role_coverage", [c.chunk_id for c in protected_role], 1.9)
                )
                for candidate in protected_role:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add("role_coverage")

        # Generic document-diverse coverage for evidence goals whose semantics explicitly
        # request a set/overview. This complements specialized procedure/role/entity lanes
        # and is deliberately domain-agnostic. It preserves one strong matching passage per
        # accessible document within configured bounds, so "all/list/overview" questions do
        # not collapse back to a global top-k popularity contest.
        if (
            evidence_plan is not None
            and profile in {"research", "priority", "priority_probe"}
            and not (procedure_sensitive or role_sensitive or entity_sensitive)
        ):
            for goal in evidence_plan.goals:
                if goal_ids is not None and goal.id not in goal_ids:
                    continue
                if not goal.required or goal.kind not in {"enumeration", "overview"}:
                    continue
                coverage_query = (goal.search_queries or goal.entity_terms or [goal.question])[0]
                goal_coverage = self._coverage_candidates(
                    coverage_query,
                    user,
                    document_ids,
                    max_documents=self.settings.coverage_max_documents,
                    scan_top_k=self.settings.coverage_scan_top_k,
                )
                truncated = len(goal_coverage) > self.settings.coverage_max_documents
                selected_goal_coverage = goal_coverage[: self.settings.coverage_max_documents]
                goal_coverage_stats[goal.id] = {
                    "documents_selected": len({candidate.document_id for candidate in selected_goal_coverage}),
                    "truncated": truncated,
                    "query": coverage_query,
                }
                if not selected_goal_coverage:
                    continue
                source_key = f"goal:{goal.id}:coverage"
                source_counts[source_key] = len(selected_goal_coverage)
                goal_source_counts[goal.id]["coverage"] += len(selected_goal_coverage)
                ranked_lists.append((source_key, [candidate.chunk_id for candidate in selected_goal_coverage], 1.7))
                limits["evidence"] = max(
                    limits["evidence"],
                    min(self.settings.compositional_max_evidence_k, len(selected_goal_coverage) + 2),
                )
                for candidate in selected_goal_coverage:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add("coverage")
                    existing.sources.add(f"goal:{goal.id}")
                    existing.sources.add(source_key)
                    goal_candidate_ids[goal.id].add(existing.chunk_id)
                    protected_goal.append(existing)

        # Independent definition lookups are essential for compositional requests. A query
        # such as ``A B C`` must not lose A merely because the best B/C passages rank higher
        # globally. The legacy single lookup remains below for ordinary one-term questions.
        if evidence_plan is not None and (evidence_plan.requires_decomposition or goal_ids is not None):
            goal_lookup_started = time.perf_counter()
            for goal_id, terms in lookup_terms_by_goal(evidence_plan, goal_ids=goal_ids).items():
                for term in terms:
                    lookup_tokens = re.findall(r"[A-Za-z0-9_./-]+", term)
                    scan_k = (
                        self.settings.compositional_lookup_scan_top_k
                        if len(lookup_tokens) == 1
                        else min(96, self.settings.compositional_lookup_scan_top_k)
                    )
                    lookup = self._lookup_candidates(term, user, document_ids, scan_k)
                    source_key = f"goal:{goal_id}:lookup"
                    source_counts[source_key] = source_counts.get(source_key, 0) + len(lookup)
                    goal_source_counts[goal_id]["lookup"] += len(lookup)
                    if lookup:
                        ranked_lists.append((source_key, [c.chunk_id for c in lookup], 1.9))
                    strong_definitions: list[Candidate] = []
                    for candidate in lookup:
                        existing = pool.setdefault(candidate.chunk_id, candidate)
                        existing.sources.add("lookup")
                        existing.sources.add(f"goal:{goal_id}")
                        existing.sources.add(f"goal:{goal_id}:lookup")
                        goal_candidate_ids[goal_id].add(existing.chunk_id)
                        if definition_score(term, candidate) >= 4:
                            strong_definitions.append(existing)
                    # Protect several definition-like passages because one acronym can have
                    # several valid meanings. If no definition pattern is found, retain one
                    # document-diverse lookup candidate for the semantic goal audit.
                    protected_goal.extend(
                        strong_definitions[: self.settings.compositional_goal_evidence_per_goal * 2]
                        or [pool[lookup[0].chunk_id]] if lookup else []
                    )
            timings_ms["goal_lookup_search"] = int((time.perf_counter() - goal_lookup_started) * 1000)

        search_pct, fusion_pct, rerank_pct, evidence_pct = progress_points or (
            (18, 30, 42, 52) if profile == "direct" else (32, 44, 56, 66)
        )
        self._emit(
            progress,
            "search",
            "Searching document indexes",
            "Batching semantic query formulations, then running keyword and exact-term retrieval over accessible documents.",
            search_pct,
        )
        embedding_started = time.perf_counter()
        query_vectors, embedding_timing = self.inference.embed_queries(queries)
        timings_ms["query_embedding_total"] = int((time.perf_counter() - embedding_started) * 1000)
        timings_ms["query_embedding_queue_wait"] = embedding_timing.queue_wait_ms
        timings_ms["query_embedding_execution"] = embedding_timing.execution_ms

        search_started = time.perf_counter()
        stage_accumulator = {
            "dense_search": 0.0, "lexical_search": 0.0, "relaxed_lexical_search": 0.0,
            "section_navigation_search": 0.0, "exact_search": 0.0,
            "table_search": 0.0, "lookup_search": 0.0, "routed_boost_search": 0.0
        }
        for query_index, (query, query_vector) in enumerate(zip(queries, query_vectors, strict=True)):
            query_meta = query_metadata[query_index] if query_index < len(query_metadata) else {}
            query_goal_ids = list(query_meta.get("goal_ids") or [])
            positional_weight = 1.0 if query_goal_ids or query_index == 0 else 0.78
            query_weight = positional_weight * float(query_meta.get("weight") or 1.0)

            dense_started = time.perf_counter()
            dense_candidates = self._dense(query_vector, user, document_ids, limits["dense"] + len(overview_ids))
            if overview_ids:
                dense_candidates = self._without_documents(dense_candidates, overview_id_set)[: limits["dense"]]
            stage_accumulator["dense_search"] += time.perf_counter() - dense_started

            lexical_started = time.perf_counter()
            lexical_candidates = self._lexical(query, user, document_ids, limits["lexical"] + len(overview_ids))
            if overview_ids:
                lexical_candidates = self._without_documents(lexical_candidates, overview_id_set)[: limits["lexical"]]
            stage_accumulator["lexical_search"] += time.perf_counter() - lexical_started

            relaxed_started = time.perf_counter()
            relaxed_top_k = (
                4
                if profile == "realtime"
                else self.settings.priority_probe_relaxed_top_k
                if profile == "priority_probe"
                else self.settings.priority_relaxed_lexical_top_k
                if source_stage == "priority"
                else self.settings.relaxed_lexical_top_k
            )
            relaxed_candidates = self._relaxed_lexical(
                query, user, document_ids, relaxed_top_k
            )
            if overview_ids:
                relaxed_candidates = self._without_documents(relaxed_candidates, overview_id_set)
            stage_accumulator["relaxed_lexical_search"] += time.perf_counter() - relaxed_started

            section_started = time.perf_counter()
            section_top_k = (
                4
                if profile == "realtime"
                else self.settings.priority_probe_section_top_k
                if profile == "priority_probe"
                else self.settings.priority_section_top_k
                if source_stage == "priority"
                else self.settings.section_navigation_top_k
            )
            section_candidates = self._section_navigation_candidates(
                query, user, document_ids, section_top_k
            )
            if overview_ids:
                section_candidates = self._without_documents(section_candidates, overview_id_set)
            stage_accumulator["section_navigation_search"] += time.perf_counter() - section_started

            exact_started = time.perf_counter()
            exact_candidates = self._exact(query, user, document_ids, limits["exact"] + len(overview_ids))
            if overview_ids:
                exact_candidates = self._without_documents(exact_candidates, overview_id_set)[: limits["exact"]]
            stage_accumulator["exact_search"] += time.perf_counter() - exact_started

            table_started = time.perf_counter()
            table_allowed = table_retrieval_relevant(
                goal_kinds=query_meta.get("goal_kinds") or [],
                facets=query_plan.facets or [],
                question=query,
            )
            table_candidates = (
                self._table_lexical(query, user, document_ids, self.settings.table_retrieval_top_k)
                if table_allowed
                else []
            )
            if overview_ids:
                table_candidates = self._without_documents(table_candidates, overview_id_set)
            stage_accumulator["table_search"] += time.perf_counter() - table_started

            routed_dense_candidates: list[Candidate] = []
            routed_lexical_candidates: list[Candidate] = []
            if routed_boost_ids and query_index < max(1, self.settings.routed_search_max_queries):
                routed_started = time.perf_counter()
                routed_dense_candidates = self._dense(
                    query_vector, user, routed_boost_ids, self.settings.routed_dense_top_k
                )
                routed_lexical_candidates = self._lexical(
                    query, user, routed_boost_ids, self.settings.routed_lexical_top_k
                )
                stage_accumulator["routed_boost_search"] += time.perf_counter() - routed_started
                for candidate in [*routed_dense_candidates, *routed_lexical_candidates]:
                    candidate.sources.add("routed")

            if profile == "realtime":
                sources = (
                    ("dense", dense_candidates, 0.72),
                    ("lexical", lexical_candidates, 1.45),
                    ("relaxed", relaxed_candidates, min(0.70, self.settings.relaxed_lexical_weight)),
                    ("section", section_candidates, max(0.95, self.settings.section_navigation_weight)),
                    ("exact", exact_candidates, 1.70),
                    ("table", table_candidates, self.settings.table_retrieval_weight),
                    ("routed_dense", routed_dense_candidates, 0.90),
                    ("routed_lexical", routed_lexical_candidates, 1.55),
                )
            elif profile in {"qa_fast", "qa_focused", "qa_research"}:
                # Interactive Q&A uses hybrid fusion as the primary ranker. Exact/lexical
                # and section-heading evidence receive a modest preference over dense-only
                # similarity; this mirrors production hybrid-search practice while avoiding
                # the 50-100s CPU cross-encoder tax on ordinary questions.
                sources = (
                    ("dense", dense_candidates, 0.90),
                    ("lexical", lexical_candidates, 1.30),
                    ("relaxed", relaxed_candidates, min(0.75, self.settings.relaxed_lexical_weight)),
                    ("section", section_candidates, max(1.10, self.settings.section_navigation_weight)),
                    ("exact", exact_candidates, 1.60),
                    ("table", table_candidates, max(1.05, self.settings.table_retrieval_weight)),
                    ("routed_dense", routed_dense_candidates, 1.00),
                    ("routed_lexical", routed_lexical_candidates, 1.35),
                )
            else:
                sources = (
                    ("dense", dense_candidates, 1.0),
                    ("lexical", lexical_candidates, 1.0),
                    ("relaxed", relaxed_candidates, self.settings.relaxed_lexical_weight),
                    ("section", section_candidates, self.settings.section_navigation_weight),
                    ("exact", exact_candidates, 1.25),
                    ("table", table_candidates, self.settings.table_retrieval_weight),
                    ("routed_dense", routed_dense_candidates, self.settings.routed_document_boost),
                    ("routed_lexical", routed_lexical_candidates, self.settings.routed_document_boost),
                )
            for source, candidates, source_weight in sources:
                key = f"q{query_index}:{source}"
                source_counts[key] = len(candidates)
                ranked_lists.append(
                    (key, [c.chunk_id for c in candidates], query_weight * source_weight)
                )
                for candidate in candidates:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add(source)
                    if source.startswith("routed_"):
                        existing.sources.add("routed")
                    if source_stage == "priority":
                        existing.evidence_lane = "priority"
                        existing.sources.add("priority_source")
                    for goal_id in query_goal_ids:
                        existing.sources.add(f"goal:{goal_id}")
                        existing.sources.add(f"goal:{goal_id}:{source}")
                        goal_candidate_ids[goal_id].add(existing.chunk_id)
                        goal_source_counts[goal_id][source] += 1

            # Dedicated generic-context lane. Overview documents are excluded from the
            # ordinary operational pool, so compositional retrieval must search them once
            # per atomic goal rather than only for the first query. Otherwise a valid goal
            # can become impossible to satisfy when its evidence exists only in an overview
            # source. Ordinary one-part queries retain the legacy one-search behavior.
            overview_goal_ids = [
                goal_id for goal_id in query_goal_ids if goal_id not in overview_goal_searched
            ]
            should_search_overview = bool(
                overview_ids
                and (
                    (evidence_plan is not None and evidence_plan.requires_decomposition and overview_goal_ids)
                    or (not (evidence_plan is not None and evidence_plan.requires_decomposition) and query_index == 0)
                )
            )
            if should_search_overview:
                overview_goal_searched.update(overview_goal_ids)
                overview_dense = self._dense(
                    query_vector, user, overview_ids, self.settings.governing_reference_dense_top_k
                )
                overview_lexical = self._lexical(
                    query, user, overview_ids, self.settings.governing_reference_lexical_top_k
                )
                merged_overview: list[Candidate] = []
                seen_overview: set[UUID] = set()
                for candidate in [*overview_lexical, *overview_dense]:
                    if candidate.chunk_id in seen_overview:
                        continue
                    candidate.evidence_lane = "governing"
                    candidate.sources.add("governing")
                    candidate.sources.add("overview")
                    merged_overview.append(candidate)
                    seen_overview.add(candidate.chunk_id)
                selected_overview = merged_overview[: max(
                    self.settings.governing_reference_evidence_k * 3, self.settings.governing_reference_evidence_k
                )]
                if selected_overview:
                    source_key = f"q{query_index}:governing"
                    source_counts[source_key] = len(selected_overview)
                    source_counts["governing"] = source_counts.get("governing", 0) + len(selected_overview)
                    ranked_lists.append(
                        (source_key, [c.chunk_id for c in selected_overview], 1.25)
                    )
                    for candidate in selected_overview:
                        existing = pool.setdefault(candidate.chunk_id, candidate)
                        existing.sources.add("governing")
                        existing.sources.add("overview")
                        existing.sources.add(source_key)
                        existing.evidence_lane = "governing"
                        for goal_id in overview_goal_ids:
                            existing.sources.add(f"goal:{goal_id}")
                            existing.sources.add(f"goal:{goal_id}:governing")
                            goal_candidate_ids[goal_id].add(existing.chunk_id)
                            goal_source_counts[goal_id]["governing"] += 1
                        if existing.chunk_id not in overview_candidate_ids:
                            overview_candidates.append(existing)
                            overview_candidate_ids.add(existing.chunk_id)

            if (
                query_index == 0
                and lookup_term
                and not (evidence_plan is not None and evidence_plan.requires_decomposition)
            ):
                lookup_started = time.perf_counter()
                lookup_tokens = re.findall(r"[A-Za-z0-9_./-]+", lookup_term)
                lookup_scan_top_k = (
                    self.settings.lookup_scan_top_k
                    if len(lookup_tokens) == 1
                    else min(96, self.settings.lookup_scan_top_k)
                )
                lookup = self._lookup_candidates(
                    lookup_term,
                    user,
                    document_ids,
                    lookup_scan_top_k,
                )
                stage_accumulator["lookup_search"] += time.perf_counter() - lookup_started
                source_counts["q0:lookup"] = len(lookup)
                ranked_lists.append(("q0:lookup", [c.chunk_id for c in lookup], 1.8))
                for candidate in lookup:
                    existing = pool.setdefault(candidate.chunk_id, candidate)
                    existing.sources.add("lookup")
                protected_lookup = [
                    c for c in lookup if definition_score(lookup_term, c) >= 4
                ][: max(6, min(10, limits["evidence"] // 2 + 2))]
        timings_ms["candidate_search"] = int((time.perf_counter() - search_started) * 1000)
        for stage_name, seconds in stage_accumulator.items():
            timings_ms[stage_name] = int(seconds * 1000)

        self._emit(
            progress,
            "fusion",
            "Combining retrieval signals",
            f"Fusing {sum(source_counts.values())} candidate hits while preserving exact-term matches.",
            fusion_pct,
        )
        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(ranked_lists, k=self.settings.rrf_k)
        adjusted_scores: dict[UUID, float] = {}
        for chunk_id, (score, _sources) in fused.items():
            bonus = 0.0
            candidate = pool.get(chunk_id)
            if (
                candidate is not None
                and self.settings.table_context_boost_enabled
                and "table" in candidate.sources
            ):
                # The structural table lane is deliberately gated per query.  A chunk that
                # happens to be a table but arrived only through dense/lexical search must not
                # receive an unrelated global table bonus.
                bonus = min(
                    self.settings.table_context_fusion_bonus,
                    table_query_affinity(question, candidate),
                )
            applicability = applicability_score(query_plan, candidate) if candidate is not None else 0.0
            if candidate is not None:
                candidate.applicability_score = applicability
            adjusted_scores[chunk_id] = score + bonus + applicability
        fused_cap = min(limits["fused"], max(8, self.settings.rerank_prefilter_max_candidates))
        globally_ranked_ids = sorted(
            fused, key=lambda chunk_id: adjusted_scores[chunk_id], reverse=True
        )

        # Reserve a bounded number of candidates for each required evidence goal before the
        # global rerank prefilter. Without this, a strong branch can crowd out a weaker but
        # mandatory branch even though retrieval found it (the classic multi-part RAG failure).
        reserved_ids: list[UUID] = []
        if evidence_plan is not None and evidence_plan.requires_decomposition:
            active_required_goals = [
                goal.id
                for goal in evidence_plan.goals
                if goal.required and (goal_ids is None or goal.id in goal_ids)
            ]
            per_goal_ranked: dict[str, list[UUID]] = {}
            for active_goal_id in active_required_goals:
                per_goal_ranked[active_goal_id] = sorted(
                    [chunk_id for chunk_id in goal_candidate_ids.get(active_goal_id, set()) if chunk_id in fused],
                    key=lambda chunk_id: adjusted_scores.get(chunk_id, 0.0),
                    reverse=True,
                )
            reserve_depth = max(1, self.settings.compositional_goal_candidate_reserve)
            if self.settings.goal_local_rerank_enabled and active_required_goals:
                # Reserve enough candidates for a real goal-local cross-encoder pass.  The
                # total remains bounded by fused_cap and is round-robin across goals.
                reserve_depth = max(
                    reserve_depth,
                    min(
                        self.settings.goal_local_rerank_candidates,
                        max(1, fused_cap // len(active_required_goals)),
                    ),
                )
            for depth in range(reserve_depth):
                for active_goal_id in active_required_goals:
                    ranked_for_goal = per_goal_ranked.get(active_goal_id, [])
                    if depth >= len(ranked_for_goal):
                        continue
                    chunk_id = ranked_for_goal[depth]
                    if chunk_id not in reserved_ids:
                        reserved_ids.append(chunk_id)
                    if len(reserved_ids) >= fused_cap:
                        break
                if len(reserved_ids) >= fused_cap:
                    break

        if self.settings.lane_reservation_enabled and len(reserved_ids) < fused_cap:
            lane_quotas = (
                (("exact", 2), ("lexical", 4), ("routed", 2), ("governing", 1), ("dense", 1), ("table", 0))
                if profile == "realtime"
                else (
                    ("table", self.settings.lane_reserve_table),
                    ("exact", self.settings.lane_reserve_exact),
                    ("governing", self.settings.lane_reserve_governing),
                    ("routed", self.settings.lane_reserve_routed),
                    ("lexical", self.settings.lane_reserve_lexical),
                    ("dense", self.settings.lane_reserve_dense),
                )
            )
            reserved_set = set(reserved_ids)
            for lane, quota in lane_quotas:
                if quota <= 0 or len(reserved_ids) >= fused_cap:
                    continue
                taken = 0
                for chunk_id in globally_ranked_ids:
                    if chunk_id in reserved_set:
                        continue
                    candidate = pool.get(chunk_id)
                    if candidate is None or lane not in candidate.sources:
                        continue
                    reserved_ids.append(chunk_id)
                    reserved_set.add(chunk_id)
                    taken += 1
                    if taken >= quota or len(reserved_ids) >= fused_cap:
                        break

        reserved_set = set(reserved_ids)
        ordered_ids = reserved_ids + [
            chunk_id for chunk_id in globally_ranked_ids if chunk_id not in reserved_set
        ]
        ordered_ids = ordered_ids[:fused_cap]
        fused_candidates: list[Candidate] = []
        for chunk_id in ordered_ids:
            candidate = pool[chunk_id]
            candidate.fused_score = adjusted_scores[chunk_id]
            candidate.final_retrieval_score = candidate.fused_score
            candidate.rank_method = "hybrid_fusion"
            _raw_score, fused_sources = fused[chunk_id]
            candidate.sources.update(fused_sources)
            if candidate.content_kind == "table" and table_query_affinity(question, candidate) > 0:
                candidate.sources.add("table_context")
            fused_candidates.append(candidate)
        fused_candidates = deduplicate_candidates(fused_candidates)
        definition_fast_path = False
        definition_scores: dict[UUID, float] = {}
        if query_plan.intent == "definition" and query_plan.lookup_term:
            structural_defs = self._definition_candidates(
                query_plan.lookup_term, user, document_ids, 8, query_plan.source_scope
            )
            definition_goal_ids = [
                goal.id for goal in (evidence_plan.goals if evidence_plan is not None else [])
                if goal.kind == "definition" and (goal_ids is None or goal.id in goal_ids)
            ]
            existing = {candidate.chunk_id for candidate in fused_candidates}
            for candidate in structural_defs:
                for goal_id in definition_goal_ids:
                    candidate.sources.update({f"goal:{goal_id}", f"goal:{goal_id}:lookup"})
                    goal_candidate_ids[goal_id].add(candidate.chunk_id)
                    goal_source_counts[goal_id]["lookup"] += 1
                definition_scores[candidate.chunk_id] = float(definition_score(query_plan.lookup_term, candidate))
            for candidate in structural_defs:
                score = definition_scores.get(candidate.chunk_id, 0.0)
                candidate.final_retrieval_score = max(
                    candidate.final_retrieval_score,
                    min(1.0, score / 25.0),
                )
                candidate.rank_method = "structural_definition"
            fused_candidates = structural_defs + [c for c in fused_candidates if c.chunk_id not in {d.chunk_id for d in structural_defs}]
            definition_fast_path = bool(structural_defs and definition_scores.get(structural_defs[0].chunk_id, 0.0) >= 12.0)
            if definition_fast_path:
                for candidate in structural_defs:
                    candidate.sources.add("structural_definition")
        timings_ms["fusion"] = int((time.perf_counter() - fusion_started) * 1000)

        # The plan records the user's requested coverage even when Direct was explicitly
        # forced. Execution flags below show which deterministic coverage lanes actually ran.
        coverage_sensitive = query_plan.coverage_sensitive
        coverage_kind = query_plan.coverage_kind

        if not fused_candidates:
            return [], {
                "queries": queries,
                "profile": profile,
                "source_stage": source_stage,
                "lookup_term": lookup_term,
                "source_counts": source_counts,
                "fused_candidates": 0,
                "reranked_candidates": 0,
                "top_rerank_score": 0.0,
                "coverage_sensitive": coverage_sensitive,
                "coverage_kind": coverage_kind,
                "coverage_complete": False if (coverage_sensitive or (evidence_plan and evidence_plan.requires_decomposition)) else True,
                "legacy_coverage_complete": False if coverage_sensitive else True,
                "compositional": bool(evidence_plan and evidence_plan.requires_decomposition),
                "evidence_plan": evidence_plan.as_dict() if evidence_plan is not None else None,
                "goal_ids_requested": sorted(goal_ids) if goal_ids else None,
                "goal_stats": {},
                "goal_retrieval_complete": False if evidence_plan and evidence_plan.required_goals else True,
                "query_specs": [
                    {"text": query, **query_metadata[index]}
                    for index, query in enumerate(queries)
                ],
                "entity_attribute_coverage_sensitive": entity_sensitive,
                "query_facets": query_plan.facets,
                "entity_term": query_plan.entity_term,
                "entity_terms": query_plan.entity_terms,
                "entity_coverage_stats": entity_stats,
                "role_coverage_sensitive": role_sensitive,
                "role_subject": role_subject,
                "role_aliases": role_aliases,
                "role_alias_resolution": role_resolution,
                "role_sections_discovered": role_stats["sections_discovered"],
                "role_documents_discovered": role_stats["documents_discovered"],
                "evidence": [],
                "timings_ms": timings_ms,
                "total_ms": int((time.perf_counter() - started) * 1000),
            }

        self._emit(
            progress,
            "rerank",
            "Reranking the strongest passages",
            (
                "Scoring bounded candidate sets against each atomic evidence goal before combining branches."
                if self.settings.goal_local_rerank_enabled
                and evidence_plan is not None
                and (evidence_plan.requires_decomposition or goal_ids is not None)
                else f"Scoring the best {len(fused_candidates)} candidates against the exact question using the local reranker."
            ),
            rerank_pct,
        )
        rerank_started = time.perf_counter()
        use_goal_local_rerank = bool(
            self.settings.goal_local_rerank_enabled
            and evidence_plan is not None
            and (evidence_plan.requires_decomposition or goal_ids is not None)
        )
        final_candidates: list[Candidate] = []
        rerank_details: dict = {}
        rerank_query = "goal-local" if use_goal_local_rerank else question

        if definition_fast_path:
            final_candidates.extend(fused_candidates[: min(6, limits["rerank"])])
            timings_ms["rerank_queue_wait"] = 0
            timings_ms["rerank_execution"] = 0
            rerank_details = {
                "mode": "structural_definition_bypass",
                "cross_encoder_called": False,
                "top_definition_score": definition_scores.get(fused_candidates[0].chunk_id, 0.0),
                "source_scope": query_plan.source_scope,
            }
        elif profile in {"realtime", "qa_fast", "qa_focused", "qa_research"}:
            # Interactive paths use bounded hybrid fusion as the first-pass ranker. Keep
            # rerank_score at zero so downstream confidence never mistakes fusion position
            # for a cross-encoder semantic score.
            source_tag = "realtime_fusion_rank" if profile == "realtime" else "qa_fusion_rank"
            for candidate in fused_candidates[: limits["rerank"]]:
                candidate.rerank_score = 0.0
                candidate.final_retrieval_score = max(candidate.final_retrieval_score, candidate.fused_score)
                candidate.rank_method = "hybrid_fusion"
                candidate.sources.add(source_tag)
                final_candidates.append(candidate)
            timings_ms["rerank_queue_wait"] = 0
            timings_ms["rerank_execution"] = 0
            rerank_details = {
                "mode": "realtime_fusion_bypass" if profile == "realtime" else "qa_fusion_bypass",
                "cross_encoder_called": False,
                "profile": profile,
            }
        elif use_goal_local_rerank:
            active_goals_for_rerank = [
                goal
                for goal in evidence_plan.goals
                if goal_ids is None or goal.id in goal_ids
            ]
            fused_by_id = {candidate.chunk_id: candidate for candidate in fused_candidates}
            per_goal_selected: dict[str, list[Candidate]] = {}
            per_goal_details: dict[str, dict] = {}
            ranked_candidates_by_goal: dict[str, list[Candidate]] = {}

            for goal in active_goals_for_rerank:
                ranked_for_goal = sorted(
                    [
                        fused_by_id[chunk_id]
                        for chunk_id in goal_candidate_ids.get(goal.id, set())
                        if chunk_id in fused_by_id
                    ],
                    key=lambda candidate: (candidate.fused_score, -candidate.ordinal),
                    reverse=True,
                )[: self.settings.goal_local_rerank_candidates]
                ranked_candidates_by_goal[goal.id] = ranked_for_goal
                if not ranked_for_goal:
                    per_goal_selected[goal.id] = []
                    per_goal_details[goal.id] = {"candidate_count": 0, "selected_count": 0}

            nonempty_goals = [
                goal for goal in active_goals_for_rerank if ranked_candidates_by_goal.get(goal.id)
            ]
            total_queue_wait = 0
            total_execution = 0
            batch_details: dict = {}

            if (
                self.settings.batch_goal_rerank_enabled
                and 1 < len(nonempty_goals) <= self.settings.batch_goal_rerank_max_groups
                and sum(len(ranked_candidates_by_goal[goal.id]) for goal in nonempty_goals)
                <= self.settings.batch_goal_rerank_max_pairs
            ):
                groups = [
                    (
                        goal.id,
                        goal.question,
                        [retrieval_text(candidate) for candidate in ranked_candidates_by_goal[goal.id]],
                        min(self.settings.goal_local_rerank_top_k, len(ranked_candidates_by_goal[goal.id])),
                    )
                    for goal in nonempty_goals
                ]
                grouped_results, batch_timing = self.inference.rerank_many(
                    groups, request_id=request_id
                )
                total_queue_wait = batch_timing.queue_wait_ms
                total_execution = batch_timing.execution_ms
                batch_details = batch_timing.details or {}
                for goal in nonempty_goals:
                    ranked_for_goal = ranked_candidates_by_goal[goal.id]
                    group = grouped_results.get(goal.id)
                    results = group.results if group is not None else []
                    selected: list[Candidate] = []
                    for result in results:
                        candidate = ranked_for_goal[result.index]
                        candidate.goal_rerank_scores[goal.id] = result.score
                        candidate.rerank_score = max(candidate.rerank_score, result.score)
                        candidate.final_retrieval_score = max(candidate.final_retrieval_score, result.score)
                        candidate.rank_method = "cross_encoder"
                        candidate.sources.add(f"goal:{goal.id}:reranked")
                        selected.append(candidate)
                    per_goal_selected[goal.id] = selected
                    per_goal_details[goal.id] = {
                        "candidate_count": len(ranked_for_goal),
                        "selected_count": len(selected),
                        "top_score": max((item.rerank_score for item in selected), default=0.0),
                        "batch_rerank": True,
                    }
                rerank_mode = "goal_local_batch"
            else:
                # Backward-compatible fallback for one goal or an oversized batch.
                for goal in nonempty_goals:
                    ranked_for_goal = ranked_candidates_by_goal[goal.id]
                    results, timing = self.inference.rerank(
                        goal.question,
                        [retrieval_text(candidate) for candidate in ranked_for_goal],
                        top_k=min(self.settings.goal_local_rerank_top_k, len(ranked_for_goal)),
                        request_id=request_id,
                    )
                    total_queue_wait += timing.queue_wait_ms
                    total_execution += timing.execution_ms
                    selected: list[Candidate] = []
                    for result in results:
                        candidate = ranked_for_goal[result.index]
                        candidate.goal_rerank_scores[goal.id] = result.score
                        candidate.rerank_score = max(candidate.rerank_score, result.score)
                        candidate.final_retrieval_score = max(candidate.final_retrieval_score, result.score)
                        candidate.rank_method = "cross_encoder"
                        candidate.sources.add(f"goal:{goal.id}:reranked")
                        selected.append(candidate)
                    per_goal_selected[goal.id] = selected
                    per_goal_details[goal.id] = {
                        "candidate_count": len(ranked_for_goal),
                        "selected_count": len(selected),
                        "top_score": max((item.rerank_score for item in selected), default=0.0),
                        "queue_wait_ms": timing.queue_wait_ms,
                        "execution_ms": timing.execution_ms,
                        "batch_rerank": False,
                    }
                rerank_mode = "goal_local"

            # Fan-in is round-robin so every required goal keeps its best evidence.
            ordered_goals = [goal for goal in active_goals_for_rerank if goal.required] + [
                goal for goal in active_goals_for_rerank if not goal.required
            ]
            seen_final: set[UUID] = set()
            max_depth = max((len(per_goal_selected.get(goal.id, [])) for goal in ordered_goals), default=0)
            for depth in range(max_depth):
                for goal in ordered_goals:
                    items = per_goal_selected.get(goal.id, [])
                    if depth >= len(items):
                        continue
                    candidate = items[depth]
                    if candidate.chunk_id in seen_final:
                        continue
                    final_candidates.append(candidate)
                    seen_final.add(candidate.chunk_id)
                    if len(final_candidates) >= limits["rerank"]:
                        break
                if len(final_candidates) >= limits["rerank"]:
                    break

            # Optional context can fill unused capacity, but it never displaces a required
            # branch.  Unscored candidates retain rerank_score=0 and are therefore weaker
            # than goal-local evidence in downstream selection.
            if len(final_candidates) < limits["rerank"]:
                for candidate in fused_candidates:
                    if candidate.chunk_id in seen_final:
                        continue
                    final_candidates.append(candidate)
                    seen_final.add(candidate.chunk_id)
                    if len(final_candidates) >= limits["rerank"]:
                        break

            timings_ms["rerank_queue_wait"] = total_queue_wait
            timings_ms["rerank_execution"] = total_execution
            rerank_details = {
                "mode": rerank_mode,
                "goal_count": len(active_goals_for_rerank),
                "per_goal": per_goal_details,
                "batch": batch_details,
            }
        else:
            rerank_query = question
            if evidence_plan is not None and goal_ids and len(goal_ids) == 1:
                only_goal_id = next(iter(goal_ids))
                goal = next((item for item in evidence_plan.goals if item.id == only_goal_id), None)
                if goal is not None:
                    rerank_query = goal.question
            reranked, rerank_timing = self.inference.rerank(
                rerank_query,
                [retrieval_text(candidate) for candidate in fused_candidates],
                top_k=min(limits["rerank"], len(fused_candidates)),
                request_id=request_id,
            )
            timings_ms["rerank_queue_wait"] = rerank_timing.queue_wait_ms
            timings_ms["rerank_execution"] = rerank_timing.execution_ms
            rerank_details = rerank_timing.details or {}
            if rerank_details.get("tokenization_ms") is not None:
                timings_ms["rerank_tokenization"] = int(rerank_details.get("tokenization_ms") or 0)
            for result in reranked:
                candidate = fused_candidates[result.index]
                candidate.rerank_score = result.score
                candidate.final_retrieval_score = result.score
                candidate.rank_method = "cross_encoder"
                final_candidates.append(candidate)

        timings_ms["rerank"] = int((time.perf_counter() - rerank_started) * 1000)

        if evidence_plan is not None and (evidence_plan.requires_decomposition or goal_ids is not None):
            # Build a balanced protected set after global reranking. This does not declare
            # support; it only guarantees that the downstream semantic audit sees the best
            # available evidence for every atomic requirement.
            final_ids = {candidate.chunk_id for candidate in final_candidates}
            seed_by_goal: dict[str, list[Candidate]] = defaultdict(list)
            for candidate in protected_goal:
                for source in candidate.sources:
                    match = re.fullmatch(r"goal:(g\d+)", source)
                    if match:
                        seed_by_goal[match.group(1)].append(candidate)

            active_goals = [
                goal
                for goal in evidence_plan.goals
                if goal_ids is None or goal.id in goal_ids
            ]
            for goal in active_goals:
                candidates_for_goal = [
                    pool[chunk_id]
                    for chunk_id in goal_candidate_ids.get(goal.id, set())
                    if chunk_id in pool
                ]

                def goal_candidate_rank(candidate: Candidate) -> tuple[float, float, float, int]:
                    tags = candidate.sources
                    lexical_strength = 0.0
                    if f"goal:{goal.id}:lookup" in tags:
                        lexical_strength += 4.0
                    if f"goal:{goal.id}:coverage" in tags:
                        lexical_strength += 3.5
                    if f"goal:{goal.id}:exact" in tags:
                        lexical_strength += 3.0
                    if f"goal:{goal.id}:table" in tags:
                        lexical_strength += 2.5
                    if f"goal:{goal.id}:lexical" in tags:
                        lexical_strength += 2.0
                    if f"goal:{goal.id}:section" in tags:
                        lexical_strength += 2.8
                    if f"goal:{goal.id}:relaxed" in tags:
                        lexical_strength += 1.2
                    if candidate.content_kind == "table" and goal.kind in {"enumeration", "attribute"}:
                        lexical_strength += 1.0
                    rerank_score = candidate.rerank_score if candidate.chunk_id in final_ids else 0.0
                    return (lexical_strength, rerank_score, candidate.fused_score, -candidate.ordinal)

                candidates_for_goal.sort(key=goal_candidate_rank, reverse=True)
                existing_ids = {candidate.chunk_id for candidate in seed_by_goal[goal.id]}
                per_goal_limit = max(1, self.settings.compositional_goal_evidence_per_goal)
                if goal.kind == "definition":
                    # Ambiguous acronyms may have several genuinely distinct expansions.
                    per_goal_limit = max(
                        per_goal_limit, self.settings.compositional_definition_evidence_per_goal
                    )
                for candidate in candidates_for_goal:
                    if candidate.chunk_id in existing_ids:
                        continue
                    seed_by_goal[goal.id].append(candidate)
                    existing_ids.add(candidate.chunk_id)
                    if len(seed_by_goal[goal.id]) >= per_goal_limit:
                        break

            protected_goal = []
            protected_goal_ids: set[UUID] = set()
            ordered_goals = [goal for goal in active_goals if goal.required] + [
                goal for goal in active_goals if not goal.required
            ]
            max_depth = max((len(seed_by_goal[goal.id]) for goal in ordered_goals), default=0)
            for depth in range(max_depth):
                for goal in ordered_goals:
                    items = seed_by_goal[goal.id]
                    if depth >= len(items):
                        continue
                    candidate = items[depth]
                    if candidate.chunk_id in protected_goal_ids:
                        continue
                    protected_goal.append(candidate)
                    protected_goal_ids.add(candidate.chunk_id)
                    if len(protected_goal) >= self.settings.compositional_max_evidence_k:
                        break
                if len(protected_goal) >= self.settings.compositional_max_evidence_k:
                    break

        if source_stage == "priority":
            for candidate in pool.values():
                candidate.evidence_lane = "priority"
                candidate.sources.add("priority_source")

        self._emit(
            progress,
            "evidence",
            "Building the evidence set",
            "Keeping the highest-value passages, source diversity and nearby structural context.",
            evidence_pct,
        )
        evidence_started = time.perf_counter()
        # Preserve high-specificity evidence found from the employee's exact wording before
        # semantic expansions. This protects discriminators such as multi-word fault names,
        # policy phrases and document terminology from being washed out by broad paraphrases.
        protected_original = sorted(
            [
                candidate
                for candidate in pool.values()
                if any(
                    tag in candidate.sources
                    for tag in ("q0:exact", "q0:lexical", "q0:section")
                )
            ],
            key=lambda candidate: (
                int("q0:exact" in candidate.sources),
                int("q0:lexical" in candidate.sources),
                int("q0:section" in candidate.sources),
                candidate.fused_score,
            ),
            reverse=True,
        )[:4]
        section_context = []
        section_anchor_evidence: list[Candidate] = []
        if self.settings.hierarchical_retrieval_enabled:
            if procedure_sensitive:
                anchors = [*protected_coverage[:8], *final_candidates[:6]]
            else:
                section_anchors = [
                    candidate for candidate in final_candidates if "section" in candidate.sources
                ]
                # Priority/scoped retrieval benefits from reconstructing the governing
                # section even for an ordinary fact question; broad corpus retrieval keeps
                # this bounded to actual section-lane winners.
                anchors = section_anchors[:6]
            if anchors:
                # The anchor is the reason this section was selected. Protect it before its
                # neighbours so hierarchical expansion cannot consume the evidence budget and
                # accidentally drop the strongest matching procedure/definition chunk.
                section_anchor_evidence = anchors[:6]
                section_context = self._section_expansion(anchors, user, document_ids)
        # Overview evidence is intentionally limited: always search it, but only reserve a
        # few relevant passages so generic explanation cannot crowd out operational rules.
        protected_overview = [
            c for c in overview_candidates if c.chunk_id in {x.chunk_id for x in final_candidates}
        ][: self.settings.governing_reference_evidence_k]
        evidence_limit = max(
            limits["evidence"],
            min(self.settings.coverage_max_evidence_k, limits["evidence"] + len(section_context[:6])),
        )
        if evidence_plan is not None and evidence_plan.requires_decomposition:
            active_required_goals = [
                goal
                for goal in evidence_plan.goals
                if goal.required and (goal_ids is None or goal.id in goal_ids)
            ]
            goal_minimum = 2 + sum(
                max(
                    self.settings.compositional_definition_evidence_per_goal
                    if goal.kind == "definition" else 1,
                    self.settings.compositional_goal_evidence_per_goal,
                )
                for goal in active_required_goals
            )
            evidence_limit = max(
                evidence_limit,
                min(self.settings.compositional_max_evidence_k, goal_minimum),
            )
        evidence = self._build_evidence(
            final_candidates,
            user,
            document_ids,
            limit=evidence_limit,
            protected=(
                protected_original
                + protected_goal
                + protected_lookup
                + protected_coverage
                + protected_role
                + protected_entity
                + protected_overview
                + section_anchor_evidence
                + section_context
            ),
            use_neighbors=(
                lookup_term is None
                and not role_sensitive
                and not procedure_sensitive
                and not entity_sensitive
                and not (evidence_plan is not None and evidence_plan.requires_decomposition)
            ),
        )
        timings_ms["evidence_build"] = int((time.perf_counter() - evidence_started) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)

        evidence_chunk_ids = {e.candidate.chunk_id for e in evidence}
        evidence_document_ids = {e.candidate.document_id for e in evidence}
        procedure_complete = (
            not procedure_sensitive
            or (
                not procedure_truncated
                and {c.document_id for c in protected_coverage}.issubset(evidence_document_ids)
            )
        )
        role_identifier_unresolved = bool(
            role_sensitive
            and role_subject
            and role_resolution.get("identifier")
            and not role_aliases
        )
        role_complete = (
            not role_sensitive
            or (
                bool(protected_role)
                and not role_truncated
                and not role_identifier_unresolved
                and {c.chunk_id for c in protected_role}.issubset(evidence_chunk_ids)
            )
        )
        entity_complete = (
            "entity_attribute" not in (query_plan.coverage_kind or "")
            or (
                entity_sensitive
                and bool(protected_entity)
                and not entity_truncated
                and {c.chunk_id for c in protected_entity}.issubset(evidence_chunk_ids)
            )
        )
        goal_stats: dict[str, dict] = {}
        active_goals = [
            goal
            for goal in (evidence_plan.goals if evidence_plan is not None else [])
            if goal_ids is None or goal.id in goal_ids
        ]
        for goal in active_goals:
            attributed = [
                item
                for item in evidence
                if f"goal:{goal.id}" in item.candidate.sources
            ]
            lexical_like = [
                item
                for item in attributed
                if any(
                    f"goal:{goal.id}:{lane}" in item.candidate.sources
                    for lane in ("lookup", "coverage", "exact", "lexical", "relaxed", "section", "table")
                )
            ]
            goal_stats[goal.id] = {
                "kind": goal.kind,
                "required": goal.required,
                "question": goal.question,
                "entity_terms": goal.entity_terms,
                "evidence_count": len(attributed),
                "lexical_like_evidence_count": len(lexical_like),
                "max_rerank_score": max(
                    (item.candidate.rerank_score for item in attributed), default=0.0
                ),
                "evidence_ids": [item.evidence_id for item in attributed],
                "candidate_count": len(goal_candidate_ids.get(goal.id, set())),
                "source_counts": dict(goal_source_counts.get(goal.id, {})),
            }
        required_active_goals = [goal for goal in active_goals if goal.required]
        goal_retrieval_complete = all(
            int(goal_stats.get(goal.id, {}).get("evidence_count", 0)) > 0
            for goal in required_active_goals
        )
        legacy_coverage_complete = procedure_complete and role_complete and entity_complete
        coverage_complete = legacy_coverage_complete
        if evidence_plan is not None and evidence_plan.requires_decomposition:
            coverage_complete = coverage_complete and goal_retrieval_complete

        coverage_documents: list[Candidate] = []
        seen_coverage_docs: set[UUID] = set()
        for candidate in [*protected_coverage, *protected_role, *protected_entity]:
            if candidate.document_id in seen_coverage_docs:
                continue
            coverage_documents.append(candidate)
            seen_coverage_docs.add(candidate.document_id)

        def preview(candidate: Candidate) -> dict:
            return {
                "chunk_id": str(candidate.chunk_id),
                "document_id": str(candidate.document_id),
                "document_title": candidate.document_title,
                "page": candidate.page_from,
                "section": candidate.section_path[-1] if candidate.section_path else None,
                "fused_score": round(candidate.fused_score, 6),
                "rerank_score": round(candidate.rerank_score, 6),
                "final_retrieval_score": round(candidate.final_retrieval_score, 6),
                "rank_method": candidate.rank_method,
                "goal_rerank_scores": {
                    key: round(value, 6) for key, value in candidate.goal_rerank_scores.items()
                },
                "sources": sorted(candidate.sources),
                "evidence_lane": candidate.evidence_lane,
                "content_kind": candidate.content_kind,
                "applicability_score": round(candidate.applicability_score, 4),
            }

        trace = {
            "queries": queries,
            "query_specs": [
                {"text": query, **query_metadata[index]}
                for index, query in enumerate(queries)
            ],
            "profile": profile,
            "source_stage": source_stage,
            "compositional": bool(evidence_plan and evidence_plan.requires_decomposition),
            "evidence_plan": evidence_plan.as_dict() if evidence_plan is not None else None,
            "goal_ids_requested": sorted(goal_ids) if goal_ids else None,
            "goal_stats": goal_stats,
            "goal_retrieval_complete": goal_retrieval_complete,
            "goal_source_counts": {goal_id: dict(counts) for goal_id, counts in goal_source_counts.items()},
            "goal_coverage_stats": goal_coverage_stats,
            "protected_goal_candidates": len(protected_goal),
            "rerank_query": rerank_query,
            "lookup_term": lookup_term,
            "entity_term": query_plan.entity_term,
            "entity_terms": query_plan.entity_terms,
            "query_facets": query_plan.facets,
            "attribute_terms": query_plan.attribute_terms,
            "lexical_queries": query_plan.lexical_queries,
            "exact_terms": query_plan.exact_terms,
            "source_counts": source_counts,
            "overview_document_pattern": self.settings.overview_document_pattern,
            "overview_document_ids": [str(item) for item in overview_ids],
            "overview_candidates": len(overview_candidates),
            "table_candidates": sum(1 for c in fused_candidates if c.content_kind == "table"),
            "hierarchical_context_chunks": len(section_context),
            "fused_candidates": len(fused_candidates),
            "reranked_candidates": len(final_candidates),
            "protected_lookup_candidates": len(protected_lookup),
            "coverage_sensitive": coverage_sensitive,
            "coverage_kind": coverage_kind,
            "coverage_documents_discovered": len(coverage_documents),
            "coverage_document_ids": [str(c.document_id) for c in coverage_documents],
            "coverage_document_titles": [c.document_title for c in coverage_documents],
            "coverage_truncated": procedure_truncated or role_truncated or entity_truncated,
            "coverage_complete": coverage_complete,
            "legacy_coverage_complete": legacy_coverage_complete,
            "entity_attribute_coverage_sensitive": entity_sensitive,
            "entity_coverage_documents": len({c.document_id for c in protected_entity}),
            "entity_coverage_candidates": len(protected_entity),
            "entity_coverage_stats": entity_stats,
            "entity_coverage_truncated": entity_truncated,
            "role_coverage_sensitive": role_sensitive,
            "role_subject": role_subject,
            "role_aliases": role_aliases,
            "role_alias_resolution": role_resolution,
            "role_sections_discovered": role_stats["sections_discovered"],
            "role_sections_selected": role_stats["sections_selected"],
            "role_documents_discovered": role_stats["documents_discovered"],
            "role_documents_selected": role_stats["documents_selected"],
            "role_coverage_sql_queries": role_stats.get("sql_queries", 0),
            "role_identifier_unresolved": role_identifier_unresolved,
            "role_coverage_truncated": role_truncated,
            "top_rerank_score": final_candidates[0].rerank_score if final_candidates else 0.0,
            "definition_fast_path_satisfied": definition_fast_path,
            "definition_scores": {str(k): v for k, v in definition_scores.items()} if definition_scores else {},
            "definition_source_scope": query_plan.source_scope,
            "rerank_details": rerank_details,
            "timings_ms": timings_ms,
            "total_ms": total_ms,
            "fused_candidate_preview": [preview(candidate) for candidate in fused_candidates[:20]],
            "reranked_candidate_preview": [preview(candidate) for candidate in final_candidates[:20]],
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "chunk_id": str(e.candidate.chunk_id),
                    "document_id": str(e.candidate.document_id),
                    "document_title": e.candidate.document_title,
                    "family_key": e.candidate.family_key,
                    "page": e.candidate.page_from,
                    "section": e.candidate.section_path[-1] if e.candidate.section_path else None,
                    "rerank_score": e.candidate.rerank_score,
                    "sources": sorted(e.candidate.sources),
                }
                for e in evidence
            ],
        }
        return evidence, trace

    def _build_evidence(
        self,
        ranked: list[Candidate],
        user: User,
        document_ids: list[UUID] | None,
        *,
        limit: int,
        protected: list[Candidate] | None = None,
        use_neighbors: bool = True,
    ) -> list[Evidence]:
        selected: list[Candidate] = []
        seen: set[UUID] = set()

        # For acronym/definition lookups, protect likely definition-bearing chunks before
        # ordinary top-k selection. This is corpus-generic and prevents one document from
        # crowding out a different meaning in another document.
        for candidate in protected or []:
            if candidate.chunk_id in seen or len(selected) >= limit:
                continue
            selected.append(candidate)
            seen.add(candidate.chunk_id)

        seed_target = min(max(4, limit // 2), len(ranked), limit)
        for candidate in ranked:
            if len(selected) >= seed_target and selected:
                break
            if candidate.chunk_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.chunk_id)

        if use_neighbors and self.settings.neighbor_radius > 0 and selected:
            for seed in list(selected):
                if len(selected) >= limit:
                    break
                low = max(0, seed.ordinal - self.settings.neighbor_radius)
                high = seed.ordinal + self.settings.neighbor_radius
                stmt = (
                    select(Chunk, Document)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(
                        *self._base_filters(user, document_ids),
                        Chunk.document_id == seed.document_id,
                        Chunk.ordinal.between(low, high),
                    )
                    .order_by(Chunk.ordinal)
                )
                for chunk, document in self.db.execute(stmt).all():
                    if chunk.id in seen or len(selected) >= limit:
                        continue
                    neighbor = self._candidate(chunk, document)
                    neighbor.sources.add("neighbor")
                    neighbor.rerank_score = max(0.0, seed.rerank_score - 0.01)
                    neighbor.final_retrieval_score = max(0.0, seed.final_retrieval_score - 0.01)
                    neighbor.rank_method = seed.rank_method
                    selected.append(neighbor)
                    seen.add(chunk.id)

        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate.chunk_id not in seen:
                selected.append(candidate)
                seen.add(candidate.chunk_id)

        return [Evidence(evidence_id=f"E{idx}", candidate=candidate) for idx, candidate in enumerate(selected, start=1)]

    @staticmethod
    def confidence(evidence: list[Evidence]) -> str:
        if not evidence:
            return "low"
        score = max((item.candidate.rerank_score for item in evidence), default=0.0)
        if score >= 0.70 and len(evidence) >= 4:
            return "high"
        if score >= 0.30 and len(evidence) >= 2:
            return "medium"
        return "low"
