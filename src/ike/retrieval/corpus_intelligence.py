from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from uuid import UUID

from sqlalchemy import delete, desc, func, select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import Chunk, CorpusTerm, Document, RetrievalNode, User
from ike.retrieval.access import document_access_clause
from ike.services.inference_client import InferenceClient
from ike.retrieval.vocabulary import fuzzy_query_tokens, normalize_term

logger = logging.getLogger(__name__)

INDEX_VERSION = "0.10.0"

_GENERIC = {
    "chapter", "section", "procedure", "procedures", "general", "information", "note",
    "notes", "table", "annexure", "appendix", "contents", "scope", "introduction",
    "page", "para", "paragraph", "revision", "document", "system", "systems", "unsectioned",
}
_DEFINED_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9 /_-]{1,60}?)\s+(?:means|refers to|is defined as|shall mean)\b",
    re.IGNORECASE,
)
_FULL_FORM_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9/& -]{2,70}?)\s*\(([A-Z][A-Z0-9/-]{1,12})\)",
)
_REVERSE_FULL_FORM_RE = re.compile(
    r"\b([A-Z][A-Z0-9/-]{1,12})\s*\(([A-Za-z][A-Za-z0-9/& -]{2,70}?)\)",
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9/-]{1,12}\b")
_TABLE_ONLY_RE = re.compile(r"^(?:table|figure|fig\.?|annexure|appendix)\s*[-.: ]*\d", re.IGNORECASE)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n:;,-")


def _term_type(value: str, *, default: str = "term") -> str:
    cleaned = _clean(value)
    if _ACRONYM_RE.fullmatch(cleaned):
        return "acronym"
    return default


def _node_key(node_type: str, value: str) -> str:
    digest = hashlib.sha1(value.casefold().encode("utf-8"), usedforsecurity=False).hexdigest()[:24]
    return f"{node_type}:{digest}"


def _section_key(path: list[str]) -> str:
    return " > ".join(_clean(part).casefold() for part in path if _clean(part))


def _high_signal_concepts(label: str, body: str, *, limit: int) -> list[str]:
    """Extract search aliases without creating standalone semantic vectors.

    0.9.0 treated almost every uppercase fragment as an embedded concept node. In real
    manuals that admitted states/colours/fragments such as LOW, RED or sentence pieces and
    multiplied vector work. 0.10 keeps only safer terminology sources: real headings,
    explicit definitions, and acronym/full-form pairs. The aliases are indexed lexically on
    the parent section node while the section's semantic vector is derived from existing
    chunk embeddings.
    """

    values: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = _clean(value)
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        if not cleaned or len(cleaned) < 3 or len(cleaned) > 90:
            return
        if cleaned.casefold() == "unsectioned" or _TABLE_ONLY_RE.match(cleaned):
            return
        words = [word.casefold() for word in re.findall(r"[A-Za-z0-9]+", cleaned)]
        if words and all(word in _GENERIC or word.isdigit() for word in words):
            return
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            values.append(cleaned)

    # Section headings are the strongest organisation-specific vocabulary source.
    for part in re.split(r"\s*[>|]\s*", label):
        add(part)
        for acronym in _ACRONYM_RE.findall(part):
            add(acronym)
        if len(values) >= limit:
            return values[:limit]

    sample = body[:9000]
    for match in _DEFINED_RE.finditer(sample):
        add(match.group(1))
        if len(values) >= limit:
            return values[:limit]

    # Acronyms are retained only when tied to a nearby full form. This avoids promoting
    # arbitrary uppercase states/colours/table cells into corpus terminology.
    for regex in (_FULL_FORM_RE, _REVERSE_FULL_FORM_RE):
        for match in regex.finditer(sample):
            add(match.group(1))
            add(match.group(2))
            if len(values) >= limit:
                return values[:limit]
    return values[:limit]


def _normalized_centroid(vectors: list[list[float] | tuple[float, ...]]) -> list[float]:
    """Return a normalized centroid over existing BGE-M3 chunk vectors."""

    usable: list[list[float]] = []
    expected_dim: int | None = None
    for raw in vectors:
        if raw is None:
            continue
        vector = [float(value) for value in raw]
        if not vector or not all(math.isfinite(value) for value in vector):
            continue
        if expected_dim is None:
            expected_dim = len(vector)
        if len(vector) != expected_dim:
            continue
        usable.append(vector)
    if not usable or expected_dim is None:
        raise RuntimeError("No valid chunk embeddings available for retrieval-intelligence centroid")

    sums = [0.0] * expected_dim
    for vector in usable:
        for index, value in enumerate(vector):
            sums[index] += value
    inv = 1.0 / len(usable)
    centroid = [value * inv for value in sums]
    norm = math.sqrt(sum(value * value for value in centroid))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("Invalid retrieval-intelligence centroid norm")
    return [value / norm for value in centroid]


@dataclass(slots=True)
class CorpusHit:
    node_id: UUID
    document_id: UUID
    node_type: str
    label: str
    section_path: list[str]
    page_from: int | None
    page_to: int | None
    score: float
    document_title: str
    filename: str
    family_key: str | None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["node_id"] = str(self.node_id)
        data["document_id"] = str(self.document_id)
        return data


@dataclass(slots=True)
class CorpusDiscovery:
    available: bool = False
    query: str = ""
    terms: list[str] = field(default_factory=list)
    resolved_terms: list[str] = field(default_factory=list)
    document_ids: list[UUID] = field(default_factory=list)
    family_keys: list[str] = field(default_factory=list)
    hits: list[CorpusHit] = field(default_factory=list)
    timings_ms: dict[str, int | float] = field(default_factory=dict)
    reason: str = ""
    scope_documents: int = 0
    indexed_documents: int = 0
    coverage_ratio: float = 0.0
    routing_safe: bool = False

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "query": self.query,
            "terms": self.terms,
            "resolved_terms": self.resolved_terms,
            "document_ids": [str(item) for item in self.document_ids],
            "family_keys": self.family_keys,
            "hits": [item.as_dict() for item in self.hits],
            "timings_ms": self.timings_ms,
            "reason": self.reason,
            "scope_documents": self.scope_documents,
            "indexed_documents": self.indexed_documents,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "routing_safe": self.routing_safe,
            "index_version": INDEX_VERSION,
        }

    def prompt_block(self, *, max_hits: int = 10) -> str:
        if not self.available or not self.hits:
            return "No corpus-intelligence hints were available."
        lines = [
            "The following are SEARCH HINTS extracted from the indexed corpus. They are not facts and must not be copied into the answer unless retrieved chunk evidence supports them.",
            f"Corpus-index coverage for this scope: {self.indexed_documents}/{self.scope_documents} ({self.coverage_ratio:.1%}); routing-index complete={self.routing_safe}.",
            "Corpus terminology: " + (", ".join(self.terms) if self.terms else "none"),
            "Fuzzy/canonical search expansions: "
            + (", ".join(self.resolved_terms) if self.resolved_terms else "none"),
            "Potentially relevant indexed sections/documents:",
        ]
        for hit in self.hits[:max_hits]:
            section = " > ".join(hit.section_path) or hit.label
            lines.append(f"- {hit.document_title}: {section}")
        return "\n".join(lines)


def build_nodes_for_document(db: Session, document_id: UUID, *, inference: InferenceClient | None = None) -> int:
    """Build document/section routing nodes from existing chunks and embeddings.

    No PDF parsing and no new document-embedding inference is performed. The semantic
    vectors are normalized centroids of the already stored BGE-M3 chunk vectors. High-signal
    terminology is attached to section nodes for PostgreSQL FTS rather than materialized as
    hundreds of duplicate vector-bearing concept rows.

    ``inference`` is retained in the function signature for rolling compatibility with the
    ingestion worker but is intentionally unused in 0.10.
    """

    del inference
    settings = get_settings()
    document = db.get(Document, document_id)
    if document is None or document.ingestion_status != "ready":
        return 0
    chunks = list(
        db.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
        ).all()
    )
    if not chunks:
        return 0

    groups: dict[str, list[Chunk]] = defaultdict(list)
    paths: dict[str, list[str]] = {}
    for chunk in chunks:
        path = [_clean(part) for part in (chunk.section_path or []) if _clean(part)]
        key = _section_key(path) or f"unsectioned:{chunk.ordinal // 4}"
        groups[key].append(chunk)
        paths[key] = path

    headings = [" > ".join(path) for path in paths.values() if path]
    doc_label = _clean(document.title or document.original_filename)
    doc_text = "\n".join(
        filter(
            None,
            [
                f"Title: {doc_label}",
                f"File: {document.original_filename}",
                f"Family: {document.family_key or ''}",
                f"Authority: {document.authority or ''}",
                f"Revision: {document.revision or ''}",
                "Sections: " + " | ".join(headings[:80]),
            ],
        )
    )[: settings.retrieval_intelligence_section_chars]

    records: list[tuple[dict, list[float]]] = [
        (
            {
                "node_type": "document",
                "node_key": _node_key("document", str(document_id)),
                "label": doc_label,
                "section_path": [],
                "page_from": min((c.page_from for c in chunks if c.page_from is not None), default=None),
                "page_to": max((c.page_to for c in chunks if c.page_to is not None), default=None),
                "ordinal_from": chunks[0].ordinal,
                "ordinal_to": chunks[-1].ordinal,
                "text": doc_text,
                "source_metadata": {
                    "generated_from": "chunk_embeddings",
                    "kind": "document_summary",
                    "index_version": INDEX_VERSION,
                    "chunk_count": len(chunks),
                },
            },
            _normalized_centroid([chunk.embedding for chunk in chunks]),
        )
    ]

    terms_limit = max(1, settings.retrieval_intelligence_terms_per_section)
    for key, section_chunks in groups.items():
        path = paths[key]
        label = " > ".join(path) if path else "Unsectioned"
        bodies = [c.contextual_text or c.text for c in section_chunks]
        joined = "\n\n".join(bodies)
        max_chars = settings.retrieval_intelligence_section_chars
        if len(joined) > max_chars:
            half = max_chars // 2
            joined = joined[:half] + "\n...\n" + joined[-half:]
        terms = _high_signal_concepts(label, joined, limit=terms_limit)
        lexical_text = joined
        if terms:
            suffix = "\n\nCorpus terminology: " + " | ".join(terms)
            if len(lexical_text) + len(suffix) <= max_chars:
                lexical_text += suffix
            else:
                lexical_text = lexical_text[: max(0, max_chars - len(suffix))] + suffix
        records.append(
            (
                {
                    "node_type": "section",
                    "node_key": _node_key("section", key),
                    "label": label[:1000],
                    "section_path": path,
                    "page_from": min((c.page_from for c in section_chunks if c.page_from is not None), default=None),
                    "page_to": max((c.page_to for c in section_chunks if c.page_to is not None), default=None),
                    "ordinal_from": section_chunks[0].ordinal,
                    "ordinal_to": section_chunks[-1].ordinal,
                    "text": lexical_text,
                    "source_metadata": {
                        "generated_from": "chunk_embeddings",
                        "kind": "section",
                        "chunk_count": len(section_chunks),
                        "terms": terms,
                        "index_version": INDEX_VERSION,
                    },
                },
                _normalized_centroid([chunk.embedding for chunk in section_chunks]),
            )
        )

    # One document is replaced atomically. This makes interrupted backfills resumable and
    # prevents half-indexed documents from becoming visible to query sessions.
    db.execute(delete(RetrievalNode).where(RetrievalNode.document_id == document_id))
    db.execute(delete(CorpusTerm).where(CorpusTerm.document_id == document_id))
    for row, vector in records:
        db.add(RetrievalNode(document_id=document_id, embedding=vector, **row))

    term_rows: dict[tuple[str, str], dict] = {}

    def register_term(
        value: str,
        term_type: str,
        *,
        section_path: list[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        cleaned = _clean(value)
        normalized = normalize_term(cleaned)
        if len(normalized) < 3 or len(cleaned) > 300:
            return
        key = (normalized, term_type)
        term_rows.setdefault(
            key,
            {
                "term": cleaned,
                "normalized_term": normalized,
                "term_type": term_type,
                "section_path": list(section_path or []),
                "page_from": page_from,
                "page_to": page_to,
                "source_metadata": {
                    "index_version": INDEX_VERSION,
                    **(metadata or {}),
                },
            },
        )

    register_term(doc_label, "document_title", metadata={"source": "document_title"})
    if document.family_key:
        register_term(document.family_key, "family", metadata={"source": "family_key"})

    for row, _vector in records[1:]:
        path = list(row.get("section_path") or [])
        for heading in path:
            register_term(
                heading,
                "heading",
                section_path=path,
                page_from=row.get("page_from"),
                page_to=row.get("page_to"),
                metadata={"source": "heading"},
            )
        for value in (row.get("source_metadata") or {}).get("terms") or []:
            register_term(
                str(value),
                _term_type(str(value)),
                section_path=path,
                page_from=row.get("page_from"),
                page_to=row.get("page_to"),
                metadata={"source": "section_concept"},
            )

    for row in term_rows.values():
        db.add(CorpusTerm(document_id=document_id, **row))
    return len(records)


class CorpusIntelligence:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.inference = InferenceClient.for_query()

    def _resolve_corpus_terms(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None,
    ) -> list[str]:
        """Return corpus vocabulary candidates for fuzzy/terminology expansion.

        Returned values are retrieval hints only.  The original query remains Q0 and the
        answer generator never treats a vocabulary hit as evidence.
        """

        if not self.settings.corpus_term_resolution_enabled:
            return []
        tokens = fuzzy_query_tokens(
            query, limit=self.settings.corpus_term_max_query_tokens
        )
        if not tokens:
            return []

        filters = [document_access_clause(user), Document.ingestion_status == "ready"]
        if document_ids:
            filters.append(Document.id.in_(document_ids))

        candidates: list[tuple[float, str]] = []
        try:
            for token in tokens:
                similarity = func.similarity(CorpusTerm.normalized_term, token).label("similarity")
                stmt = (
                    select(CorpusTerm.term, CorpusTerm.normalized_term, similarity)
                    .join(Document, Document.id == CorpusTerm.document_id)
                    .where(*filters, CorpusTerm.normalized_term.op("%")(token))
                    .order_by(desc(similarity))
                    .limit(self.settings.corpus_term_candidates_per_token)
                )
                for term, normalized, score in self.db.execute(stmt).all():
                    numeric = float(score or 0.0)
                    if numeric < self.settings.corpus_term_similarity_threshold:
                        continue
                    # Exact tokens do not need an expansion; preserve only alternate corpus
                    # formulations that can improve downstream retrieval.
                    if normalized == token:
                        continue
                    candidates.append((numeric, str(term)))
        except ProgrammingError:
            self.db.rollback()
            return []
        except Exception:
            logger.warning("corpus_term_resolution_failed", exc_info=True)
            return []

        ordered: list[str] = []
        seen: set[str] = set()
        for _score, term in sorted(candidates, key=lambda item: item[0], reverse=True):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(term)
            if len(ordered) >= self.settings.corpus_term_expansions_max:
                break
        return ordered

    def _coverage(
        self,
        user: User,
        document_ids: list[UUID] | None,
    ) -> tuple[int, int, float]:
        filters = [document_access_clause(user), Document.ingestion_status == "ready"]
        if document_ids:
            filters.append(Document.id.in_(document_ids))
        total = int(self.db.scalar(select(func.count(Document.id)).where(*filters)) or 0)
        indexed = int(
            self.db.scalar(
                select(func.count(func.distinct(RetrievalNode.document_id)))
                .select_from(RetrievalNode)
                .join(Document, Document.id == RetrievalNode.document_id)
                .where(*filters)
            )
            or 0
        )
        ratio = (indexed / total) if total else 0.0
        return total, indexed, ratio

    def discover(
        self,
        query: str,
        user: User,
        document_ids: list[UUID] | None = None,
    ) -> CorpusDiscovery:
        if not self.settings.retrieval_intelligence_enabled:
            return CorpusDiscovery(query=query, reason="disabled")
        import time

        started = time.perf_counter()
        try:
            scope_documents, indexed_documents, coverage_ratio = self._coverage(user, document_ids)
        except ProgrammingError:
            self.db.rollback()
            return CorpusDiscovery(query=query, reason="retrieval_nodes_unavailable")
        except Exception as exc:
            logger.warning("corpus_intelligence_coverage_failed", exc_info=True)
            return CorpusDiscovery(query=query, reason=f"coverage_error:{type(exc).__name__}")

        min_hint_coverage = max(0.0, min(1.0, self.settings.retrieval_intelligence_min_hint_coverage_ratio))
        if scope_documents == 0:
            return CorpusDiscovery(
                query=query,
                reason="empty_scope",
                scope_documents=0,
                indexed_documents=0,
                coverage_ratio=0.0,
            )
        if indexed_documents == 0 or coverage_ratio < min_hint_coverage:
            return CorpusDiscovery(
                query=query,
                reason="coverage_below_hint_threshold",
                scope_documents=scope_documents,
                indexed_documents=indexed_documents,
                coverage_ratio=coverage_ratio,
                routing_safe=False,
                timings_ms={"total": int((time.perf_counter() - started) * 1000)},
            )

        # ``routing_safe`` remains an observability/coverage indicator only. 0.10 never uses
        # it to turn a probabilistic document router into a hard scope.
        routing_safe = indexed_documents == scope_documents
        vocabulary_terms = self._resolve_corpus_terms(query, user, document_ids)
        filters = [document_access_clause(user), Document.ingestion_status == "ready"]
        if document_ids:
            filters.append(Document.id.in_(document_ids))
        try:
            vectors, embed_timing = self.inference.embed_queries([query])
            vector = vectors[0]
            dense_distance = RetrievalNode.embedding.cosine_distance(vector).label("distance")
            dense_stmt = (
                select(RetrievalNode, Document, dense_distance)
                .join(Document, Document.id == RetrievalNode.document_id)
                .where(*filters)
                .order_by(dense_distance)
                .limit(self.settings.retrieval_intelligence_dense_top_k)
            )
            dense_rows = self.db.execute(dense_stmt).all()

            tsquery = func.websearch_to_tsquery("simple", query)
            rank = func.ts_rank_cd(RetrievalNode.search_vector, tsquery).label("rank")
            lexical_stmt = (
                select(RetrievalNode, Document, rank)
                .join(Document, Document.id == RetrievalNode.document_id)
                .where(*filters, RetrievalNode.search_vector.op("@@")(tsquery))
                .order_by(desc(rank))
                .limit(self.settings.retrieval_intelligence_lexical_top_k)
            )
            lexical_rows = self.db.execute(lexical_stmt).all()
        except ProgrammingError:
            self.db.rollback()
            return CorpusDiscovery(
                query=query,
                reason="retrieval_nodes_unavailable",
                scope_documents=scope_documents,
                indexed_documents=indexed_documents,
                coverage_ratio=coverage_ratio,
            )
        except Exception as exc:
            logger.warning("corpus_intelligence_discovery_failed", exc_info=True)
            return CorpusDiscovery(
                query=query,
                reason=f"error:{type(exc).__name__}",
                scope_documents=scope_documents,
                indexed_documents=indexed_documents,
                coverage_ratio=coverage_ratio,
            )

        scores: defaultdict[UUID, float] = defaultdict(float)
        rows: dict[UUID, tuple[RetrievalNode, Document]] = {}
        rrf_k = max(1, self.settings.rrf_k)
        for rank_index, row in enumerate(dense_rows, start=1):
            node, document = row[0], row[1]
            rows[node.id] = (node, document)
            scores[node.id] += 1.0 / (rrf_k + rank_index)
        for rank_index, row in enumerate(lexical_rows, start=1):
            node, document = row[0], row[1]
            rows[node.id] = (node, document)
            scores[node.id] += 1.15 / (rrf_k + rank_index)

        ordered = sorted(scores, key=scores.get, reverse=True)[: self.settings.retrieval_intelligence_fused_top_k]
        hits: list[CorpusHit] = []
        terms: list[str] = list(vocabulary_terms[: self.settings.retrieval_intelligence_terms_max])
        term_seen: set[str] = {item.casefold() for item in terms}
        routed: list[UUID] = []

        def add_term(value: str) -> None:
            cleaned = _clean(value)
            key = cleaned.casefold()
            if (
                cleaned
                and len(terms) < self.settings.retrieval_intelligence_terms_max
                and key not in term_seen
                and cleaned.casefold() != "unsectioned"
            ):
                term_seen.add(key)
                terms.append(cleaned)

        for node_id in ordered:
            node, document = rows[node_id]
            hits.append(
                CorpusHit(
                    node_id=node.id,
                    document_id=document.id,
                    node_type=node.node_type,
                    label=node.label,
                    section_path=node.section_path or [],
                    page_from=node.page_from,
                    page_to=node.page_to,
                    score=float(scores[node_id]),
                    document_title=document.title,
                    filename=document.original_filename,
                    family_key=document.family_key,
                )
            )
            if document.id not in routed:
                routed.append(document.id)
            if node.node_type == "section":
                add_term(node.label)
                metadata_terms = (node.source_metadata or {}).get("terms") or []
                for term in metadata_terms:
                    add_term(str(term))

        family_keys = list(
            dict.fromkeys(
                hit.family_key.strip()
                for hit in hits
                if hit.family_key and hit.family_key.strip()
            )
        )
        if family_keys and not document_ids and routing_safe:
            family_stmt = (
                select(Document.id)
                .where(
                    document_access_clause(user),
                    Document.ingestion_status == "ready",
                    Document.family_key.in_(family_keys[:6]),
                )
                .order_by(desc(Document.updated_at))
                .limit(self.settings.retrieval_intelligence_family_expansion_max_documents)
            )
            for related_id in self.db.scalars(family_stmt).all():
                if related_id not in routed:
                    routed.append(related_id)

        elapsed = int((time.perf_counter() - started) * 1000)
        return CorpusDiscovery(
            available=bool(hits),
            query=query,
            terms=terms,
            resolved_terms=vocabulary_terms,
            document_ids=routed,
            family_keys=family_keys,
            hits=hits,
            timings_ms={
                "total": elapsed,
                "embedding_queue_wait": embed_timing.queue_wait_ms,
                "embedding_execution": embed_timing.execution_ms,
                "dense_hits": len(dense_rows),
                "lexical_hits": len(lexical_rows),
                "coverage_ratio": coverage_ratio,
            },
            reason="ok" if hits else "no_hits",
            scope_documents=scope_documents,
            indexed_documents=indexed_documents,
            coverage_ratio=coverage_ratio,
            routing_safe=routing_safe,
        )
