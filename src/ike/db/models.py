from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Boolean, CheckConstraint, Computed, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ike.db.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin','analyst','user')", name="ck_users_role"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="user")
    department: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("workspace_scope IN ('organization','personal')", name="ck_documents_workspace_scope"),
        CheckConstraint("access_scope IN ('organization','department','restricted')", name="ck_documents_access_scope"),
        CheckConstraint("lifecycle_status IN ('active','archived')", name="ck_documents_lifecycle_status"),
        CheckConstraint("ingestion_status IN ('queued','processing','ready','failed')", name="ck_documents_ingestion_status"),
        CheckConstraint("effective_from IS NULL OR effective_to IS NULL OR effective_to >= effective_from", name="ck_documents_effective_dates"),
        CheckConstraint(
            "authority_level IS NULL OR (authority_level >= 0 AND authority_level <= 100)",
            name="ck_documents_authority_level",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    family_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(500))
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_path: Mapped[str] = mapped_column(Text)
    parsed_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision: Mapped[str | None] = mapped_column(String(100), nullable=True)
    authority: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Generic evidence authority metadata. source_role is administrator-confirmable and
    # intentionally domain-neutral (for example governing_reference, policy,
    # technical_specification, operating_procedure).
    source_role: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    authority_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supersedes_document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    department: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    workspace_scope: Mapped[str] = mapped_column(String(32), default="organization", index=True)
    access_scope: Mapped[str] = mapped_column(String(32), default="organization")
    allowed_roles: Mapped[list[str]] = mapped_column(ARRAY(String(32)), default=list)
    allowed_departments: Mapped[list[str]] = mapped_column(ARRAY(String(120)), default=list)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    lifecycle_status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    ingestion_status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    ingestion_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    retrieval_nodes: Mapped[list["RetrievalNode"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    corpus_terms: Mapped[list["CorpusTerm"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    shares: Mapped[list["DocumentShare"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentShare(Base):
    __tablename__ = "document_shares"
    __table_args__ = (UniqueConstraint("document_id", "user_id", name="uq_document_share_user"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    shared_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="shares")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "ordinal", name="uq_chunk_document_ordinal"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    content_kind: Mapped[str] = mapped_column(String(64), default="text")
    text: Mapped[str] = mapped_column(Text)
    contextual_text: Mapped[str] = mapped_column(Text)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(1024))
    search_vector: Mapped[Any] = mapped_column(TSVECTOR, Computed("to_tsvector('simple', coalesce(contextual_text, ''))", persisted=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")


class RetrievalNode(Base):
    """Hierarchical/corpus-vocabulary search node built from already ingested chunks."""

    __tablename__ = "retrieval_nodes"
    __table_args__ = (
        UniqueConstraint("document_id", "node_key", name="uq_retrieval_node_document_key"),
        CheckConstraint("node_type IN ('document','section','concept')", name="ck_retrieval_nodes_type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    node_type: Mapped[str] = mapped_column(String(32), index=True)
    node_key: Mapped[str] = mapped_column(String(320))
    label: Mapped[str] = mapped_column(String(1000))
    section_path: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ordinal_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ordinal_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(1024))
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(label, '') || ' ' || coalesce(text, ''))", persisted=True),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="retrieval_nodes")


class CorpusTerm(Base):
    """High-signal organization vocabulary derived from trusted document structure.

    These rows are retrieval hints, never factual answer evidence. They support typo and
    terminology resolution without turning application code into a domain-specific ontology.
    """

    __tablename__ = "corpus_terms"
    __table_args__ = (
        UniqueConstraint("document_id", "normalized_term", "term_type", name="uq_corpus_term_document_value_type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    term: Mapped[str] = mapped_column(String(300))
    normalized_term: Mapped[str] = mapped_column(String(300), index=True)
    term_type: Mapped[str] = mapped_column(String(32), default="term", index=True)
    section_path: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="corpus_terms")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(32))
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    retrieval_trace: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    query_log_id: Mapped[UUID] = mapped_column(ForeignKey("query_logs.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    rating: Mapped[int] = mapped_column(Integer)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    complete: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReportJob(Base):
    __tablename__ = "report_jobs"
    __table_args__ = (CheckConstraint("status IN ('queued','running','complete','failed')", name="ck_report_jobs_status"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    objective: Mapped[str] = mapped_column(Text)
    document_ids: Mapped[list[UUID]] = mapped_column(ARRAY(PGUUID(as_uuid=True)))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    result_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
