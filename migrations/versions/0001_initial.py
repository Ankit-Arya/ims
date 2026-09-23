"""Initial schema with pgvector, ACL-aware documents, chunks, query logs, feedback, and reports."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(120), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="user"),
        sa.Column("department", sa.String(120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('admin','analyst','user')", name="ck_users_role"),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("family_key", sa.String(200), nullable=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("original_filename", sa.String(500), nullable=False),
        sa.Column("checksum_sha256", sa.String(64), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("parsed_path", sa.Text(), nullable=True),
        sa.Column("revision", sa.String(100), nullable=True),
        sa.Column("authority", sa.String(100), nullable=True),
        sa.Column("department", sa.String(120), nullable=True),
        sa.Column("access_scope", sa.String(32), nullable=False, server_default="organization"),
        sa.Column("allowed_roles", postgresql.ARRAY(sa.String(32)), nullable=False, server_default="{}"),
        sa.Column("allowed_departments", postgresql.ARRAY(sa.String(120)), nullable=False, server_default="{}"),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("lifecycle_status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("ingestion_status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("ingestion_error", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("extra_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("access_scope IN ('organization','department','restricted')", name="ck_documents_access_scope"),
        sa.CheckConstraint("lifecycle_status IN ('active','archived')", name="ck_documents_lifecycle_status"),
        sa.CheckConstraint("ingestion_status IN ('queued','processing','ready','failed')", name="ck_documents_ingestion_status"),
        sa.CheckConstraint("effective_from IS NULL OR effective_to IS NULL OR effective_to >= effective_from", name="ck_documents_effective_dates"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column("section_path", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("content_kind", sa.String(64), nullable=False, server_default="text"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("contextual_text", sa.Text(), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("embedding", VECTOR(1024), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), sa.Computed("to_tsvector('simple', coalesce(contextual_text, ''))", persisted=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunk_document_ordinal"),
    )
    op.execute("CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 96)")

    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("retrieval_trace", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_log_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=True),
        sa.Column("complete", sa.Boolean(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating"),
    )

    op.create_table(
        "report_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("document_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("result_markdown", sa.Text(), nullable=True),
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('queued','running','complete','failed')", name="ck_report_jobs_status"),
    )

    # Explicit operational indexes; vector and FTS indexes are created above.
    op.create_index("ix_documents_family_key", "documents", ["family_key"])
    op.create_index("ix_documents_checksum_sha256", "documents", ["checksum_sha256"])
    op.create_index("ix_documents_department", "documents", ["department"])
    op.create_index("ix_documents_lifecycle_status", "documents", ["lifecycle_status"])
    op.create_index("ix_documents_ingestion_status", "documents", ["ingestion_status"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_page_from", "chunks", ["page_from"])
    op.create_index("ix_query_logs_user_id", "query_logs", ["user_id"])
    op.create_index("ix_feedback_query_log_id", "feedback", ["query_log_id"])
    op.create_index("ix_report_jobs_created_by", "report_jobs", ["created_by"])
    op.create_index("ix_report_jobs_status", "report_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_report_jobs_status", table_name="report_jobs")
    op.drop_index("ix_report_jobs_created_by", table_name="report_jobs")
    op.drop_index("ix_feedback_query_log_id", table_name="feedback")
    op.drop_index("ix_query_logs_user_id", table_name="query_logs")
    op.drop_index("ix_chunks_page_from", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("ix_documents_ingestion_status", table_name="documents")
    op.drop_index("ix_documents_lifecycle_status", table_name="documents")
    op.drop_index("ix_documents_department", table_name="documents")
    op.drop_index("ix_documents_checksum_sha256", table_name="documents")
    op.drop_index("ix_documents_family_key", table_name="documents")
    op.drop_table("report_jobs")
    op.drop_table("feedback")
    op.drop_table("query_logs")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("users")
