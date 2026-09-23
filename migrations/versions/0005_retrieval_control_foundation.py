"""Add generic source authority metadata and corpus vocabulary for IMS 0.10.

This migration is additive and does not rewrite existing chunks or embeddings.  Existing
installations can continue serving queries while retrieval intelligence is backfilled.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_retrieval_control"
down_revision = "0004_retrieval_intelligence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_role", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("authority_level", sa.Integer(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "supersedes_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "ck_documents_authority_level",
        "documents",
        "authority_level IS NULL OR (authority_level >= 0 AND authority_level <= 100)",
    )
    op.create_index("ix_documents_source_role", "documents", ["source_role"])
    op.create_index(
        "ix_documents_supersedes_document_id", "documents", ["supersedes_document_id"]
    )

    op.create_table(
        "corpus_terms",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("term", sa.String(300), nullable=False),
        sa.Column("normalized_term", sa.String(300), nullable=False),
        sa.Column("term_type", sa.String(32), nullable=False, server_default="term"),
        sa.Column(
            "section_path", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"
        ),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column(
            "source_metadata", postgresql.JSONB(), nullable=False, server_default="{}"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "document_id",
            "normalized_term",
            "term_type",
            name="uq_corpus_term_document_value_type",
        ),
    )
    op.create_index("ix_corpus_terms_document_id", "corpus_terms", ["document_id"])
    op.create_index("ix_corpus_terms_normalized_term", "corpus_terms", ["normalized_term"])
    op.create_index("ix_corpus_terms_term_type", "corpus_terms", ["term_type"])
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_corpus_terms_normalized_term_trgm "
        "ON corpus_terms USING gin (normalized_term gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_corpus_terms_normalized_term_trgm")
    op.drop_index("ix_corpus_terms_term_type", table_name="corpus_terms")
    op.drop_index("ix_corpus_terms_normalized_term", table_name="corpus_terms")
    op.drop_index("ix_corpus_terms_document_id", table_name="corpus_terms")
    op.drop_table("corpus_terms")
    op.drop_index("ix_documents_supersedes_document_id", table_name="documents")
    op.drop_constraint("ck_documents_authority_level", "documents", type_="check")
    op.drop_index("ix_documents_source_role", table_name="documents")
    op.drop_column("documents", "supersedes_document_id")
    op.drop_column("documents", "authority_level")
    op.drop_column("documents", "source_role")
