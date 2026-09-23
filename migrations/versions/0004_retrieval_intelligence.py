"""Add the lightweight corpus-intelligence index used by IMS 0.9.0.

The migration is additive. It does not rewrite chunks or their embeddings.  Retrieval
nodes are backfilled separately from the already-ingested corpus and can be rebuilt at
any time without changing source documents.
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision = "0004_retrieval_intelligence"
down_revision = "0003_query_performance_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("node_type", sa.String(32), nullable=False),
        sa.Column("node_key", sa.String(320), nullable=False),
        sa.Column("label", sa.String(1000), nullable=False),
        sa.Column("section_path", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column("ordinal_from", sa.Integer(), nullable=True),
        sa.Column("ordinal_to", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(1024), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(label, '') || ' ' || coalesce(text, ''))",
                persisted=True,
            ),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "node_key", name="uq_retrieval_node_document_key"),
        sa.CheckConstraint(
            "node_type IN ('document','section','concept')",
            name="ck_retrieval_nodes_type",
        ),
    )
    op.create_index("ix_retrieval_nodes_document_id", "retrieval_nodes", ["document_id"])
    op.create_index("ix_retrieval_nodes_node_type", "retrieval_nodes", ["node_type"])
    op.execute("CREATE INDEX ix_retrieval_nodes_search_vector ON retrieval_nodes USING gin (search_vector)")
    op.execute(
        "CREATE INDEX ix_retrieval_nodes_embedding_hnsw ON retrieval_nodes "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 96)"
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_nodes_embedding_hnsw", table_name="retrieval_nodes")
    op.drop_index("ix_retrieval_nodes_search_vector", table_name="retrieval_nodes")
    op.drop_index("ix_retrieval_nodes_node_type", table_name="retrieval_nodes")
    op.drop_index("ix_retrieval_nodes_document_id", table_name="retrieval_nodes")
    op.drop_table("retrieval_nodes")
