"""Add exact-search and provenance indexes for interactive Q&A.

This migration is additive. It does not rewrite chunks or embeddings and is safe for
existing 0.4.x corpora. Trigram indexing accelerates the existing leading-wildcard
ILIKE access pattern used by exact/role coverage retrieval.
"""

from alembic import op

revision = "0003_query_performance_indexes"
down_revision = "0002_personal_workspaces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # CREATE INDEX CONCURRENTLY cannot run inside Alembic's normal transaction.
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_contextual_text_trgm "
            "ON chunks USING gin (contextual_text gin_trgm_ops)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_document_page_range "
            "ON chunks (document_id, page_from, page_to)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_query_logs_user_created_at "
            "ON query_logs (user_id, created_at DESC)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_query_logs_user_created_at")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_document_page_range")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_contextual_text_trgm")
