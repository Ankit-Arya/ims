"""Add personal/shared document workspaces and report progress metadata."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_personal_workspaces"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("workspace_scope", sa.String(32), nullable=False, server_default="organization"),
    )
    op.create_check_constraint(
        "ck_documents_workspace_scope",
        "documents",
        "workspace_scope IN ('organization','personal')",
    )
    op.create_index("ix_documents_workspace_scope", "documents", ["workspace_scope"])

    op.create_table(
        "document_shares",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("shared_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "user_id", name="uq_document_share_user"),
    )
    op.create_index("ix_document_shares_document_id", "document_shares", ["document_id"])
    op.create_index("ix_document_shares_user_id", "document_shares", ["user_id"])

    op.add_column("report_jobs", sa.Column("progress_stage", sa.String(64), nullable=True))
    op.add_column("report_jobs", sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("report_jobs", sa.Column("progress_message", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("report_jobs", "progress_message")
    op.drop_column("report_jobs", "progress_percent")
    op.drop_column("report_jobs", "progress_stage")
    op.drop_index("ix_document_shares_user_id", table_name="document_shares")
    op.drop_index("ix_document_shares_document_id", table_name="document_shares")
    op.drop_table("document_shares")
    op.drop_index("ix_documents_workspace_scope", table_name="documents")
    op.drop_constraint("ck_documents_workspace_scope", "documents", type_="check")
    op.drop_column("documents", "workspace_scope")
