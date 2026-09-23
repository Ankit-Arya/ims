from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SharedUserOut(BaseModel):
    id: UUID
    username: str


class DocumentOut(BaseModel):
    id: UUID
    family_key: str | None
    title: str
    original_filename: str
    revision: str | None
    authority: str | None
    source_role: str | None
    authority_level: int | None
    supersedes_document_id: UUID | None
    department: str | None
    workspace_scope: str
    access_scope: str
    allowed_roles: list[str]
    allowed_departments: list[str]
    effective_from: date | None
    effective_to: date | None
    lifecycle_status: str
    ingestion_status: str
    ingestion_error: str | None
    page_count: int | None
    created_by: UUID
    can_manage: bool = False
    can_share: bool = False
    is_owner: bool = False
    shared_with: list[SharedUserOut] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentAuthorityUpdate(BaseModel):
    source_role: str | None = Field(default=None, max_length=64)
    authority_level: int | None = Field(default=None, ge=0, le=100)
    supersedes_document_id: UUID | None = None


class DocumentShareUpdate(BaseModel):
    user_ids: list[UUID] = Field(default_factory=list, max_length=100)
