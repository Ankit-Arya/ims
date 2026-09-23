from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ReportCreate(BaseModel):
    title: str = Field(min_length=2, max_length=500)
    objective: str = Field(min_length=5, max_length=10000)
    document_ids: list[UUID] = Field(min_length=1, max_length=50)


class ReportOut(BaseModel):
    id: UUID
    title: str
    objective: str
    document_ids: list[UUID]
    status: str
    progress_stage: str | None
    progress_percent: int
    progress_message: str | None
    result_markdown: str | None
    citations: list[dict]
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
