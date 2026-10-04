from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ike.schemas.visuals import VisualEvidence


class OperationalContext(BaseModel):
    line: str | None = Field(default=None, max_length=40)
    rolling_stock: str | None = Field(default=None, max_length=80)
    department: str | None = Field(default=None, max_length=120)
    role: str | None = Field(default=None, max_length=80)
    operating_mode: str | None = Field(default=None, max_length=80)
    location_type: str | None = Field(default=None, max_length=80)
    station_or_section: str | None = Field(default=None, max_length=160)

    def prompt_block(self) -> str:
        labels = {
            "line": "Line", "rolling_stock": "Rolling stock", "department": "Department",
            "role": "Operational role", "operating_mode": "Operating mode",
            "location_type": "Location type", "station_or_section": "Station/section",
        }
        values = self.model_dump()
        return "; ".join(f"{labels[key]}={value}" for key, value in values.items() if value)


class QueryRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=2, max_length=6000)
    experience: Literal["qa", "realtime"] = "qa"
    operational_context: OperationalContext | None = None
    # "qa" remains accepted for API compatibility with v0.1.x clients.
    mode: Literal["auto", "direct", "qa", "research"] = "auto"
    document_ids: list[UUID] | None = Field(default=None, max_length=50)


class Citation(BaseModel):
    evidence_id: str
    document_id: UUID
    document_title: str
    filename: str
    page_from: int | None
    page_to: int | None
    section_path: list[str]
    chunk_id: UUID
    excerpt: str


class QueryResponse(BaseModel):
    query_id: UUID
    mode: str
    resolved_mode: str
    route_reason: str | None = None
    answer: str
    citations: list[Citation]
    visuals: list[VisualEvidence] = Field(default_factory=list)
    confidence: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    debug_download_available: bool = False


class QueryHistoryItem(QueryResponse):
    question: str
    created_at: datetime


class FeedbackRequest(BaseModel):
    query_id: UUID
    rating: int = Field(ge=1, le=5)
    correct: bool | None = None
    complete: bool | None = None
    comment: str | None = Field(default=None, max_length=4000)
