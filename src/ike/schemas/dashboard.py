from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class RecentQuestionOut(BaseModel):
    query_id: UUID
    question: str
    mode: str
    resolved_mode: str
    confidence: str
    created_at: datetime


class DashboardSummaryOut(BaseModel):
    question_count: int
    deep_analysis_count: int
    available_pdf_count: int
    knowledge_pdf_count: int
    personal_pdf_count: int
    ready_pdf_count: int
    indexing_pdf_count: int
    failed_pdf_count: int
    recent_questions: list[RecentQuestionOut]
