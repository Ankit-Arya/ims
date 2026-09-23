from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class VisualEvidence(BaseModel):
    document_id: UUID
    document_title: str
    page: int
    kind: Literal["figure", "diagram", "table", "page"] = "page"
    caption: str
    token: str
    evidence_id: str | None = None
