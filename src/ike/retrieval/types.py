from dataclasses import dataclass, field
from uuid import UUID


@dataclass(slots=True)
class Candidate:
    chunk_id: UUID
    document_id: UUID
    ordinal: int
    page_from: int | None
    page_to: int | None
    section_path: list[str]
    content_kind: str
    text: str
    contextual_text: str
    document_title: str
    filename: str
    revision: str | None
    authority: str | None
    family_key: str | None = None
    source_metadata: dict = field(default_factory=dict)
    document_profile: dict = field(default_factory=dict)
    evidence_lane: str = "operational"
    applicability_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_retrieval_score: float = 0.0
    rank_method: str = "unranked"
    judge_score: float = 0.0
    judge_details: dict[str, float] = field(default_factory=dict)
    goal_rerank_scores: dict[str, float] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    candidate: Candidate

    def prompt_block(self) -> str:
        page = "unknown" if self.candidate.page_from is None else str(self.candidate.page_from)
        if self.candidate.page_to and self.candidate.page_to != self.candidate.page_from:
            page = f"{page}-{self.candidate.page_to}"
        section = " / ".join(self.candidate.section_path) or "Unsectioned"
        authority = self.candidate.authority or "unspecified"
        revision = self.candidate.revision or "unspecified"
        from ike.retrieval.table_context import retrieval_text

        content = retrieval_text(self.candidate)
        return (
            f"[{self.evidence_id}] Evidence lane: {self.candidate.evidence_lane}\n"
            f"Document: {self.candidate.document_title}\n"
            f"File: {self.candidate.filename}\nPage: {page}\nSection: {section}\n"
            f"Revision: {revision}\nAuthority: {authority}\n"
            f"Document family: {self.candidate.family_key or 'unspecified'}\n"
            f"Content:\n{content}"
        )
