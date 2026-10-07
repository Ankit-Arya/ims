import time
from collections.abc import Callable
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.agent.service import AgenticQAService
from ike.core.config import get_settings
from ike.db.models import Document, QueryLog, User
from ike.retrieval.access import document_access_clause
from ike.schemas.query import Citation, QueryRequest, QueryResponse
from ike.services.query_metrics import observe_trace
from ike.services.visuals import resolve_visual_evidence

ProgressFn = Callable[[str, str, str, int], None]
AnswerDeltaFn = Callable[[str], None]
CancelCheckFn = Callable[[], None]


def _validate_scope(
    db: Session,
    user: User,
    document_ids: list[UUID] | None,
) -> None:
    if not document_ids:
        return
    accessible_ids = set(
        db.scalars(
            select(Document.id).where(
                Document.id.in_(document_ids),
                document_access_clause(user),
            )
        ).all()
    )
    if accessible_ids != set(document_ids):
        raise HTTPException(
            status_code=403,
            detail=(
                "One or more selected documents are unavailable, inactive, not ready, "
                "not effective, or inaccessible"
            ),
        )


def execute_query(
    db: Session,
    user: User,
    payload: QueryRequest,
    *,
    progress: ProgressFn | None = None,
    answer_delta: AnswerDeltaFn | None = None,
    cancel_check: CancelCheckFn | None = None,
) -> QueryResponse:
    """Execute one Q&A request through the single v5 planned-AI research runtime."""

    _validate_scope(db, user, payload.document_ids)
    settings = get_settings()
    if not settings.agentic_qa_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agentic Q&A is disabled by configuration.",
        )

    started = time.perf_counter()
    if cancel_check:
        cancel_check()

    operational_context = (
        payload.operational_context.model_dump(exclude_none=True)
        if payload.operational_context
        else {}
    )
    requested_mode = (
        "direct" if payload.experience == "realtime" or payload.mode == "qa" else payload.mode
    )

    agent = AgenticQAService(
        db,
        user,
        progress=progress,
        answer_delta=answer_delta,
        cancel_check=cancel_check,
        request_id=str(payload.request_id),
    )
    state = agent.run(
        payload.question,
        requested_mode,
        payload.document_ids,
        experience=payload.experience,
        operational_context=operational_context,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    by_id = {evidence.evidence_id: evidence for evidence in state.get("evidence", [])}
    citations: list[dict] = []
    for evidence_id in state.get("cited_ids", []):
        evidence = by_id.get(evidence_id)
        if evidence is None:
            continue
        candidate = evidence.candidate
        citations.append(
            {
                "evidence_id": evidence_id,
                "document_id": str(candidate.document_id),
                "document_title": candidate.document_title,
                "filename": candidate.filename,
                "page_from": candidate.page_from,
                "page_to": candidate.page_to,
                "section_path": candidate.section_path,
                "chunk_id": str(candidate.chunk_id),
                "excerpt": candidate.text[:650],
            }
        )

    cited_evidence = [
        by_id[evidence_id] for evidence_id in state.get("cited_ids", []) if evidence_id in by_id
    ]
    visuals = resolve_visual_evidence(
        db,
        user,
        payload.question,
        cited_evidence,
    )

    confidence = str(state.get("confidence") or "low")
    evidence_status = str(
        state.get("retrieval_trace", {}).get("agent_evidence_status") or ""
    )
    if (
        state.get("answer")
        and not citations
        and evidence_status != "conversation_context"
    ):
        confidence = "low"

    resolved_mode = str(state.get("resolved_mode") or requested_mode)
    route_reason = state.get("route_reason")
    trace = dict(state.get("retrieval_trace", {}))
    trace.update(
        {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "route_reason": route_reason,
            "confidence": confidence,
            "verified": bool(state.get("verified")),
            "workflow_timings_ms": state.get("workflow_timings_ms", {}),
            "visual_evidence_count": len(visuals),
            "experience": payload.experience,
            "operational_context": operational_context,
        }
    )

    observe_trace(
        resolved_mode=resolved_mode,
        total_ms=latency_ms,
        retrieval_trace=trace,
        workflow_timings=state.get("workflow_timings_ms", {}),
    )

    if progress:
        progress(
            "save",
            "Saving this Q&A to your history",
            (
                "The completed answer is being stored in your private history "
                "with its retrieval audit trace."
            ),
            97,
        )

    log = QueryLog(
        user_id=user.id,
        question=payload.question,
        mode=requested_mode,
        answer=state["answer"],
        citations=citations,
        retrieval_trace=trace,
        latency_ms=latency_ms,
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    if progress:
        progress(
            "complete",
            "Answer ready",
            "Grounded answer and sources are ready to review.",
            100,
        )

    return QueryResponse(
        query_id=log.id,
        mode=requested_mode,
        resolved_mode=resolved_mode,
        route_reason=route_reason,
        answer=state["answer"],
        citations=[Citation(**item) for item in citations],
        visuals=visuals,
        confidence=confidence,
        latency_ms=latency_ms,
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        debug_download_available=settings.query_debug_download_enabled,
    )
