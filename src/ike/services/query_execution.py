import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.agent.controller import AgenticQAService
from ike.core.config import get_settings
from ike.db.models import Document, QueryLog, User
from ike.retrieval.access import document_access_clause
from ike.schemas.query import Citation, QueryRequest, QueryResponse
from ike.services.query_metrics import observe_trace
from ike.services.visuals import resolve_visual_evidence
from ike.workflows.qa_graph import QAGraphService

ProgressFn = Callable[[str, str, str, int], None]
AnswerDeltaFn = Callable[[str], None]
CancelCheckFn = Callable[[], None]

logger = logging.getLogger(__name__)


def _validate_scope(db: Session, user: User, document_ids: list[UUID] | None) -> None:
    if not document_ids:
        return
    accessible_ids = set(
        db.scalars(
            select(Document.id).where(Document.id.in_(document_ids), document_access_clause(user))
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
    _validate_scope(db, user, payload.document_ids)
    started = time.perf_counter()
    if cancel_check:
        cancel_check()
    settings = get_settings()
    operational_context = (
        payload.operational_context.model_dump(exclude_none=True)
        if payload.operational_context
        else {}
    )
    state: dict
    if settings.agentic_qa_enabled and payload.experience == "qa":
        try:
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
                payload.mode,
                payload.document_ids,
                experience=payload.experience,
                operational_context=operational_context,
            )
        except Exception as exc:
            if not settings.agentic_qa_fallback_enabled:
                raise
            logger.exception("agentic_qa_failed_falling_back")
            workflow = QAGraphService(
                db,
                user,
                progress=progress,
                answer_delta=answer_delta,
                cancel_check=cancel_check,
                request_id=str(payload.request_id),
            )
            state = workflow.run(
                payload.question,
                payload.mode,
                payload.document_ids,
                experience=payload.experience,
                operational_context=operational_context,
            )
            trace = dict(state.get("retrieval_trace", {}))
            trace["agentic_fallback"] = True
            trace["agentic_fallback_error"] = type(exc).__name__
            state["retrieval_trace"] = trace
    else:
        workflow = QAGraphService(
            db,
            user,
            progress=progress,
            answer_delta=answer_delta,
            cancel_check=cancel_check,
            request_id=str(payload.request_id),
        )
        state = workflow.run(
            payload.question,
            "direct" if payload.experience == "realtime" else payload.mode,
            payload.document_ids,
            experience=payload.experience,
            operational_context=operational_context,
        )
    latency_ms = int((time.perf_counter() - started) * 1000)

    by_id = {e.evidence_id: e for e in state.get("evidence", [])}
    citations: list[dict] = []
    for evidence_id in state.get("cited_ids", []):
        evidence = by_id.get(evidence_id)
        if not evidence:
            continue
        c = evidence.candidate
        citations.append(
            {
                "evidence_id": evidence_id,
                "document_id": str(c.document_id),
                "document_title": c.document_title,
                "filename": c.filename,
                "page_from": c.page_from,
                "page_to": c.page_to,
                "section_path": c.section_path,
                "chunk_id": str(c.chunk_id),
                "excerpt": c.text[:650],
            }
        )
    cited_evidence = [by_id[eid] for eid in state.get("cited_ids", []) if eid in by_id]
    visuals = resolve_visual_evidence(
        db, user, payload.question, cited_evidence
    )

    confidence = state.get("confidence", "low")
    if state.get("answer") and not citations:
        confidence = "low"

    requested_mode = "direct" if payload.experience == "realtime" or payload.mode == "qa" else payload.mode
    resolved_mode = state.get("resolved_mode", requested_mode)
    route_reason = state.get("route_reason")
    trace = dict(state.get("retrieval_trace", {}))
    query_plan = state.get("query_plan")
    query_frame = state.get("query_frame")
    evidence_plan = state.get("evidence_plan")
    goal_satisfaction = state.get("goal_satisfaction")
    answer_plan = state.get("answer_plan")
    source_policy = state.get("source_policy")
    trace.update(
        {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "route_reason": route_reason,
            "confidence": confidence,
            "verified": state.get("verified"),
            "workflow_timings_ms": state.get("workflow_timings_ms", {}),
            "query_plan": asdict(query_plan) if query_plan is not None and is_dataclass(query_plan) else None,
            "query_frame": (
                query_frame.model_dump()
                if query_frame is not None and hasattr(query_frame, "model_dump")
                else trace.get("query_frame")
            ),
            "evidence_plan": asdict(evidence_plan) if evidence_plan is not None and is_dataclass(evidence_plan) else trace.get("evidence_plan"),
            "goal_satisfaction": asdict(goal_satisfaction) if goal_satisfaction is not None and is_dataclass(goal_satisfaction) else trace.get("goal_satisfaction"),
            "recovery_attempted": bool(state.get("recovery_attempted", False)),
            "answer_plan": asdict(answer_plan) if answer_plan is not None and is_dataclass(answer_plan) else None,
            "corpus_discovery": state.get("corpus_discovery"),
            "okf_resolution": state.get("okf_resolution"),
            "source_policy": (
                source_policy.as_dict()
                if source_policy is not None and hasattr(source_policy, "as_dict")
                else trace.get("source_policy")
            ),
            "governing_reference_short_circuit": False,
            "visual_evidence_count": len(visuals),
            "cross_query_context_used": False,
            "experience": payload.experience,
            "operational_context": payload.operational_context.model_dump(exclude_none=True) if payload.operational_context else {},
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
            "The completed answer is being stored in your private history with its retrieval audit trace.",
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
        progress("complete", "Answer ready", "Grounded answer and sources are ready to review.", 100)

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
        debug_download_available=get_settings().query_debug_download_enabled,
    )
