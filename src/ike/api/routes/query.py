import json
import logging
import queue
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user
from ike.db.models import Feedback, QueryLog, User
from ike.db.session import SessionLocal, get_db
from ike.schemas.query import FeedbackRequest, QueryHistoryItem, QueryRequest, QueryResponse
from ike.services.query_admission import QueryCapacityError, query_admission
from ike.services.query_execution import execute_query
from ike.services.query_control import QueryCancelled, QueryControl
from ike.services.inference_client import InferenceClient

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


@router.post("", response_model=QueryResponse)
def ask(
    payload: QueryRequest,
    user: User = Depends(get_current_user),
) -> QueryResponse:
    user_id = user.id
    control = QueryControl(payload.request_id, user_id)
    control.clear()

    def run_query() -> QueryResponse:
        with SessionLocal() as db:
            current_user = db.get(User, user_id)
            if not current_user or not current_user.is_active:
                raise HTTPException(status_code=403, detail="Your user account is no longer active.")
            return execute_query(db, current_user, payload, cancel_check=control.checkpoint)

    try:
        future = query_admission.submit(run_query)
    except QueryCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return future.result()


@router.post("/stream")
def ask_stream(payload: QueryRequest, user: User = Depends(get_current_user)) -> StreamingResponse:
    """Stream truthful backend progress as newline-delimited JSON.

    Q&A remains synchronous from a product perspective, but the workflow runs in a
    bounded per-replica execution pool with its own database session. The browser sees
    admission, retrieval and model stages while the request is still running.
    """

    user_id = user.id
    control = QueryControl(payload.request_id, user_id)
    control.clear()

    def events():
        messages: queue.Queue[dict | None] = queue.Queue()

        def progress(stage: str, label: str, detail: str, percent: int) -> None:
            messages.put(
                {
                    "type": "progress",
                    "stage": stage,
                    "label": label,
                    "detail": detail,
                    "percent": percent,
                }
            )

        def answer_delta(delta: str) -> None:
            messages.put({"type": "answer_delta", "delta": delta})

        def run_query() -> None:
            try:
                with SessionLocal() as db:
                    current_user = db.get(User, user_id)
                    if not current_user or not current_user.is_active:
                        messages.put({"type": "error", "message": "Your user account is no longer active."})
                        return
                    result = execute_query(
                        db, current_user, payload, progress=progress,
                        answer_delta=answer_delta, cancel_check=control.checkpoint,
                    )
                    messages.put({"type": "result", "data": result.model_dump(mode="json")})
            except QueryCancelled:
                messages.put({"type": "cancelled", "request_id": str(payload.request_id)})
            except HTTPException as exc:
                messages.put({"type": "error", "message": str(exc.detail), "status": exc.status_code})
            except Exception:
                logger.exception("streamed_query_failed", extra={"user_id": str(user_id)})
                messages.put(
                    {
                        "type": "error",
                        "message": "The query could not be completed. Check server diagnostics or retry.",
                        "status": 500,
                    }
                )
            finally:
                if not control.is_cancelled():
                    control.clear()
                messages.put(None)

        def on_wait(wait_ms: int) -> None:
            if wait_ms > 25:
                progress(
                    "queue",
                    "Waiting for Q&A capacity",
                    f"Request admitted after {wait_ms / 1000:.1f}s. IMS bounds concurrent Q&A so bursts do not overload local inference.",
                    3,
                )

        progress(
            "queue",
            "Entering the Q&A queue",
            "IMS bounds expensive online work so bursts do not overwhelm the local query inference service.",
            2,
        )
        try:
            query_admission.submit(run_query, on_wait=on_wait)
        except QueryCapacityError as exc:
            yield json.dumps({"type": "error", "message": str(exc), "status": 429}) + "\n"
            return
        while True:
            try:
                item = messages.get(timeout=12)
            except queue.Empty:
                yield json.dumps({"type": "heartbeat"}) + "\n"
                continue
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )



@router.post("/{request_id}/cancel")
def cancel_query(request_id: UUID, user: User = Depends(get_current_user)) -> dict:
    # Cancellation is scoped to an opaque request UUID. It does not expose data and can be
    # honored by whichever API replica currently owns the work because Valkey is shared.
    control = QueryControl(request_id, user.id)
    control.cancel()
    # Best-effort signal to the local cross-encoder so long reranks stop between batches.
    InferenceClient.for_query().cancel(str(request_id))
    return {"status": "cancel_requested", "request_id": str(request_id)}


@router.get("/history", response_model=list[QueryHistoryItem])
def query_history(
    limit: int = Query(default=30, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[QueryHistoryItem]:
    rows = list(
        db.scalars(
            select(QueryLog)
            .where(QueryLog.user_id == user.id)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
        ).all()
    )
    history: list[QueryHistoryItem] = []
    for row in rows:
        trace = row.retrieval_trace or {}
        history.append(
            QueryHistoryItem(
                query_id=row.id,
                question=row.question,
                mode=row.mode,
                resolved_mode=str(trace.get("resolved_mode") or row.mode),
                route_reason=trace.get("route_reason"),
                answer=row.answer,
                citations=row.citations or [],
                confidence=str(trace.get("confidence") or "low"),
                latency_ms=row.latency_ms,
                input_tokens=row.input_tokens or 0,
                output_tokens=row.output_tokens or 0,
                created_at=row.created_at,
            )
        )
    # API returns chronological order so the browser can render a natural transcript.
    history.reverse()
    return history


@router.post("/feedback", status_code=201)
def submit_feedback(
    payload: FeedbackRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    query_log = db.get(QueryLog, payload.query_id)
    if not query_log:
        raise HTTPException(status_code=404, detail="Query not found")
    if query_log.user_id != user.id:
        raise HTTPException(status_code=403, detail="You may only rate your own query")
    feedback = Feedback(
        query_log_id=query_log.id,
        user_id=user.id,
        rating=payload.rating,
        correct=payload.correct,
        complete=payload.complete,
        comment=payload.comment,
    )
    db.add(feedback)
    db.commit()
    return {"status": "recorded"}
