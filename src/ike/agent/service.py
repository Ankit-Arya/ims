from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.agent.answer import EvidenceAnswerAgent
from ike.agent.query_intelligence import QueryIntelligenceAgent
from ike.agent.research import ResearchBundle, ResearchExecutor
from ike.core.config import get_settings
from ike.db.models import QueryLog, User
from ike.retrieval.types import Evidence
from ike.services.llm import LLMClient

ProgressFn = Callable[[str, str, str, int], None]
AnswerDeltaFn = Callable[[str], None]
CancelCheckFn = Callable[[], None]


class AgenticQAService:
    """Planned AI research pipeline.

    Query Intelligence Agent -> bounded fast retrieval -> Evidence/Answer Agent ->
    optional one targeted gap round -> final grounded answer.
    """

    def __init__(
        self,
        db: Session,
        user: User,
        *,
        progress: ProgressFn | None = None,
        answer_delta: AnswerDeltaFn | None = None,
        cancel_check: CancelCheckFn | None = None,
        request_id: str | None = None,
    ) -> None:
        self.db = db
        self.user = user
        self.settings = get_settings()
        self.llm = LLMClient()
        self.progress = progress
        self.answer_delta = answer_delta
        self.cancel_check = cancel_check
        self.request_id = request_id

    def _checkpoint(self) -> None:
        if self.cancel_check:
            self.cancel_check()

    def _emit(
        self,
        stage: str,
        label: str,
        detail: str,
        percent: int,
    ) -> None:
        if self.progress:
            self.progress(stage, label, detail, percent)

    def _recent_history(self) -> list[dict]:
        rows = list(
            self.db.scalars(
                select(QueryLog)
                .where(QueryLog.user_id == self.user.id)
                .order_by(QueryLog.created_at.desc())
                .limit(self.settings.agent_recent_history_count)
            ).all()
        )
        return [
            {
                "query_id": str(row.id),
                "created_at": (
                    row.created_at.isoformat()
                    if row.created_at is not None
                    else None
                ),
                "question": row.question,
                "answer_excerpt": (row.answer or "")[
                    : self.settings.agent_recent_history_answer_chars
                ],
            }
            for row in rows
        ]

    @staticmethod
    def _selected_evidence(bundle, evidence_ids: list[str]) -> list[Evidence]:
        selected: list[Evidence] = []
        seen: set[str] = set()
        for evidence_id in evidence_ids:
            if evidence_id in seen:
                continue
            candidate = bundle.candidate_for_evidence(evidence_id)
            if candidate is None:
                continue
            seen.add(evidence_id)
            selected.append(
                Evidence(
                    evidence_id=evidence_id,
                    candidate=candidate,
                )
            )
        return selected

    def _failure_result(
        self,
        *,
        message: str,
        started: float,
        trace: dict,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        trace["workflow_timings_ms"] = {
            **trace.get("workflow_timings_ms", {}),
            "agent_total": elapsed_ms,
        }
        trace["agent_stop_reason"] = "pipeline_failure"
        return {
            "answer": message,
            "evidence": [],
            "cited_ids": [],
            "confidence": "low",
            "resolved_mode": "direct",
            "route_reason": "planned_ai_research_v5_1",
            "retrieval_trace": trace,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "verified": False,
            "workflow_timings_ms": trace["workflow_timings_ms"],
            "recovery_attempted": False,
            "query_plan": None,
            "query_frame": None,
            "evidence_plan": None,
            "goal_satisfaction": None,
            "answer_plan": None,
            "source_policy": None,
            "corpus_discovery": None,
            "okf_resolution": None,
        }

    def run(
        self,
        question: str,
        requested_mode: str,
        document_ids: list[UUID] | None,
        *,
        experience: str = "qa",
        operational_context: dict | None = None,
    ) -> dict:
        started = time.perf_counter()
        operational_context = dict(operational_context or {})
        recent_history = self._recent_history()
        input_tokens = 0
        output_tokens = 0
        timings: dict[str, object] = {}
        trace: dict = {
            "agentic": True,
            "agent_architecture": "planned_ai_research_v5_1",
            "planner_model": self.settings.llm_strong_model,
            "planner_reasoning": self.settings.agent_planner_reasoning,
            "answer_model": self.settings.llm_strong_model,
            "answer_reasoning": self.settings.agent_answer_reasoning,
            "recent_context_query_ids": [
                item["query_id"] for item in recent_history
            ],
        }

        self._checkpoint()
        self._emit(
            "planning",
            "Understanding the complete question",
            "AI is identifying the facts, conditions and relationships that must be answered.",
            10,
        )

        planner = QueryIntelligenceAgent(self.llm)
        plan_started = time.perf_counter()
        try:
            plan, in_tokens, out_tokens = planner.plan(
                question,
                operational_context=operational_context,
                selected_document_ids=document_ids,
                recent_history=recent_history,
            )
        except Exception:
            return self._failure_result(
                message=(
                    "I couldn't complete the query-understanding step reliably. "
                    "Please retry the question."
                ),
                started=started,
                trace=trace,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        timings["planning"] = int((time.perf_counter() - plan_started) * 1000)
        input_tokens += in_tokens
        output_tokens += out_tokens
        trace["research_plan"] = plan.model_dump()

        executor = ResearchExecutor(
            self.db,
            self.user,
            request_id=self.request_id,
            allowed_document_ids=document_ids,
            cancel_check=self.cancel_check,
        )
        bundle = ResearchBundle()

        if plan.needs_corpus:
            self._checkpoint()
            self._emit(
                "research",
                "Searching the knowledge base",
                (
                    f"AI planned {len(plan.tasks)} focused research task"
                    f"{'s' if len(plan.tasks) != 1 else ''}."
                ),
                28,
            )
            research_started = time.perf_counter()
            bundle = executor.execute(
                plan.tasks,
                round_name="primary",
                time_budget_seconds=self.settings.agent_primary_research_time_seconds,
            )
            timings["primary_research"] = int(
                (time.perf_counter() - research_started) * 1000
            )
        else:
            timings["primary_research"] = 0

        self._checkpoint()
        self._emit(
            "reasoning",
            "Reasoning across the evidence",
            "AI is checking every requested part and how the retrieved facts relate.",
            62,
        )

        answer_agent = EvidenceAnswerAgent(self.llm)
        answer_started = time.perf_counter()
        try:
            decision, in_tokens, out_tokens = answer_agent.decide(
                question=question,
                operational_context=operational_context,
                recent_history=recent_history,
                plan=plan,
                bundle=bundle,
            )
        except Exception:
            trace.update(
                {
                    "research_observations": bundle.observations,
                    "tool_calls": bundle.traces,
                    "skipped_task_ids": bundle.skipped_task_ids,
                }
            )
            return self._failure_result(
                message=(
                    "I found source material but couldn't complete the answer reasoning "
                    "reliably. Please retry the question."
                ),
                started=started,
                trace=trace,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        timings["evidence_reasoning"] = int(
            (time.perf_counter() - answer_started) * 1000
        )
        input_tokens += in_tokens
        output_tokens += out_tokens
        trace["first_answer_decision"] = decision.model_dump()

        final_answer = decision.answer
        selected_ids = decision.selected_evidence_ids
        confidence = decision.confidence

        if decision.status == "needs_evidence":
            gap_tasks = decision.gap_tasks[: self.settings.agent_max_gap_tasks]
            self._emit(
                "gap_research",
                "Checking one remaining gap",
                (
                    "AI identified a specific missing point and is running a targeted "
                    "follow-up search."
                ),
                74,
            )
            gap_started = time.perf_counter()
            if gap_tasks:
                bundle = executor.execute(
                    gap_tasks,
                    round_name="gap",
                    time_budget_seconds=self.settings.agent_gap_research_time_seconds,
                )
            timings["gap_research"] = int(
                (time.perf_counter() - gap_started) * 1000
            )

            self._checkpoint()
            final_started = time.perf_counter()
            try:
                final_decision, in_tokens, out_tokens = answer_agent.finalize(
                    question=question,
                    operational_context=operational_context,
                    recent_history=recent_history,
                    plan=plan,
                    bundle=bundle,
                )
            except Exception:
                trace.update(
                    {
                        "research_observations": bundle.observations,
                        "tool_calls": bundle.traces,
                        "skipped_task_ids": bundle.skipped_task_ids,
                    }
                )
                return self._failure_result(
                    message=(
                        "I couldn't complete the final answer reliably after the targeted "
                        "follow-up search. Please retry the question."
                    ),
                    started=started,
                    trace=trace,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            timings["final_reasoning"] = int(
                (time.perf_counter() - final_started) * 1000
            )
            input_tokens += in_tokens
            output_tokens += out_tokens
            trace["final_answer_decision"] = final_decision.model_dump()
            final_answer = final_decision.answer
            selected_ids = final_decision.selected_evidence_ids
            confidence = final_decision.confidence

        evidence = self._selected_evidence(
            bundle,
            selected_ids[: self.settings.agent_max_evidence],
        )
        cited_ids = [item.evidence_id for item in evidence]
        if final_answer and not evidence and plan.needs_corpus:
            confidence = "low"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        timings["agent_total"] = elapsed_ms
        trace.update(
            {
                "research_observations": bundle.observations,
                "tool_calls": bundle.traces,
                "completed_task_ids": sorted(bundle.completed_task_ids),
                "skipped_task_ids": bundle.skipped_task_ids,
                "selected_evidence_ids": cited_ids,
                "selected_chunk_ids": [
                    str(item.candidate.chunk_id) for item in evidence
                ],
                "agent_evidence_status": (
                    "conversation_context"
                    if not plan.needs_corpus
                    else "sufficient"
                    if evidence and confidence in {"high", "medium"}
                    else "partial"
                ),
                "agent_stop_reason": (
                    "answered_after_targeted_gap"
                    if decision.status == "needs_evidence"
                    else "answered_from_primary_research"
                ),
                "workflow_timings_ms": timings,
            }
        )

        self._emit(
            "answer",
            "Answer ready",
            "The answer has been grounded in the selected documentary evidence.",
            92,
        )

        return {
            "answer": final_answer,
            "evidence": evidence,
            "cited_ids": cited_ids,
            "confidence": confidence,
            "resolved_mode": (
                "research" if requested_mode == "research" else "direct"
            ),
            "route_reason": "planned_ai_research_v5_1",
            "retrieval_trace": trace,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "verified": bool(evidence) or not plan.needs_corpus,
            "workflow_timings_ms": timings,
            "recovery_attempted": decision.status == "needs_evidence",
            "query_plan": plan.model_dump(),
            "query_frame": None,
            "evidence_plan": None,
            "goal_satisfaction": None,
            "answer_plan": None,
            "source_policy": None,
            "corpus_discovery": None,
            "okf_resolution": None,
        }
