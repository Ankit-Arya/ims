from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


QUERY_ACTIVE = Gauge("ike_qa_active", "Q&A requests currently executing", ["api_instance"])
QUERY_WAITING = Gauge("ike_qa_waiting", "Q&A requests waiting for API execution capacity", ["api_instance"])
QUERY_TOTAL = Histogram("ike_qa_total_duration_seconds", "Total Q&A latency", ["resolved_mode"])
QUERY_ADMISSION_WAIT = Histogram(
    "ike_qa_admission_wait_seconds",
    "Time a Q&A request waited for an API execution slot",
    ["api_instance"],
)
QUERY_STAGE = Histogram(
    "ike_qa_stage_duration_seconds",
    "Q&A stage latency including explicit inference queue/execution breakdowns",
    ["stage", "resolved_mode"],
)

QUERY_COMPOSITIONAL = Counter(
    "ike_qa_compositional_total",
    "Compositional Q&A outcomes",
    ["strategy", "goal_status", "recovery"],
)
QUERY_GOAL_COUNT = Histogram(
    "ike_qa_evidence_goal_count",
    "Number of evidence goals planned for one Q&A request",
    buckets=(1, 2, 3, 4, 6, 8, 12),
)


def observe_trace(*, resolved_mode: str, total_ms: int, retrieval_trace: dict, workflow_timings: dict) -> None:
    QUERY_TOTAL.labels(resolved_mode).observe(max(0, total_ms) / 1000)
    for stage, value in (retrieval_trace.get("timings_ms") or {}).items():
        if isinstance(value, (int, float)):
            QUERY_STAGE.labels(str(stage), resolved_mode).observe(max(0.0, float(value)) / 1000)
    for stage, value in (workflow_timings or {}).items():
        if isinstance(value, (int, float)):
            QUERY_STAGE.labels(str(stage), resolved_mode).observe(max(0.0, float(value)) / 1000)
    evidence_plan = retrieval_trace.get("evidence_plan") or {}
    if retrieval_trace.get("compositional") and isinstance(evidence_plan, dict):
        goals = evidence_plan.get("goals") or []
        QUERY_GOAL_COUNT.observe(len(goals))
        goal_status = "complete" if retrieval_trace.get("goal_complete") else "incomplete"
        recovery = "yes" if retrieval_trace.get("recovery_attempted") else "no"
        QUERY_COMPOSITIONAL.labels(
            str(evidence_plan.get("strategy") or "unknown"),
            goal_status,
            recovery,
        ).inc()
