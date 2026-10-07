from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.db.models import Chunk, Document, QueryLog, User
from ike.retrieval.access import document_access_clause


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _selected_evidence(
    db: Session,
    user: User,
    trace: dict,
) -> list[dict]:
    ordered_ids: list[UUID] = []
    for raw in _as_list(trace.get("selected_chunk_ids")):
        try:
            value = UUID(str(raw))
        except (TypeError, ValueError):
            continue
        if value not in ordered_ids:
            ordered_ids.append(value)
    if not ordered_ids:
        return []

    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.id.in_(ordered_ids),
            document_access_clause(user),
            Document.ingestion_status == "ready",
        )
    ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}

    result: list[dict] = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        pair = by_id.get(chunk_id)
        if pair is None:
            continue
        chunk, document = pair
        result.append(
            {
                "rank": rank,
                "chunk_id": str(chunk.id),
                "document_id": str(document.id),
                "document_title": document.title,
                "filename": document.original_filename,
                "page_from": chunk.page_from,
                "page_to": chunk.page_to,
                "section_path": chunk.section_path or [],
                "content_kind": chunk.content_kind,
                "snippet": " ".join((chunk.text or "").split())[:1800],
            }
        )
    return result


def _tool_summary(call: dict) -> dict:
    return {
        "round": call.get("round"),
        "task_id": call.get("task_id"),
        "purpose": call.get("purpose"),
        "depends_on": _as_list(call.get("depends_on")),
        "tool": call.get("tool"),
        "arguments": _as_dict(call.get("arguments")),
        "metadata": _as_dict(call.get("metadata")),
        "elapsed_ms": call.get("elapsed_ms"),
        "note": call.get("note"),
        "items": _as_list(call.get("items")),
    }


def build_query_debug_report(
    db: Session,
    user: User,
    log: QueryLog,
) -> dict:
    """Build a v5 planned-research diagnostic from the persisted query trace."""

    trace = _as_dict(log.retrieval_trace)
    tool_calls = [
        _tool_summary(value)
        for value in _as_list(trace.get("tool_calls"))
        if isinstance(value, dict)
    ]

    return {
        "debug_version": 5,
        "query_id": str(log.id),
        "created_at": log.created_at.isoformat() if log.created_at else None,
        "original_question": log.question,
        "mode": log.mode,
        "agentic": bool(trace.get("agentic")),
        "agent_architecture": trace.get("agent_architecture"),
        "planner_model": trace.get("planner_model"),
        "planner_reasoning": trace.get("planner_reasoning"),
        "answer_model": trace.get("answer_model"),
        "answer_reasoning": trace.get("answer_reasoning"),
        "research_plan": _as_dict(trace.get("research_plan")),
        "first_answer_decision": _as_dict(trace.get("first_answer_decision")),
        "final_answer_decision": _as_dict(trace.get("final_answer_decision")),
        "tool_calls": tool_calls,
        "completed_task_ids": _as_list(trace.get("completed_task_ids")),
        "skipped_task_ids": _as_list(trace.get("skipped_task_ids")),
        "selected_evidence_ids": _as_list(trace.get("selected_evidence_ids")),
        "selected_chunk_ids": _as_list(trace.get("selected_chunk_ids")),
        "selected_evidence": _selected_evidence(db, user, trace),
        "agent_evidence_status": trace.get("agent_evidence_status"),
        "agent_stop_reason": trace.get("agent_stop_reason"),
        "workflow_timings_ms": _as_dict(trace.get("workflow_timings_ms")),
        "final_answer": log.answer,
        "citations": log.citations or [],
        "notes": [
            "The Query Intelligence Agent creates the research graph.",
            "Python executes retrieval tasks and resource limits but does not decide semantic sufficiency.",
            "The Evidence/Answer Agent selects applicability, requests at most one targeted gap round, and writes the answer.",
            "Selected evidence is reloaded through the current user's ACL for this diagnostic.",
        ],
    }


def debug_report_markdown(report: dict) -> str:
    lines = [
        "# IMS Query Debug Report",
        "",
        "- Debug version: " + _cell(report.get("debug_version")),
        "- Query ID: " + _cell(report.get("query_id")),
        "- Created: " + _cell(report.get("created_at")),
        "- Mode: " + _cell(report.get("mode")),
        "",
        "## Original question",
        "",
        str(report.get("original_question") or ""),
        "",
        "## AI runtime",
        "",
        "- Architecture: " + _cell(report.get("agent_architecture")),
        "- Planner: "
        + _cell(report.get("planner_model"))
        + " / "
        + _cell(report.get("planner_reasoning")),
        "- Answer agent: "
        + _cell(report.get("answer_model"))
        + " / "
        + _cell(report.get("answer_reasoning")),
        "- Evidence status: " + _cell(report.get("agent_evidence_status")),
        "- Stop reason: " + _cell(report.get("agent_stop_reason")),
        "",
        "## Query Intelligence plan",
        "",
        "~~~json",
        json.dumps(
            report.get("research_plan"),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "~~~",
        "",
        "## Research task execution",
        "",
    ]

    tool_calls = report.get("tool_calls") or []
    if not tool_calls:
        lines.append("- No retrieval tasks were executed.")
    for index, call in enumerate(tool_calls, start=1):
        lines.extend(
            [
                f"### {index}. {_cell(call.get('task_id'))} / {_cell(call.get('tool'))}",
                "",
                "- Round: " + _cell(call.get("round")),
                "- Purpose: " + _cell(call.get("purpose")),
                "- Depends on: "
                + (
                    ", ".join(str(value) for value in call.get("depends_on") or [])
                    or "none"
                ),
                "- Arguments: " + _cell(call.get("arguments")),
                "- Elapsed: " + _cell(call.get("elapsed_ms")) + " ms",
                "- Metadata: " + _cell(call.get("metadata")),
                "- Note: " + _cell(call.get("note")),
            ]
        )
        items = call.get("items") or []
        if items:
            lines.extend(
                [
                    "",
                    "| # | Evidence | Document | Page | Section / label | Observation |",
                    "|---:|---|---|---:|---|---|",
                ]
            )
            for item_index, item in enumerate(items[:40], start=1):
                if not isinstance(item, dict):
                    continue
                section_path = item.get("section_path") or []
                section = (
                    " > ".join(str(value) for value in section_path)
                    if section_path
                    else item.get("label")
                )
                observation = (
                    item.get("snippet")
                    or item.get("title")
                    or item.get("filename")
                    or ""
                )
                lines.append(
                    "| "
                    + str(item_index)
                    + " | "
                    + _cell(item.get("evidence_id"))
                    + " | "
                    + _cell(item.get("document_title") or item.get("title"))
                    + " | "
                    + _cell(item.get("page_from"))
                    + " | "
                    + _cell(section)
                    + " | "
                    + _cell(str(observation)[:700])
                    + " |"
                )
        lines.append("")

    lines.extend(
        [
            "## First evidence/answer decision",
            "",
            "~~~json",
            json.dumps(
                report.get("first_answer_decision"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "~~~",
        ]
    )

    if report.get("final_answer_decision"):
        lines.extend(
            [
                "",
                "## Final decision after targeted gap research",
                "",
                "~~~json",
                json.dumps(
                    report.get("final_answer_decision"),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                "~~~",
            ]
        )

    lines.extend(
        [
            "",
            "## Task completion",
            "",
            "- Completed: "
            + (
                ", ".join(str(value) for value in report.get("completed_task_ids") or [])
                or "none"
            ),
            "- Skipped by hard research budget: "
            + (
                ", ".join(str(value) for value in report.get("skipped_task_ids") or [])
                or "none"
            ),
            "",
            "## AI-selected evidence",
            "",
            "| # | Document | Page | Section | Chunk | Source text |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    selected = report.get("selected_evidence") or []
    if not selected:
        lines.append("|  | No selected evidence |  |  |  |  |")
    for row in selected:
        section = " > ".join(str(value) for value in row.get("section_path") or [])
        lines.append(
            "| "
            + _cell(row.get("rank"))
            + " | "
            + _cell(row.get("document_title"))
            + " | "
            + _cell(row.get("page_from"))
            + " | "
            + _cell(section)
            + " | "
            + _cell(row.get("chunk_id"))
            + " | "
            + _cell(row.get("snippet"))
            + " |"
        )

    lines.extend(
        [
            "",
            "## Timings",
            "",
            "~~~json",
            json.dumps(
                report.get("workflow_timings_ms"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "~~~",
            "",
            "## Final answer",
            "",
            str(report.get("final_answer") or ""),
            "",
            "## Final citations",
            "",
            "~~~json",
            json.dumps(
                report.get("citations"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "~~~",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend("- " + str(note) for note in report.get("notes") or [])
    return "\n".join(lines)
