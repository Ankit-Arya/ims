from __future__ import annotations

import json
from collections.abc import Iterable
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


def _unique_strings(values: Iterable[Any], limit: int = 200) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split())
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _flatten_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            result.extend(_flatten_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_strings(item))
    return result


def _candidate_rows(trace: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("fused_candidate_preview", "reranked_candidate_preview", "evidence"):
        for raw in _as_list(trace.get(key)):
            if isinstance(raw, dict):
                item = dict(raw)
                item["_trace_lane"] = key
                rows.append(item)

    recovery = _as_dict(trace.get("recovery_summary"))
    for key in ("fused_candidate_preview", "reranked_candidate_preview", "evidence"):
        for raw in _as_list(recovery.get(key)):
            if isinstance(raw, dict):
                item = dict(raw)
                item["_trace_lane"] = "recovery." + key
                rows.append(item)

    seen: set[str] = set()
    result: list[dict] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        identity = chunk_id or json.dumps(row, sort_keys=True, default=str)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
        if len(result) >= 120:
            break
    return result


def _enrich_candidates(db: Session, user: User, candidates: list[dict]) -> list[dict]:
    ids: list[UUID] = []
    for row in candidates:
        try:
            ids.append(UUID(str(row.get("chunk_id"))))
        except (TypeError, ValueError):
            continue
    ids = list(dict.fromkeys(ids))
    if not ids:
        return candidates

    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.id.in_(ids),
            document_access_clause(user),
            Document.ingestion_status == "ready",
        )
    ).all()
    by_id = {str(chunk.id): (chunk, document) for chunk, document in rows}

    enriched: list[dict] = []
    for rank, raw in enumerate(candidates, start=1):
        row = dict(raw)
        row["debug_rank"] = rank
        pair = by_id.get(str(row.get("chunk_id") or ""))
        if pair is not None:
            chunk, document = pair
            row.setdefault("document_id", str(document.id))
            row.setdefault("document_title", document.title)
            row.setdefault("filename", document.original_filename)
            row.setdefault("page_from", chunk.page_from)
            row.setdefault("page_to", chunk.page_to)
            row.setdefault("section_path", chunk.section_path or [])
            row.setdefault("content_kind", chunk.content_kind)
            row["snippet"] = " ".join((chunk.text or "").split())[:1600]
        enriched.append(row)
    return enriched


def build_query_debug_report(db: Session, user: User, log: QueryLog) -> dict:
    trace = _as_dict(log.retrieval_trace)
    query_plan = _as_dict(trace.get("query_plan"))
    query_frame = _as_dict(trace.get("query_frame"))
    evidence_plan = _as_dict(trace.get("evidence_plan"))
    recovery = _as_dict(trace.get("recovery_summary"))

    goal_queries: list[str] = []
    goal_rows: list[dict] = []
    for raw_goal in _as_list(evidence_plan.get("goals")):
        if not isinstance(raw_goal, dict):
            continue
        goal = dict(raw_goal)
        goal_rows.append(goal)
        goal_queries.extend(_as_list(goal.get("search_queries")))

    inferred_queries = _unique_strings([
        *_as_list(query_plan.get("semantic_queries")),
        *_as_list(query_plan.get("lexical_queries")),
        *_as_list(query_plan.get("exact_terms")),
        *_as_list(query_frame.get("alternate_phrasings")),
        *_as_list(query_frame.get("canonical_terms")),
        *goal_queries,
    ])
    executed_queries = _unique_strings([
        *_as_list(trace.get("queries")),
        *[
            item.get("text")
            for item in _as_list(trace.get("query_specs"))
            if isinstance(item, dict)
        ],
    ])
    recovery_queries = _unique_strings([
        *_flatten_strings(trace.get("recovery_queries")),
        *_flatten_strings(recovery.get("recovery_queries")),
        *_flatten_strings(recovery.get("enumeration_section_rediscovery_queries")),
    ])
    per_query_results: list[dict] = []
    preview_groups = [
        ("initial", _as_list(trace.get("query_result_previews"))),
        ("recovery", _as_list(recovery.get("query_result_previews"))),
    ]
    for phase, previews in preview_groups:
        for raw_preview in previews:
            if not isinstance(raw_preview, dict):
                continue
            per_query_results.append({
                "phase": phase,
                "query_index": raw_preview.get("query_index"),
                "query": raw_preview.get("query"),
                "goal_ids": raw_preview.get("goal_ids") or [],
                "goal_kinds": raw_preview.get("goal_kinds") or [],
                "origins": raw_preview.get("origins") or [],
                "results": _enrich_candidates(db, user, _as_list(raw_preview.get("results"))),
            })

    agent_decisions = _as_list(trace.get("agent_decisions"))
    agent_interpretation = ""
    if agent_decisions and isinstance(agent_decisions[0], dict):
        agent_interpretation = str(agent_decisions[0].get("decision_summary") or "")

    return {
        "debug_version": 2,
        "query_id": str(log.id),
        "created_at": log.created_at.isoformat() if log.created_at else None,
        "original_question": log.question,
        "mode": log.mode,
        "agentic": bool(trace.get("agentic")),
        "agent_architecture": trace.get("agent_architecture"),
        "agent_decisions": agent_decisions,
        "tool_calls": _as_list(trace.get("tool_calls")),
        "agent_stop_reason": trace.get("agent_stop_reason"),
        "agent_evidence_status": trace.get("agent_evidence_status"),
        "agent_unresolved": trace.get("agent_unresolved") or [],
        "semantic_interpretation": (
            evidence_plan.get("interpretation")
            or agent_interpretation
            or query_frame.get("original")
        ),
        "answer_shape": evidence_plan.get("answer_shape") or query_frame.get("answer_shape"),
        "source_hints": evidence_plan.get("source_hints") or query_frame.get("explicit_scope") or [],
        "entities": evidence_plan.get("entities") or query_frame.get("entities") or [],
        "evidence_goals": goal_rows,
        "inferred_or_rephrased_queries": inferred_queries,
        "executed_search_queries": executed_queries,
        "recovery_search_queries": recovery_queries,
        "per_query_search_results": per_query_results,
        "corpus_discovery": trace.get("corpus_discovery"),
        "okf_resolution": trace.get("okf_resolution"),
        "query_plan": query_plan,
        "query_frame": query_frame,
        "search_results": _enrich_candidates(db, user, _candidate_rows(trace)),
        "source_counts": trace.get("source_counts"),
        "rerank_details": trace.get("rerank_details"),
        "goal_satisfaction": trace.get("goal_satisfaction"),
        "retrieval_quality_issues": trace.get("retrieval_quality_issues"),
        "recovery_attempted": trace.get("recovery_attempted"),
        "recovery_summary": recovery,
        "workflow_timings_ms": trace.get("workflow_timings_ms"),
        "final_answer": log.answer,
        "citations": log.citations or [],
        "notes": [
            "Search queries and corpus hints are retrieval hypotheses, not answer evidence.",
            "Search results are reconstructed from stored candidate/evidence previews and enriched with accessible chunk text.",
            "This artifact is intended for testing and acceptance diagnostics.",
        ],
    }


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def debug_report_markdown(report: dict) -> str:
    lines = [
        "# IMS Query Debug Report",
        "",
        "- Query ID: " + _cell(report.get("query_id")),
        "- Created: " + _cell(report.get("created_at")),
        "- Mode: " + _cell(report.get("mode")),
        "",
        "## Original question",
        "",
        str(report.get("original_question") or ""),
        "",
        "## AI interpretation",
        "",
        "- Interpretation: " + _cell(report.get("semantic_interpretation")),
        "- Answer shape: " + _cell(report.get("answer_shape")),
        "- Source hints: " + (", ".join(str(x) for x in (report.get("source_hints") or [])) or "none"),
        "- Entities: " + (", ".join(str(x) for x in (report.get("entities") or [])) or "none"),
        "",
    ]

    if report.get("agentic"):
        lines.extend([
            "## Agent research loop",
            "",
            "- Architecture: " + _cell(report.get("agent_architecture")),
            "- Evidence status: " + _cell(report.get("agent_evidence_status")),
            "- Stop reason: " + _cell(report.get("agent_stop_reason")),
            "- Unresolved: " + (
                ", ".join(str(x) for x in (report.get("agent_unresolved") or []))
                or "none"
            ),
            "",
        ])
        for index, decision in enumerate(report.get("agent_decisions") or [], 1):
            if not isinstance(decision, dict):
                continue
            lines.append(
                str(index)
                + ". "
                + _cell(decision.get("action"))
                + " - "
                + _cell(decision.get("decision_summary"))
            )
        lines.append("")
        lines.extend(["### Tool calls", ""])
        for index, call in enumerate(report.get("tool_calls") or [], 1):
            if not isinstance(call, dict):
                continue
            lines.append(
                str(index)
                + ". "
                + _cell(call.get("tool"))
                + " "
                + _cell(call.get("arguments"))
                + " ("
                + _cell(call.get("elapsed_ms"))
                + " ms)"
            )
        lines.append("")

    lines.extend([
        "## Evidence goals",
        "",
    ])

    goals = report.get("evidence_goals") or []
    if not goals:
        lines.append("- No evidence goals stored.")
    for goal in goals:
        lines.extend([
            "### " + _cell(goal.get("id")) + " - " + _cell(goal.get("kind")),
            "- Need: " + _cell(goal.get("question")),
            "- Coverage: " + _cell(goal.get("coverage_contract")),
            "- Retrieval tools: " + (", ".join(str(x) for x in (goal.get("retrieval_tools") or [])) or "search"),
            "- Search probes: " + (", ".join(str(x) for x in (goal.get("search_queries") or [])) or "none"),
            "",
        ])

    for title, key in (
        ("Inferred / rephrased search questions", "inferred_or_rephrased_queries"),
        ("Actually executed search queries", "executed_search_queries"),
        ("Recovery search queries", "recovery_search_queries"),
    ):
        lines.extend(["## " + title, ""])
        values = report.get(key) or []
        if values:
            lines.extend([str(index) + ". " + str(value) for index, value in enumerate(values, 1)])
        else:
            lines.append("- None stored.")
        lines.append("")

    lines.extend(["## Search results by executed query", ""])
    per_query = report.get("per_query_search_results") or []
    if not per_query:
        lines.append("- No per-query previews were stored for this run.")
        lines.append("")
    for item in per_query:
        lines.append("### [" + _cell(item.get("phase") or "initial") + "] " + _cell(item.get("query")))
        meta = []
        if item.get("goal_ids"):
            meta.append("goals=" + ",".join(str(x) for x in item.get("goal_ids") or []))
        if item.get("origins"):
            meta.append("origins=" + ",".join(str(x) for x in item.get("origins") or []))
        if meta:
            lines.append("- " + "; ".join(meta))
        for row in item.get("results") or []:
            section = " > ".join(str(x) for x in (row.get("section_path") or []))
            lines.append(
                "- [" + _cell(row.get("source")) + "] "
                + _cell(row.get("document_title"))
                + " p." + _cell(row.get("page_from"))
                + (" - " + _cell(section) if section else "")
                + ": " + _cell((row.get("snippet") or "")[:500])
            )
        if not (item.get("results") or []):
            lines.append("- No previewed candidates for this probe.")
        lines.append("")

    lines.extend([
        "## Combined top search results",
        "",
        "| # | Lane | Document | Page | Section | Scores / sources | Snippet |",
        "|---:|---|---|---:|---|---|---|",
    ])
    results = report.get("search_results") or []
    for row in results[:80]:
        page = row.get("page_from") if row.get("page_from") is not None else ""
        section = " > ".join(str(x) for x in (row.get("section_path") or []))
        score_bits = []
        for name in ("rerank_score", "final_retrieval_score", "fused_score", "judge_score"):
            if row.get(name) not in (None, ""):
                score_bits.append(name + "=" + str(row.get(name)))
        if row.get("sources"):
            score_bits.append("sources=" + ",".join(str(x) for x in row.get("sources") or []))
        lines.append(
            "| " + _cell(row.get("debug_rank")) +
            " | " + _cell(row.get("_trace_lane")) +
            " | " + _cell(row.get("document_title") or row.get("document")) +
            " | " + _cell(page) +
            " | " + _cell(section or row.get("section")) +
            " | " + _cell("; ".join(score_bits)) +
            " | " + _cell((row.get("snippet") or row.get("text") or row.get("excerpt") or "")[:700]) + " |"
        )
    if not results:
        lines.append("|  |  | No stored candidate preview |  |  |  |  |")

    lines.extend([
        "",
        "## Goal audit",
        "",
        "~~~json",
        json.dumps(report.get("goal_satisfaction"), ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## Recovery / replanning",
        "",
        "~~~json",
        json.dumps(report.get("recovery_summary"), ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## OKF / document routing",
        "",
        "~~~json",
        json.dumps(report.get("okf_resolution"), ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## Corpus discovery",
        "",
        "~~~json",
        json.dumps(report.get("corpus_discovery"), ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## Final answer",
        "",
        str(report.get("final_answer") or ""),
        "",
        "## Final citations",
        "",
        "~~~json",
        json.dumps(report.get("citations"), ensure_ascii=False, indent=2, default=str),
        "~~~",
        "",
        "## Notes",
        "",
    ])
    lines.extend("- " + str(note) for note in report.get("notes") or [])
    return "\n".join(lines)
