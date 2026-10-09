from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.orm import Session

from ike.agent.models import ResearchTask
from ike.agent.tools import CorpusTools, ToolResult
from ike.db.models import User
from ike.retrieval.types import Candidate

CancelCheckFn = Callable[[], None]


@dataclass(slots=True)
class ResearchBundle:
    observations: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    candidate_store: dict[str, Candidate] = field(default_factory=dict)
    evidence_id_by_chunk: dict[str, str] = field(default_factory=dict)
    evidence_order: list[str] = field(default_factory=list)
    task_evidence_order: dict[str, list[str]] = field(default_factory=dict)
    completed_task_ids: set[str] = field(default_factory=set)
    skipped_task_ids: list[str] = field(default_factory=list)

    def add_candidates(
        self,
        candidates: list[Candidate],
        *,
        task_id: str,
    ) -> None:
        bucket = self.task_evidence_order.setdefault(task_id, [])
        for candidate in candidates:
            chunk_id = str(candidate.chunk_id)
            existing = self.candidate_store.get(chunk_id)
            if existing is None:
                self.candidate_store[chunk_id] = candidate
                evidence_id = f"E{len(self.evidence_order) + 1}"
                self.evidence_id_by_chunk[chunk_id] = evidence_id
                self.evidence_order.append(chunk_id)
            else:
                existing.sources.update(candidate.sources)
                existing.final_retrieval_score = max(
                    existing.final_retrieval_score,
                    candidate.final_retrieval_score,
                )
            if chunk_id not in bucket:
                bucket.append(chunk_id)

    def evidence_rows(self, *, max_items: int = 120) -> list[dict]:
        """Return evidence balanced across research tasks, not only earliest searches."""

        ordered_ids: list[str] = []
        buckets = [
            list(values)
            for values in self.task_evidence_order.values()
            if values
        ]
        positions = [0] * len(buckets)

        while buckets and len(ordered_ids) < max_items:
            added = False
            for index, bucket in enumerate(buckets):
                position = positions[index]
                while position < len(bucket) and bucket[position] in ordered_ids:
                    position += 1
                positions[index] = position
                if position >= len(bucket):
                    continue
                ordered_ids.append(bucket[position])
                positions[index] += 1
                added = True
                if len(ordered_ids) >= max_items:
                    break
            if not added:
                break

        if len(ordered_ids) < max_items:
            for chunk_id in self.evidence_order:
                if chunk_id in ordered_ids:
                    continue
                ordered_ids.append(chunk_id)
                if len(ordered_ids) >= max_items:
                    break

        rows: list[dict] = []
        for chunk_id in ordered_ids:
            candidate = self.candidate_store[chunk_id]
            rows.append(
                {
                    "evidence_id": self.evidence_id_by_chunk[chunk_id],
                    "chunk_id": chunk_id,
                    "document_id": str(candidate.document_id),
                    "document_title": candidate.document_title,
                    "filename": candidate.filename,
                    "page_from": candidate.page_from,
                    "page_to": candidate.page_to,
                    "section_path": list(candidate.section_path or []),
                    "content_kind": candidate.content_kind,
                    "sources": sorted(candidate.sources),
                    "text": (
                        candidate.contextual_text
                        or candidate.text
                        or ""
                    )[:4500],
                }
            )
        return rows

    def evidence_selection_rows(
        self,
        *,
        max_items: int,
        excerpt_chars: int,
    ) -> list[dict]:
        rows = self.evidence_rows(max_items=max_items)
        compact: list[dict] = []
        for row in rows:
            candidate = self.candidate_store.get(str(row["chunk_id"]))
            if candidate is None:
                continue
            compact.append(
                {
                    **row,
                    "text": str(row.get("text") or "")[:excerpt_chars],
                    "score": round(
                        float(
                            candidate.final_retrieval_score
                            or candidate.rerank_score
                            or candidate.fused_score
                            or 0.0
                        ),
                        6,
                    ),
                    "direct_score": round(
                        float(
                            candidate.judge_details.get(
                                "direct_evidence_score",
                                0.0,
                            )
                            or 0.0
                        ),
                        6,
                    ),
                    "coverage_groups": list(
                        candidate.judge_details.get("coverage_groups", []) or []
                    ),
                    "family_key": candidate.family_key,
                    "document_profile": dict(candidate.document_profile or {}),
                }
            )
        return compact

    def evidence_rows_for_ids(
        self,
        evidence_ids: list[str],
        *,
        max_items: int,
        reserve_items: int = 0,
    ) -> list[dict]:
        requested = [
            str(value)
            for value in evidence_ids
            if str(value)
        ]
        requested_set = set(requested)
        by_id = {
            row["evidence_id"]: row
            for row in self.evidence_rows(
                max_items=max(max_items, len(self.evidence_order))
            )
        }
        selected = [
            by_id[evidence_id]
            for evidence_id in requested
            if evidence_id in by_id
        ]
        if reserve_items > 0 and len(selected) < max_items:
            for row in by_id.values():
                if row["evidence_id"] in requested_set:
                    continue
                selected.append(row)
                if (
                    len(selected) >= max_items
                    or len(selected) >= len(requested) + reserve_items
                ):
                    break
        return selected[:max_items]

    def answer_observations(
        self,
        *,
        max_observations: int = 44,
        max_items_per_observation: int = 12,
    ) -> list[dict]:
        """Compact retrieval observations for the answer LLM.

        Full tool traces remain available in query debug. The answer model receives routing
        metadata and evidence IDs, while chunk prose is supplied once via evidence_rows().
        """
        compact: list[dict] = []
        for observation in self.observations[-max_observations:]:
            row = dict(observation)
            items: list[dict] = []
            for item in list(observation.get("items") or [])[:max_items_per_observation]:
                item_row = {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "evidence_id",
                        "chunk_id",
                        "document_id",
                        "document_title",
                        "title",
                        "filename",
                        "page_from",
                        "page_to",
                        "section_path",
                        "score",
                        "metadata_hints",
                        "matched_sections",
                    }
                }
                items.append(item_row)
            row["items"] = items
            compact.append(row)
        return compact

    def candidate_for_evidence(self, evidence_id: str) -> Candidate | None:
        for chunk_id, current_id in self.evidence_id_by_chunk.items():
            if current_id == evidence_id:
                return self.candidate_store.get(chunk_id)
        return None


class ResearchExecutor:
    """Execute an AI-created research graph within hard access/time bounds."""

    def __init__(
        self,
        db: Session,
        user: User,
        *,
        request_id: str | None,
        allowed_document_ids: list[UUID] | None,
        cancel_check: CancelCheckFn | None = None,
    ) -> None:
        self.tools = CorpusTools(
            db,
            user,
            request_id=request_id,
            allowed_document_ids=allowed_document_ids,
        )
        self.cancel_check = cancel_check
        self.bundle = ResearchBundle()

    def _checkpoint(self) -> None:
        if self.cancel_check:
            self.cancel_check()

    def _record(
        self,
        *,
        task: ResearchTask,
        result: ToolResult,
        round_name: str,
        evidence_bucket: str | None = None,
    ) -> None:
        self.bundle.add_candidates(
            result.candidates,
            task_id=evidence_bucket or task.id,
        )
        items: list[dict] = []
        for item in result.items:
            row = dict(item)
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id and chunk_id in self.bundle.evidence_id_by_chunk:
                row["evidence_id"] = self.bundle.evidence_id_by_chunk[chunk_id]
            if "snippet" in row:
                row["snippet"] = str(row.get("snippet") or "")[:1000]
            items.append(row)
        observation = {
            "round": round_name,
            "task_id": task.id,
            "purpose": task.purpose,
            "requirement_ids": task.requirement_ids,
            "kind": task.kind,
            "depends_on": task.depends_on,
            "tool": result.tool,
            "arguments": result.arguments,
            "items": items,
            "metadata": result.metadata,
            "elapsed_ms": result.elapsed_ms,
            "note": result.note,
        }
        self.bundle.observations.append(observation)
        trace = result.trace()
        trace.update(
            {
                "round": round_name,
                "task_id": task.id,
                "purpose": task.purpose,
                "requirement_ids": task.requirement_ids,
                "depends_on": task.depends_on,
            }
        )
        self.bundle.traces.append(trace)

    def _expand_requested_context(
        self,
        task: ResearchTask,
        result: ToolResult,
        round_name: str,
    ) -> None:
        if (
            task.context_hits <= 0
            or (task.context_before <= 0 and task.context_after <= 0)
            or not result.candidates
        ):
            return

        context_candidates = list(result.candidates[: task.context_hits])
        if task.kind == "enumerate" and task.coverage_facets:
            coverage = dict(result.metadata.get("coverage_balance") or {})
            groups = [
                str(value)
                for value in (coverage.get("groups") or [])
                if str(value)
            ][
                : self.tools.settings.agent_enumeration_context_group_limit
            ]
            target_hits = max(task.context_hits, len(groups))
            selected: list[Candidate] = []
            selected_chunks: set[str] = set()

            for group in groups:
                for candidate in result.candidates:
                    chunk_id = str(candidate.chunk_id)
                    candidate_groups = set(
                        str(value)
                        for value in (
                            candidate.judge_details.get("coverage_groups", [])
                            or []
                        )
                    )
                    if (
                        chunk_id in selected_chunks
                        or group not in candidate_groups
                    ):
                        continue
                    selected.append(candidate)
                    selected_chunks.add(chunk_id)
                    break

            for candidate in result.candidates:
                if len(selected) >= target_hits:
                    break
                chunk_id = str(candidate.chunk_id)
                if chunk_id in selected_chunks:
                    continue
                selected.append(candidate)
                selected_chunks.add(chunk_id)

            context_candidates = selected[:target_hits]

        seen_chunks: set[str] = set()
        for index, candidate in enumerate(
            context_candidates,
            start=1,
        ):
            chunk_id = str(candidate.chunk_id)
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            context_task = ResearchTask(
                id=f"{task.id}_context_{index}",
                purpose=(
                    "AI-requested neighboring source context for: "
                    + task.purpose
                ),
                requirement_ids=task.requirement_ids,
                kind="context",
                chunk_id=chunk_id,
                context_before=task.context_before,
                context_after=task.context_after,
            )
            context = self.tools.inspect_context(
                chunk_id,
                before=task.context_before,
                after=task.context_after,
            )
            self._record(
                task=context_task,
                result=context,
                round_name=round_name,
                evidence_bucket=task.id,
            )

    def _execute_task(self, task: ResearchTask, round_name: str) -> None:
        self._checkpoint()

        if task.kind == "source_lookup":
            routing_query = " ".join(
                value
                for value in (
                    task.source_query,
                    task.query,
                    *task.query_variants,
                )
                if value and value.strip()
            )
            result = self.tools.search_documents(routing_query)
            self._record(task=task, result=result, round_name=round_name)
            return

        boost_document_ids: list[str] = []
        if task.source_query:
            routing_query = " ".join(
                value
                for value in (
                    task.source_query,
                    task.query,
                    *task.query_variants,
                )
                if value and value.strip()
            )
            routing = self.tools.search_documents(routing_query)
            self._record(task=task, result=routing, round_name=round_name)
            boost_document_ids = [
                str(item["document_id"])
                for item in routing.items
                if item.get("document_id")
            ]

        # Only an explicit document_id is a hard per-task scope. AI-inferred source
        # candidates are ranking hints so globally strong chunk evidence can still win.
        hard_document_ids = [task.document_id] if task.document_id else []

        if task.kind == "search":
            result = self.tools.search(
                task.query,
                mode=task.search_mode,
                query_variants=task.query_variants,
                exact_terms=task.exact_terms,
                document_ids=hard_document_ids,
                boost_document_ids=boost_document_ids,
                top_k=task.top_k,
            )
            self._record(task=task, result=result, round_name=round_name)
            self._expand_requested_context(task, result, round_name)
            return

        if task.kind == "enumerate":
            result = self.tools.enumerate(
                task.query,
                mode=task.search_mode,
                query_variants=task.query_variants,
                exact_terms=task.exact_terms,
                document_ids=hard_document_ids,
                boost_document_ids=boost_document_ids,
                coverage_facets=task.coverage_facets,
                top_k=task.top_k,
            )
            self._record(task=task, result=result, round_name=round_name)
            self._expand_requested_context(task, result, round_name)
            return

        if task.kind == "structure":
            if not task.document_id:
                self.bundle.observations.append(
                    {
                        "round": round_name,
                        "task_id": task.id,
                        "purpose": task.purpose,
                        "requirement_ids": task.requirement_ids,
                        "kind": task.kind,
                        "status": "requires_document_choice",
                        "source_query": task.source_query,
                    }
                )
                return
            result = self.tools.inspect_structure(
                task.document_id,
                query=task.query,
            )
            self._record(task=task, result=result, round_name=round_name)
            return

        if task.kind == "context":
            if not task.chunk_id:
                self.bundle.observations.append(
                    {
                        "round": round_name,
                        "task_id": task.id,
                        "purpose": task.purpose,
                        "requirement_ids": task.requirement_ids,
                        "kind": task.kind,
                        "status": "requires_chunk_choice",
                    }
                )
                return
            result = self.tools.inspect_context(
                task.chunk_id,
                before=task.context_before,
                after=task.context_after,
            )
            self._record(task=task, result=result, round_name=round_name)

    def execute(
        self,
        tasks: list[ResearchTask],
        *,
        round_name: str,
        time_budget_seconds: float,
    ) -> ResearchBundle:
        started = time.perf_counter()
        pending = {task.id: task for task in tasks}

        while pending:
            if time.perf_counter() - started >= time_budget_seconds:
                skipped = list(pending)
                self.bundle.skipped_task_ids.extend(skipped)
                self.bundle.observations.append(
                    {
                        "round": round_name,
                        "status": "research_time_budget_exhausted",
                        "task_ids": skipped,
                    }
                )
                break

            ready = [
                task
                for task in pending.values()
                if all(
                    dependency in self.bundle.completed_task_ids
                    or dependency not in pending
                    for dependency in task.depends_on
                )
            ]
            if not ready:
                self.bundle.skipped_task_ids.extend(pending)
                self.bundle.observations.append(
                    {
                        "round": round_name,
                        "status": "dependency_cycle_or_unresolved_dependency",
                        "task_ids": list(pending),
                    }
                )
                break

            for task in ready:
                if time.perf_counter() - started >= time_budget_seconds:
                    self.bundle.skipped_task_ids.extend(
                        task_id
                        for task_id in pending
                        if task_id not in self.bundle.skipped_task_ids
                    )
                    return self.bundle
                self._execute_task(task, round_name)
                self.bundle.completed_task_ids.add(task.id)
                pending.pop(task.id, None)

        return self.bundle
