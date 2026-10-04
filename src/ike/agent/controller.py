from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ike.core.config import get_settings
from ike.db.models import User
from ike.mcp.server import CorpusMCPServer, ToolExecution
from ike.retrieval.search_plan import token_overlap
from ike.retrieval.table_context import retrieval_text
from ike.retrieval.types import Candidate, Evidence
from ike.services.llm import LLMClient


ProgressFn = Callable[[str, str, str, int], None]
AnswerDeltaFn = Callable[[str], None]
CancelCheckFn = Callable[[], None]


class AgentDecision(BaseModel):
    action: Literal[
        "search_documents",
        "search_chunks",
        "search_many",
        "search_lists",
        "search_sections",
        "get_document_structure",
        "get_section",
        "search_tables",
        "exact_lookup",
        "finish",
    ]
    decision_summary: str = Field(default="", max_length=500)
    query: str = Field(default="", max_length=1200)
    queries: list[str] = Field(default_factory=list, max_length=4)
    subjects: list[str] = Field(default_factory=list, max_length=4)
    document_ids: list[str] = Field(default_factory=list, max_length=8)
    document_id: str | None = None
    section_selector: str = Field(default="", max_length=700)
    evidence_status: Literal["insufficient", "partial", "sufficient"] = "insufficient"
    selected_chunk_ids: list[str] = Field(default_factory=list, max_length=20)
    unresolved: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("queries", "subjects", "document_ids", "selected_chunk_ids", "unresolved", mode="before")
    @classmethod
    def _coerce_empty_lists(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator("evidence_status", mode="before")
    @classmethod
    def _coerce_empty_status(cls, value):
        normalized = str(value or "").strip().casefold()
        if normalized in {"insufficient", "partial", "sufficient"}:
            return normalized
        # Harmless controller vocabulary such as pending/unknown/continue means the
        # evidence is not yet sufficient; do not throw away an otherwise valid action.
        return "insufficient"


class AgenticQAService:
    """Bounded research agent over read-only corpus tools.

    The controller owns semantic interpretation. Application code only enforces tool
    permissions, evidence/citation grounding and resource limits.
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

    @staticmethod
    def _candidate_rank(candidate: Candidate) -> tuple[float, float, float, int]:
        return (
            candidate.final_retrieval_score,
            candidate.rerank_score,
            candidate.fused_score,
            -candidate.ordinal,
        )

    @staticmethod
    def _compact_tool_observation(execution: ToolExecution) -> dict:
        items: list[dict] = []
        observation_limit = (
            60
            if execution.tool == "get_document_structure"
            else 32
            if execution.tool == "get_section"
            else 10
        )
        for item in execution.items[:observation_limit]:
            compact = {
                key: value
                for key, value in item.items()
                if key
                in {
                    "document_id",
                    "document_title",
                    "title",
                    "filename",
                    "chunk_id",
                    "page_from",
                    "page_to",
                    "section_path",
                    "label",
                    "score",
                    "source",
                    "source_role",
                    "authority",
                    "query",
                    "revision",
                    "child_headings",
                    "snippet",
                }
            }
            if isinstance(compact.get("snippet"), str):
                compact["snippet"] = compact["snippet"][:850]
            items.append(compact)

        query_groups: list[dict] = []
        group_item_limit = 8 if execution.tool == "search_lists" else 5
        group_snippet_limit = 1200 if execution.tool == "search_lists" else 700
        for group in execution.query_groups[:4]:
            group_items: list[dict] = []
            for item in (group.get("items") or [])[:group_item_limit]:
                compact = {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "document_id",
                        "document_title",
                        "title",
                        "filename",
                        "chunk_id",
                        "page_from",
                        "page_to",
                        "section_path",
                        "label",
                        "score",
                        "source",
                        "query",
                        "snippet",
                    }
                }
                if isinstance(compact.get("snippet"), str):
                    compact["snippet"] = compact["snippet"][:group_snippet_limit]
                group_items.append(compact)
            query_groups.append(
                {
                    "query": group.get("query"),
                    "elapsed_ms": group.get("elapsed_ms"),
                    "items": group_items,
                }
            )
        return {
            "tool": execution.tool,
            "arguments": execution.arguments,
            "elapsed_ms": execution.elapsed_ms,
            "note": execution.note,
            "items": items,
            "query_groups": query_groups,
        }

    def _controller_system_prompt(self) -> str:
        catalog = json.dumps(CorpusMCPServer.tool_catalog(), ensure_ascii=False)
        return (
            "You are the research controller for an internal-document Q&A product. "
            "Your job is to understand the user's natural-language request, choose one safe "
            "read-only corpus tool at a time, inspect the returned evidence, and STOP as soon "
            "as the actual user request is sufficiently supported. You do not answer from "
            "your own knowledge. You do not force the question into a predefined intent "
            "taxonomy. You do not invent additional requirements, scenarios, exceptions, "
            "documents, entities or completeness conditions that the user did not ask for. "
            "Search hypotheses may be broader than the final requirement, but they are only "
            "search hypotheses and must never become new answer requirements. "
            "\n\n"
            "A simple question should normally take one good search and then finish. If a "
            "result directly answers every requested part, select its chunk ids and finish; "
            "do not keep researching for theoretical completeness. If you identify a document "
            "whose title/section directly describes the user's requested event/topic and its "
            "returned chunks contain the requested actors, facts or procedure, treat that as "
            "strong evidence and finish unless a SPECIFIC user-requested part is still missing. "
            "Never search the same document repeatedly just to discover unspecified extra cases. "
            "For a request containing multiple independent categories, entities, or subquestions, "
            "treat each requested part independently. Evidence that resolves one part must not "
            "scope later searches for other unresolved parts to the same document unless the user "
            "explicitly requested one common source. If a combined search is dominated by one part, "
            "search each remaining part corpus-wide and keep the already-supported evidence. "
            "For a request asking to list, name, enumerate, or provide all members of one or "
            "more categories, prefer search_lists and provide one minimal semantic subject per "
            "requested set (for example the category noun itself, not phrases like 'complete list "
            "of ...' or 'names of ...'). Do not re-search a list subject already supported by a "
            "direct list-bearing section. For other multi-part requests, prefer search_many with one semantic query per "
            "part; do not concatenate independent parts into one retrieval query. "
            "\n\n"
            "Tool guidance: use search_chunks as the default factual/procedural search for one need. Use "
            "search_documents only when the user names or clearly refers to a document/source "
            "and you need to resolve it. After resolving a document, use get_section directly "
            "when the user supplied a specific chapter/section name or number; use search_sections "
            "when the section identity is uncertain. Once two chunk searches point to the same likely "
            "document or named section but details are still missing, switch to search_sections or "
            "get_section instead of issuing another paraphrased chunk search. If search_lists returns a clearly matching list section but its snippet appears to start or end mid-list, use get_section on that document/section to retrieve the bounded adjacent chunks before finishing. Use get_document_structure for contents/hierarchy/list-of-sections "
            "questions. When using get_document_structure for a requested structural category, put the user's own "
            "category wording in the query field so the tool can filter the hierarchy. For a structure/list request, the hierarchy labels returned by get_document_structure are authoritative names as stored in the source. If the user explicitly asks for descriptive names/titles and a hierarchy item is only a bare number/identifier, inspect its child_headings and, if still useful, do one scoped search/exact lookup inside that document before finishing; never invent a missing title. Use search_tables for row/column/list/rate information. Use exact_lookup "
            "for precise identifiers, rule numbers, quoted phrases or codes. "
            "\n\n"
            "Never treat a role, station, equipment name or ordinary entity as a source document "
            "just because a filename happens to contain the same token. A source is a source only "
            "when the user's wording or a document-search result supports that relationship. "
            "Preserve scope: when the user asks about a condition generally and results include both "
            "a general procedure and a location-, equipment-, line-, or scenario-specific variant, "
            "prefer the general procedure. Use a narrower variant only when the user asked for that "
            "scope or when no broader evidence exists. Do not silently generalize a special case. "
            "\n\n"
            "When finishing, selected_chunk_ids must contain only chunk ids already returned by "
            "tools. Select the smallest set that supports all explicitly requested parts. Unless "
            "the user asks for all, complete, exhaustive, every, or an equivalent completeness "
            "claim, do NOT mark the answer partial merely because additional duties, exceptions, "
            "or scenarios might exist beyond what was asked. If an explicitly requested point "
            "remains unsupported after useful attempts, finish with evidence_status=partial and "
            "name only that concrete unresolved point; do not continue searching indefinitely. "
            "\n\n"
            "decision_summary is a brief operational justification, not hidden chain-of-thought. "
            "Available tools:\n"
            + catalog
        )

    def _controller_user_prompt(
        self,
        *,
        question: str,
        operational_context: dict,
        history: list[dict],
        step: int,
        max_steps: int,
    ) -> str:
        return (
            f"User question:\n{question}\n\n"
            f"Optional operational context:\n{json.dumps(operational_context, ensure_ascii=False)}\n\n"
            f"Research step: {step}/{max_steps}\n\n"
            "Tool history (observations only):\n"
            + json.dumps(history[-5:], ensure_ascii=False, indent=2)
            + "\n\nChoose the single next action. If current evidence already supports the "
            "user's actual request, choose finish now.\n\n"
            "Return one JSON object with fields: action, decision_summary, query, queries, subjects, "
            "document_ids, document_id, section_selector, evidence_status, "
            "selected_chunk_ids, unresolved. Use empty strings/lists for unused fields."
        )

    @staticmethod
    def _decision_arguments(decision: AgentDecision, question: str) -> dict:
        if decision.action == "search_documents":
            return {"query": decision.query.strip() or question}
        if decision.action == "search_many":
            return {
                "queries": [value.strip() for value in decision.queries if value.strip()],
                "document_ids": decision.document_ids,
            }
        if decision.action == "search_lists":
            return {
                "subjects": [value.strip() for value in decision.subjects if value.strip()],
                "document_ids": decision.document_ids,
            }
        if decision.action in {"search_chunks", "search_sections", "search_tables", "exact_lookup"}:
            return {
                "query": decision.query.strip() or question,
                "document_ids": decision.document_ids,
            }
        if decision.action == "get_document_structure":
            return {
                "document_id": decision.document_id or "",
                "query": decision.query.strip(),
            }
        if decision.action == "get_section":
            return {
                "document_id": decision.document_id or "",
                "section_selector": decision.section_selector.strip() or decision.query.strip(),
            }
        return {}

    @staticmethod
    def _resolved_research_focus(question: str, decisions: list[dict]) -> str:
        parts: list[str] = [" ".join(str(question or "").split())]
        seen = {parts[0].casefold()} if parts[0] else set()
        for raw in decisions:
            if not isinstance(raw, dict):
                continue
            values = [raw.get("query"), *(raw.get("queries") or [])]
            for value in values:
                cleaned = " ".join(str(value or "").split())
                if not cleaned:
                    continue
                key = cleaned.casefold()
                if key in seen:
                    continue
                seen.add(key)
                parts.append(cleaned)
                if len(parts) >= 5:
                    return " ; ".join(parts)
        return " ; ".join(parts)

    @staticmethod
    def _focus_support_score(focus: str, candidate: Candidate) -> float:
        haystack = (
            retrieval_text(candidate)
            + "\n"
            + " > ".join(candidate.section_path or [])
            + "\n"
            + (candidate.document_title or "")
        )
        return token_overlap(focus, haystack)

    def _selected_candidates(
        self,
        *,
        decision: AgentDecision | None,
        candidate_store: dict[str, Candidate],
        last_execution: ToolExecution | None,
        resolved_focus: str,
    ) -> list[Candidate]:
        selected: list[Candidate] = []
        seen: set[UUID] = set()
        if decision is not None:
            for raw_id in decision.selected_chunk_ids:
                candidate = candidate_store.get(str(raw_id))
                if candidate is None or candidate.chunk_id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.chunk_id)
                if len(selected) >= self.settings.agent_max_evidence:
                    break

        # For compact factual/procedural answers, validate model-selected chunk IDs
        # against the agent's own resolved search focus. This catches cases where the model
        # describes the correct entity/role but accidentally returns a neighboring chunk ID
        # for a similarly named concept. Structure/list outputs preserve source order.
        preserve_source_order = (
            (last_execution is not None and last_execution.tool in {
                "get_document_structure", "get_section", "search_lists"
            })
            or any(
                candidate.evidence_lane in {"document_structure", "section_navigation"}
                and len(selected) > 6
                for candidate in selected
            )
        )
        if candidate_store and resolved_focus and not preserve_source_order:
            support_ranked = sorted(
                candidate_store.values(),
                key=lambda candidate: (
                    self._focus_support_score(resolved_focus, candidate),
                    *self._candidate_rank(candidate),
                ),
                reverse=True,
            )
            best_support = (
                self._focus_support_score(resolved_focus, support_ranked[0])
                if support_ranked
                else 0.0
            )
            promoted: list[Candidate] = []
            promoted_seen: set[UUID] = set()
            floor = max(0.25, best_support - 0.20)
            for candidate in support_ranked:
                score = self._focus_support_score(resolved_focus, candidate)
                if score < floor:
                    continue
                promoted.append(candidate)
                promoted_seen.add(candidate.chunk_id)
                if len(promoted) >= 4:
                    break
            compact_limit = min(6, self.settings.agent_max_evidence)
            for candidate in selected:
                if candidate.chunk_id not in promoted_seen:
                    promoted.append(candidate)
                    promoted_seen.add(candidate.chunk_id)
                if len(promoted) >= compact_limit:
                    break
            selected = promoted[:compact_limit]
            seen = {candidate.chunk_id for candidate in selected}

        # If the controller did not explicitly select ids, preserve the strongest evidence
        # from the research ledger. A later exploratory tool call must not erase stronger
        # evidence obtained earlier.
        if not selected:
            ranked = sorted(
                candidate_store.values(),
                key=self._candidate_rank,
                reverse=True,
            )
            for candidate in ranked:
                if candidate.chunk_id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.chunk_id)
                if len(selected) >= self.settings.agent_max_evidence:
                    break
        return selected[: self.settings.agent_max_evidence]

    @staticmethod
    def _citation_ids(answer: str, evidence: list[Evidence]) -> list[str]:
        valid = {item.evidence_id for item in evidence}
        found = []
        seen = set()
        for value in re.findall(r"\[(E\d+)\]", answer or "", flags=re.IGNORECASE):
            evidence_id = value.upper()
            if evidence_id in valid and evidence_id not in seen:
                seen.add(evidence_id)
                found.append(evidence_id)
        return found

    def _answer_system_prompt(self) -> str:
        return (
            "Answer the user's question using ONLY the supplied documentary evidence. "
            "Do not use outside knowledge. Give the useful answer immediately; do not expose "
            "search/retrieval internals. Cite factual statements with [E#] references. "
            "If the evidence directly supports the requested answer, answer confidently and "
            "concisely. If one specific point explicitly requested by the user remains unresolved, "
            "answer the supported parts first and say only that the remaining point is 'not yet "
            "verified from the available evidence'. Do not add caveats about hypothetical extra "
            "duties, exceptions, scenarios, or details the user did not request. Avoid phrases "
            "such as 'could not retrieve', 'documents do not "
            "contain', 'not present in supplied extracts', or similar system-failure wording. "
            "Do not add scenarios or requirements the user did not ask for. "
            "For role/responsibility questions, organize the answer by the requested roles. If the research focus explicitly expanded an acronym or ambiguous term, preserve that resolved meaning and do not switch to another expansion merely because a neighboring evidence chunk uses the same abbreviation differently. "
            "For lists or document structure, preserve source order and use the exact hierarchy labels provided by the source. Do not invent missing titles or treat a number-only source label as incomplete unless the user explicitly asks for an expanded descriptive title. When summarizing a named parent section/chapter, cover every immediate rule/subheading visible in the selected evidence, including a following heading embedded in the same source chunk at a page/section boundary."
        )

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
        max_steps = max(1, self.settings.agent_max_tool_calls)
        server = CorpusMCPServer(
            self.db,
            self.user,
            request_id=self.request_id,
            allowed_document_ids=document_ids,
        )

        candidate_store: dict[str, Candidate] = {}
        history: list[dict] = []
        decisions: list[dict] = []
        tool_traces: list[dict] = []
        input_tokens = 0
        output_tokens = 0
        controller_timings_ms: list[int] = []
        answer_generation_ms = 0
        citation_repair_ms = 0
        final_decision: AgentDecision | None = None
        last_execution: ToolExecution | None = None

        self._emit(
            "agent",
            "Understanding your question",
            "The research agent is deciding the smallest useful corpus search.",
            12,
        )

        for step in range(1, max_steps + 1):
            self._checkpoint()
            controller_started = time.perf_counter()
            try:
                payload_json, result = self.llm.generate_json(
                    system=self._controller_system_prompt(),
                    user=self._controller_user_prompt(
                        question=question,
                        operational_context=operational_context,
                        history=history,
                        step=step,
                        max_steps=max_steps,
                    ),
                    strong=False,
                    max_output_tokens=min(
                        420, self.settings.agent_controller_max_output_tokens
                    ),
                )
                decision = AgentDecision.model_validate(payload_json)
                input_tokens += result.input_tokens
                output_tokens += result.output_tokens
            except Exception:
                # Controller-format failures should not send the request back into the old
                # orchestration graph. Use a safe direct-search/finish fallback inside the
                # agent itself.
                if candidate_store:
                    strongest = sorted(
                        candidate_store.values(),
                        key=self._candidate_rank,
                        reverse=True,
                    )[: min(6, self.settings.agent_max_evidence)]
                    decision = AgentDecision(
                        action="finish",
                        decision_summary=(
                            "Controller output was invalid; use the strongest evidence already gathered."
                        ),
                        evidence_status="partial",
                        selected_chunk_ids=[
                            str(candidate.chunk_id) for candidate in strongest
                        ],
                        unresolved=[],
                    )
                else:
                    decision = AgentDecision(
                        action="search_chunks",
                        decision_summary=(
                            "Controller output was invalid; run one direct hybrid corpus search."
                        ),
                        query=question,
                        evidence_status="insufficient",
                    )
            if (
                decision.action == "finish"
                and decision.selected_chunk_ids
                and not decision.unresolved
                and decision.evidence_status == "insufficient"
            ):
                decision.evidence_status = "sufficient"
            controller_timings_ms.append(
                int((time.perf_counter() - controller_started) * 1000)
            )
            decisions.append(decision.model_dump())

            if decision.action == "finish":
                final_decision = decision
                break

            arguments = self._decision_arguments(decision, question)
            self._emit(
                "agent_search",
                f"Researching source evidence ({step}/{max_steps})",
                f"Using {decision.action.replace('_', ' ')}.",
                min(20 + step * 10, 70),
            )
            execution = server.execute(decision.action, arguments)
            last_execution = execution
            tool_traces.append(execution.trace())

            for candidate in execution.candidates:
                candidate_store[str(candidate.chunk_id)] = candidate

            observation = self._compact_tool_observation(execution)
            observation["controller_summary"] = decision.decision_summary
            history.append(observation)

            # Hard early stop protection: if the model itself called the evidence sufficient
            # on this tool action, one more controller call is unnecessary. It can select the
            # strongest returned evidence deterministically.
            if (
                decision.evidence_status == "sufficient"
                and execution.candidates
            ):
                final_decision = AgentDecision(
                    action="finish",
                    decision_summary="Current tool result directly supports the requested answer.",
                    evidence_status="sufficient",
                    selected_chunk_ids=[
                        str(candidate.chunk_id)
                        for candidate in execution.candidates[
                            : (
                                self.settings.agent_max_evidence
                                if execution.tool == "get_document_structure"
                                else min(6, self.settings.agent_max_evidence)
                            )
                        ]
                    ],
                    unresolved=[],
                )
                break

        if final_decision is None:
            final_decision = AgentDecision(
                action="finish",
                decision_summary="Tool-call budget reached; answer from the best verified evidence gathered.",
                evidence_status="partial" if candidate_store else "insufficient",
                selected_chunk_ids=[],
                unresolved=["Any point not directly supported by the gathered evidence"],
            )

        resolved_focus = self._resolved_research_focus(question, decisions)
        selected_candidates = self._selected_candidates(
            decision=final_decision,
            candidate_store=candidate_store,
            last_execution=last_execution,
            resolved_focus=resolved_focus,
        )
        evidence = [
            Evidence(evidence_id=f"E{index}", candidate=candidate)
            for index, candidate in enumerate(selected_candidates, start=1)
        ]

        self._emit(
            "answer",
            "Drafting from selected evidence",
            f"Using {len(evidence)} evidence item(s) selected by the research agent.",
            82,
        )

        if evidence:
            evidence_block = "\n\n---\n\n".join(
                item.prompt_block()[
                    : (
                        1800
                        if item.candidate.evidence_lane == "document_structure"
                        else 5500
                    )
                ]
                for item in evidence
            )
            structural_headings: list[str] = []
            seen_headings: set[str] = set()
            for item in evidence:
                if item.candidate.section_path:
                    heading = str(item.candidate.section_path[-1]).strip()
                    if heading and heading.casefold() not in seen_headings:
                        seen_headings.add(heading.casefold())
                        structural_headings.append(heading)
                for match in re.finditer(
                    r"(?m)^\s*(\d{1,3}\.\s+[A-Z][^\n]{2,180})",
                    item.candidate.text or "",
                ):
                    heading = " ".join(match.group(1).split())
                    key = heading.casefold()
                    if key in seen_headings:
                        continue
                    seen_headings.add(key)
                    structural_headings.append(heading)
            heading_block = (
                "\n\nSTRUCTURAL HEADING CHECK (source-derived; include relevant headings in section/list answers):\n- "
                + "\n- ".join(structural_headings[:80])
                if structural_headings
                else ""
            )
            answer_user = (
                f"Question:\n{question}\n\n"
                f"Resolved research focus:\n{resolved_focus}\n\n"
                f"Evidence status: {final_decision.evidence_status}\n"
                f"Unresolved points: {json.dumps(final_decision.unresolved, ensure_ascii=False)}\n\n"
                f"DOCUMENTARY EVIDENCE:\n{evidence_block}"
                f"{heading_block}"
            )
            answer_started = time.perf_counter()
            answer_result = self.llm.generate(
                system=self._answer_system_prompt(),
                user=answer_user,
                strong=requested_mode == "research",
                max_output_tokens=self.settings.agent_answer_max_output_tokens,
                on_delta=self.answer_delta,
                cancel_check=self.cancel_check,
            )
            answer_generation_ms = int((time.perf_counter() - answer_started) * 1000)
            input_tokens += answer_result.input_tokens
            output_tokens += answer_result.output_tokens
            answer = answer_result.text.strip()
        else:
            answer = (
                "I do not have enough verified source material to give a reliable answer yet. "
                "A more specific source or additional document context may be needed."
            )

        cited_ids = self._citation_ids(answer, evidence)
        if evidence and not cited_ids:
            # Keep the response source-grounded even if the model omitted citation markers.
            # This is a bounded formatting repair, not another research pass.
            repair_started = time.perf_counter()
            repair = self.llm.generate(
                system=(
                    self._answer_system_prompt()
                    + " Rewrite the draft without changing its substance, adding [E#] citations "
                    "to every factual claim supported by the supplied evidence."
                ),
                user=(
                    f"Question:\n{question}\n\nDraft:\n{answer}\n\nEvidence:\n"
                    + "\n\n---\n\n".join(item.prompt_block()[:4500] for item in evidence)
                ),
                strong=False,
                max_output_tokens=self.settings.agent_answer_max_output_tokens,
                cancel_check=self.cancel_check,
            )
            citation_repair_ms = int((time.perf_counter() - repair_started) * 1000)
            input_tokens += repair.input_tokens
            output_tokens += repair.output_tokens
            answer = repair.text.strip()
            cited_ids = self._citation_ids(answer, evidence)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        confidence = (
            "high"
            if final_decision.evidence_status == "sufficient" and cited_ids
            else "medium"
            if evidence and cited_ids
            else "low"
        )


        query_records: list[dict] = []
        for call in tool_traces:
            arguments = call.get("arguments", {})
            single_query = arguments.get("query")
            if single_query:
                query_records.append(
                    {
                        "query": str(single_query),
                        "tool": call.get("tool"),
                        "items": call.get("items", []),
                    }
                )
            for group in call.get("query_groups", []) or []:
                if not isinstance(group, dict) or not group.get("query"):
                    continue
                query_records.append(
                    {
                        "query": str(group.get("query")),
                        "tool": call.get("tool"),
                        "items": group.get("items", []),
                    }
                )

        trace = {
            "agentic": True,
            "agent_architecture": "bounded_mcp_tool_loop_v1",
            "agent_max_tool_calls": max_steps,
            "agent_decisions": decisions,
            "tool_calls": tool_traces,
            "agent_stop_reason": final_decision.decision_summary,
            "resolved_research_focus": resolved_focus,
            "agent_evidence_status": final_decision.evidence_status,
            "agent_unresolved": final_decision.unresolved,
            "selected_chunk_ids": [str(item.candidate.chunk_id) for item in evidence],
            "queries": [record["query"] for record in query_records],
            "query_specs": [
                {
                    "text": record["query"],
                    "tool": record["tool"],
                    "origins": ["agent"],
                }
                for record in query_records
            ],
            "query_result_previews": [
                {
                    "query_index": index,
                    "query": record["query"],
                    "origins": ["agent", record["tool"]],
                    "goal_ids": [],
                    "goal_kinds": [],
                    "results": record["items"],
                }
                for index, record in enumerate(query_records)
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": str(item.candidate.chunk_id),
                    "document_id": str(item.candidate.document_id),
                    "document_title": item.candidate.document_title,
                    "page": item.candidate.page_from,
                    "section": (
                        item.candidate.section_path[-1]
                        if item.candidate.section_path
                        else None
                    ),
                    "rerank_score": item.candidate.rerank_score,
                    "final_retrieval_score": item.candidate.final_retrieval_score,
                    "rank_method": item.candidate.rank_method,
                    "sources": sorted(item.candidate.sources),
                }
                for item in evidence
            ],
            "workflow_timings_ms": {
                "agent_total": elapsed_ms,
                "controller_calls": controller_timings_ms,
                "tool_calls": [call.get("elapsed_ms", 0) for call in tool_traces],
                "answer_generation": answer_generation_ms,
                "citation_repair": citation_repair_ms,
            },
        }

        return {
            "answer": answer,
            "evidence": evidence,
            "cited_ids": cited_ids,
            "confidence": confidence,
            "resolved_mode": "research" if requested_mode == "research" else "direct",
            "route_reason": "agentic_mcp_tool_loop",
            "retrieval_trace": trace,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "verified": bool(cited_ids),
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
