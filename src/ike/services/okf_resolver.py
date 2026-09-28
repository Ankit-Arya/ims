from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.db.models import Document, User
from ike.retrieval.access import document_access_clause
from ike.retrieval.query_plan import QueryPlan, build_query_plan
from ike.workflows.evidence_planning import EvidencePlan

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_STOP = {
    "a","an","and","are","as","at","be","by","for","from","how","i","if","in","is","it",
    "of","on","or","the","to","what","when","where","which","who","why","with","during",
    "must","should","can","does","do","all","any","their","this","that","these","those",
    "establish","documented","documentary","requested","applicable",
}
_ROUTING_GENERIC = {
    "train","operation","operate","operating","actions","action","conditions","condition",
    "speed","restrictions","restriction","responsible","monitoring","monitor","apply","take",
    "procedure","procedures","requirements","requirement","authority","authorizes","approves",
    "imposes","document","documents","manual","manuals","rule","rules",
}
_ANCHOR_GENERIC = {
    "due","high","low","train","operation","operating","mode","area","condition","conditions",
    "section","fails","failed","failure","isolated","isolation","open","closed","required","apply",
}


@dataclass(slots=True)
class OKFDocumentMatch:
    document_id: UUID
    title: str
    score: float
    reasons: list[str]
def _canon(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def _tokens(value: str | None) -> set[str]:
    result: set[str] = set()
    for match in _TOKEN_RE.finditer(value or ""):
        token = match.group(0).casefold().strip("._-")
        if len(token) >= 3 and token not in _STOP:
            result.add(token)
    return result


def _shared_anchor(evidence_plan: EvidencePlan | None) -> set[str]:
    if evidence_plan is None:
        return set()
    anchors: set[str] = set()
    for goal in evidence_plan.goals:
        for qualifier in goal.qualifiers:
            if qualifier.startswith("shared_context:"):
                anchors.update(_tokens(qualifier.split(":", 1)[1]) - _ANCHOR_GENERIC)
    return anchors


def _related_hits(needles: set[str], haystack: set[str]) -> set[str]:
    hits: set[str] = set()
    for needle in needles:
        for value in haystack:
            if needle == value or (
                min(len(needle), len(value)) >= 5
                and (needle.startswith(value) or value.startswith(needle))
            ):
                hits.add(needle)
                break
    return hits


def _constraint_topic_anchor(question: str, plan: QueryPlan, evidence_plan: EvidencePlan | None) -> set[str]:
    if evidence_plan is None or "constrained_attribute_query" not in evidence_plan.warnings:
        return set()
    terms = _tokens(question) - _ROUTING_GENERIC - _ANCHOR_GENERIC
    removable: set[str] = set()
    for value in [
        plan.source_scope,
        plan.line,
        plan.rolling_stock,
        plan.system,
        plan.subsystem,
    ]:
        removable.update(_tokens(value))
    # Explicit technical identifiers are applicability constraints, not the topic itself.
    removable.update(token.casefold() for token in re.findall(r"\b[A-Z][A-Z0-9_-]{1,15}\b", question))
    return {term for term in terms if term not in removable}


def _query_terms(question: str, plan: QueryPlan, evidence_plan: EvidencePlan | None) -> set[str]:
    # Use literal user wording + deterministic query metadata only. Do not ingest generated
    # goal prose here; a bad planner must never contaminate OKF routing.
    values = [
        question,
        plan.topic,
        plan.scenario,
        plan.system,
        plan.subsystem,
        plan.source_scope,
        *plan.content_terms,
        *plan.scope_terms,
        *plan.entity_terms,
        *plan.exact_terms,
    ]
    terms: set[str] = set()
    for value in values:
        terms.update(_tokens(value))
    anchors = _shared_anchor(evidence_plan)
    return (terms - _ROUTING_GENERIC) | anchors
class OKFResolver:
    """Resolve likely governing/applicable documents from OKF-derived metadata.

    This is an additive routing prior, never a hard scope. Global retrieval remains active.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def resolve(
        self,
        question: str,
        user: User,
        query_plan: QueryPlan,
        evidence_plan: EvidencePlan | None,
        *,
        limit: int = 12,
    ) -> tuple[list[UUID], dict]:
        if (
            evidence_plan is not None
            and evidence_plan.requires_decomposition
            and evidence_plan.strategy == "multi_lookup"
            and any(warning == "multipart_independent_subqueries" for warning in evidence_plan.warnings)
            and len(evidence_plan.goals) > 1
        ):
            per_goal_limit = max(2, min(4, max(1, limit // len(evidence_plan.goals))))
            merged_ids: list[UUID] = []
            merged_matches: list[dict] = []
            seen: set[UUID] = set()
            per_goal_trace: dict[str, dict] = {}
            for goal in evidence_plan.goals:
                sub_plan = EvidencePlan(
                    original=goal.question,
                    strategy="single",
                    entities=list(goal.entity_terms),
                    goals=[goal],
                    requires_decomposition=False,
                    needs_research=False,
                    needs_verification=False,
                    warnings=[],
                )
                sub_qp = build_query_plan(goal.question)
                goal_ids, goal_trace = self.resolve(
                    goal.question,
                    user,
                    sub_qp,
                    sub_plan,
                    limit=per_goal_limit,
                )
                per_goal_trace[goal.id] = goal_trace
                match_by_id = {
                    UUID(item["document_id"]): item
                    for item in goal_trace.get("matches", [])
                    if item.get("document_id")
                }
                for document_id in goal_ids:
                    if document_id in seen:
                        continue
                    seen.add(document_id)
                    merged_ids.append(document_id)
                    if document_id in match_by_id:
                        item = dict(match_by_id[document_id])
                        item["goal_id"] = goal.id
                        merged_matches.append(item)
                    if len(merged_ids) >= limit:
                        break
                if len(merged_ids) >= limit:
                    break
            return merged_ids[:limit], {
                "enabled": True,
                "independent_goal_resolution": True,
                "selected_count": len(merged_ids[:limit]),
                "matches": merged_matches[:limit],
                "per_goal": per_goal_trace,
            }

        terms = _query_terms(question, query_plan, evidence_plan)
        anchors = _shared_anchor(evidence_plan) | _constraint_topic_anchor(question, query_plan, evidence_plan)
        if not terms:
            return [], {"enabled": True, "matches": [], "query_terms": []}

        stmt = select(Document).where(
            document_access_clause(user),
            Document.ingestion_status == "ready",
            Document.lifecycle_status == "active",
        )
        documents = list(self.db.scalars(stmt))
        superseded_ids = {
            document.supersedes_document_id
            for document in documents
            if document.supersedes_document_id is not None
        }
        matches: list[OKFDocumentMatch] = []
        for document in documents:
            metadata = document.extra_metadata or {}
            okf = metadata.get("okf") or {}
            profile = metadata.get("operational_profile") or {}
            if okf.get("status") != "ready":
                continue

            title_tokens = _tokens(f"{document.title} {document.original_filename}")
            family_tokens = _tokens(document.family_key)
            authority_tokens = _tokens(document.authority)
            role_tokens = _tokens(document.source_role)
            type_tokens = _tokens(profile.get("document_type"))
            title_compact = _canon(f"{document.title} {document.original_filename}")

            reasons: list[str] = []
            score = 0.0
            rolling_stock_title_match = bool(
                query_plan.rolling_stock
                and _canon(query_plan.rolling_stock) in title_compact
            )

            anchor_surface = title_tokens | family_tokens | role_tokens | type_tokens | ({_canon(query_plan.rolling_stock)} if rolling_stock_title_match and query_plan.rolling_stock else set())
            anchor_hits = _related_hits(anchors, anchor_surface) if anchors else set()
            if anchors:
                if anchor_hits:
                    score += 6.0
                    reasons.append("shared_context:" + ",".join(sorted(anchor_hits)))
                else:
                    score -= 3.5
                    reasons.append("shared_context_miss")

            title_hits = _related_hits(terms, title_tokens)
            if title_hits:
                score += min(4.0, 1.35 * len(title_hits))
                reasons.append("title:" + ",".join(sorted(title_hits)[:5]))

            family_hits = terms & family_tokens
            if family_hits:
                score += min(2.0, 0.9 * len(family_hits))
                reasons.append("family:" + ",".join(sorted(family_hits)[:4]))

            role_hits = terms & role_tokens
            if role_hits:
                score += min(1.25, 0.65 * len(role_hits))
                reasons.append("source_role:" + ",".join(sorted(role_hits)[:3]))

            type_hits = terms & type_tokens
            if type_hits:
                score += min(1.0, 0.5 * len(type_hits))
                reasons.append("document_type:" + ",".join(sorted(type_hits)[:3]))

            authority_hits = terms & authority_tokens
            if authority_hits:
                score += min(0.75, 0.35 * len(authority_hits))
                reasons.append("authority:" + ",".join(sorted(authority_hits)[:3]))
            if query_plan.line:
                line_values = [
                    *profile.get("primary_line_codes", []),
                    *profile.get("line_codes", []),
                ]
                if line_values:
                    if any(_canon(str(value)) == _canon(query_plan.line) for value in line_values):
                        score += 2.0
                        reasons.append("line_match")
                    else:
                        score -= 1.25
                        reasons.append("line_mismatch")

            if query_plan.rolling_stock:
                stock_values = [
                    *profile.get("primary_rolling_stock", []),
                    *profile.get("rolling_stock", []),
                ]
                if stock_values:
                    if any(_canon(str(value)) == _canon(query_plan.rolling_stock) for value in stock_values):
                        score += 2.5
                        reasons.append("rolling_stock_match")
                    else:
                        score -= 1.5
                        reasons.append("rolling_stock_mismatch")

            if document.id in superseded_ids:
                score -= 3.0
                reasons.append("superseded")

            if document.authority_level is not None and score > 0:
                score += min(0.6, max(0.0, float(document.authority_level) / 200.0))
                reasons.append("authority_level")

            # Require actual semantic/identity overlap. For shared-scenario multipart
            # questions, generic document overlap is insufficient without the scenario anchor.
            # Applicability metadata (line/rolling stock) may boost a relevant document,
            # but cannot make an unrelated document relevant by itself.
            identity_overlap = bool(anchor_hits or title_hits or family_hits or type_hits or role_hits)
            anchor_ok = (not anchors) or bool(anchor_hits)
            if score >= 1.0 and identity_overlap and anchor_ok:
                matches.append(OKFDocumentMatch(document.id, document.title, score, reasons))

        matches.sort(key=lambda item: item.score, reverse=True)
        selected = matches[: max(1, limit)]
        return [item.document_id for item in selected], {
            "enabled": True,
            "query_terms": sorted(terms)[:40],
            "match_count": len(matches),
            "selected_count": len(selected),
            "matches": [
                {
                    "document_id": str(item.document_id),
                    "title": item.title,
                    "score": round(item.score, 4),
                    "reasons": item.reasons,
                }
                for item in selected
            ],
        }
