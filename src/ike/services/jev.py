from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ike.core.config import Settings, get_settings
from ike.retrieval.table_context import retrieval_text
from ike.retrieval.types import Candidate

logger = logging.getLogger(__name__)

_FIT_LEVELS = [
    "0 - irrelevant or wrong scope; does not help answer this exact requirement",
    "1 - loosely related background; not directly usable as evidence for this requirement",
    "2 - partially useful; contains some relevant evidence but is incomplete, indirect, or scope-uncertain",
    "3 - strong direct evidence; applicable to the requested scenario and supports most of the requirement",
    "4 - exact, directly citable evidence for the requirement, with the correct operational scope",
]


@dataclass(slots=True)
class JevJudgment:
    fit: float
    confidence: float

    @property
    def score(self) -> float:
        # Keep the ranking signal on a stable 0..1 scale. Confidence is traced
        # separately so calibration can be evaluated without double-counting it.
        return self.fit


class JevEvidenceJudge:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    @property
    def configured(self) -> bool:
        return self.settings.jev_mode != "off" and bool(self.settings.typesafe_api_key.strip())

    @property
    def endpoint_url(self) -> str:
        key = self.settings.typesafe_api_key.strip()
        provider = getattr(self.settings, "jev_provider", "auto")
        if provider == "hosted" or (provider == "auto" and key.startswith("jv_live_")):
            return "https://jevtypesafeai.com/api/v1/decide"
        return self.settings.jev_base_url.rstrip("/") + "/v1/systemone"

    @staticmethod
    def _goal_key(goal_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", goal_id)

    @staticmethod
    def _candidate_goal_ids(candidate: Candidate) -> list[str]:
        values: list[str] = []
        for source in candidate.sources:
            match = re.fullmatch(r"goal:([^:]+)", source)
            if match:
                values.append(match.group(1))
        return sorted(set(values))

    @staticmethod
    def _parse_score(answers: dict, key: str) -> JevJudgment:
        value = answers.get(key, {})
        try:
            raw = float(value.get("score", 0.0))
        except (TypeError, ValueError, AttributeError):
            raw = 0.0
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError, AttributeError):
            confidence = 0.0
        return JevJudgment(
            fit=max(0.0, min(1.0, raw / 4.0)),
            confidence=max(0.0, min(1.0, confidence)),
        )

    @staticmethod
    def _fit_question(prefix: str, target: str) -> dict[str, dict]:
        return {
            f"fit_{prefix}": {
                "type": "score",
                "instructions": (
                    "Rate how well this candidate passage can serve as evidence for the exact requirement below. "
                    "Be strict: generic operational text, merely related topics, wrong line/rolling stock/role/revision, "
                    "different failure modes, or background that cannot directly support the answer must stay low. "
                    f"Requirement: {target}"
                ),
                "criteria": _FIT_LEVELS,
            }
        }

    def _request(
        self,
        question: str,
        candidates: list[Candidate],
        evidence_plan: Any | None = None,
        goal_ids: set[str] | None = None,
    ) -> tuple[dict[str, JevJudgment], dict[str, dict[str, float]], dict]:
        limited = candidates[: max(1, self.settings.jev_max_candidates)]
        active_goals = []
        if evidence_plan is not None:
            active_goals = [
                goal for goal in evidence_plan.goals
                if goal_ids is None or goal.id in goal_ids
            ]
        complex_mode = bool(
            getattr(self.settings, "jev_complex_enabled", True)
            and active_goals
            and (getattr(evidence_plan, "requires_decomposition", False) or len(active_goals) > 1)
        )

        candidate_chars = (
            getattr(self.settings, "jev_complex_candidate_chars", 1800)
            if complex_mode
            else self.settings.jev_candidate_chars
        )
        state_candidates = [{
            "id": str(candidate.chunk_id),
            "document": candidate.document_title,
            "revision": candidate.revision,
            "authority": candidate.authority,
            "section": candidate.section_path,
            "page": candidate.page_from,
            "existing_rank_method": candidate.rank_method,
            "existing_score": candidate.final_retrieval_score,
            "goal_tags": self._candidate_goal_ids(candidate),
            "text": retrieval_text(candidate)[:candidate_chars],
        } for candidate in limited]

        questions: dict[str, dict] = {}
        pair_map: list[tuple[int, str, str]] = []
        if complex_mode:
            goal_by_id = {goal.id: goal for goal in active_goals}
            max_pairs = max(1, getattr(self.settings, "jev_max_goal_pairs", 24))
            max_per_candidate = max(1, getattr(self.settings, "jev_max_goals_per_candidate", 2))
            for index, candidate in enumerate(limited):
                candidate_goal_ids = [
                    goal_id for goal_id in self._candidate_goal_ids(candidate)
                    if goal_id in goal_by_id
                ]
                if not candidate_goal_ids:
                    scored_goal_ids = [
                        goal.id for goal in active_goals
                        if goal.id in candidate.goal_rerank_scores
                    ]
                    candidate_goal_ids = scored_goal_ids or [
                        goal.id for goal in active_goals if goal.required
                    ]
                for goal_id in candidate_goal_ids[:max_per_candidate]:
                    if len(pair_map) >= max_pairs:
                        break
                    goal = goal_by_id[goal_id]
                    prefix = f"{index}_{self._goal_key(goal_id)}"
                    questions.update(self._fit_question(prefix, goal.question))
                    pair_map.append((index, goal_id, prefix))
                if len(pair_map) >= max_pairs:
                    break
        else:
            for index, _candidate in enumerate(limited):
                questions.update(self._fit_question(str(index), question))

        payload = {
            "model": self.settings.jev_model,
            "state": {
                "question": question,
                "complex_query": complex_mode,
                "evidence_goals": [{
                    "id": goal.id,
                    "question": goal.question,
                    "kind": goal.kind,
                    "required": goal.required,
                    "qualifiers": goal.qualifiers,
                    "coverage_contract": goal.coverage_contract,
                } for goal in active_goals],
                "candidates": state_candidates,
            },
            "questions": questions,
        }

        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=self.settings.jev_timeout_seconds)
        try:
            response = client.post(
                self.endpoint_url,
                headers={"Authorization": f"Bearer {self.settings.typesafe_api_key}"},
                json=payload,
            )
            if response.is_error:
                body = response.text[:1200]
                raise RuntimeError(f"Jev HTTP {response.status_code}: {body}")
            body = response.json()
        finally:
            if owns_client:
                client.close()

        answers = body.get("answers", {})
        judgments: dict[str, JevJudgment] = {}
        goal_scores: dict[str, dict[str, float]] = {}

        if complex_mode:
            per_candidate: dict[int, list[tuple[str, JevJudgment]]] = {}
            for index, goal_id, prefix in pair_map:
                judgment = self._parse_score(answers, f"fit_{prefix}")
                per_candidate.setdefault(index, []).append((goal_id, judgment))
                goal_scores.setdefault(str(limited[index].chunk_id), {})[goal_id] = judgment.score

            for index, candidate in enumerate(limited):
                rows = per_candidate.get(index, [])
                if rows:
                    _goal_id, best = max(rows, key=lambda row: row[1].score)
                    judgments[str(candidate.chunk_id)] = best
        else:
            for index, candidate in enumerate(limited):
                judgments[str(candidate.chunk_id)] = self._parse_score(answers, f"fit_{index}")

        return judgments, goal_scores, {
            "provider": "hosted" if self.endpoint_url.startswith("https://jevtypesafeai.com/") else "typesafe",
            "endpoint": self.endpoint_url,
            "model": body.get("model", self.settings.jev_model),
            "candidate_count": len(limited),
            "complex_mode": complex_mode,
            "goal_count": len(active_goals),
            "goal_pair_count": len(pair_map),
            "question_count": len(questions),
            "usage": body.get("usage", {}),
        }

    def judge_and_apply(
        self,
        question: str,
        candidates: list[Candidate],
        *,
        evidence_plan: Any | None = None,
        goal_ids: set[str] | None = None,
    ) -> tuple[list[Candidate], dict]:
        if not self.configured or not candidates:
            return candidates, {"mode": self.settings.jev_mode, "called": False}
        try:
            judgments, goal_scores, meta = self._request(
                question, candidates, evidence_plan=evidence_plan, goal_ids=goal_ids
            )
        except Exception as exc:
            logger.exception("jev_evidence_judgment_failed")
            if not self.settings.jev_fail_open:
                raise
            return candidates, {
                "mode": self.settings.jev_mode,
                "called": True,
                "failed": True,
                "error": str(exc)[:1200],
            }

        original_scores = [float(c.final_retrieval_score) for c in candidates]
        lo = min(original_scores)
        hi = max(original_scores)
        denom = hi - lo
        total = max(1, len(candidates) - 1)
        details: list[dict] = []

        for index, candidate in enumerate(candidates):
            judgment = judgments.get(str(candidate.chunk_id))
            if judgment is None:
                continue
            candidate.judge_score = judgment.score
            candidate.judge_details = {
                "fit": judgment.fit,
                "confidence": judgment.confidence,
            }
            candidate.sources.add("jev_judged" if self.settings.jev_mode == "apply" else "jev_shadow")
            per_goal = goal_scores.get(str(candidate.chunk_id), {})

            if denom > 1e-9:
                base = (candidate.final_retrieval_score - lo) / denom
            else:
                base = 1.0 - index / total

            if self.settings.jev_mode == "apply":
                weight = max(0.0, min(1.0, self.settings.jev_weight))
                candidate.final_retrieval_score = (1.0 - weight) * base + weight * judgment.score
                candidate.rank_method = f"{candidate.rank_method}+jev"
                for goal_id, score in per_goal.items():
                    candidate.goal_rerank_scores[goal_id] = max(
                        candidate.goal_rerank_scores.get(goal_id, 0.0), score
                    )

            details.append({
                "chunk_id": str(candidate.chunk_id),
                "score": round(judgment.score, 6),
                "fit": round(judgment.fit, 6),
                "confidence": round(judgment.confidence, 6),
                "goal_scores": {
                    goal_id: round(score, 6) for goal_id, score in per_goal.items()
                },
            })

        if self.settings.jev_mode == "apply":
            candidates = sorted(candidates, key=lambda c: c.final_retrieval_score, reverse=True)

        return candidates, {
            "mode": self.settings.jev_mode,
            "called": True,
            **meta,
            "judgments": details,
        }
