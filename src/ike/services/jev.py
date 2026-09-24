from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ike.core.config import Settings, get_settings
from ike.retrieval.table_context import retrieval_text
from ike.retrieval.types import Candidate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JevJudgment:
    relevance: float
    applicability: float
    support: float

    @property
    def score(self) -> float:
        return 0.50 * self.relevance + 0.30 * self.applicability + 0.20 * self.support


class JevEvidenceJudge:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    @property
    def configured(self) -> bool:
        return self.settings.jev_mode != "off" and bool(self.settings.typesafe_api_key.strip())
    @staticmethod
    def _noul(answers: dict, key: str) -> float:
        value = answers.get(key, {})
        try:
            return max(0.0, min(1.0, float(value.get("noul", 0.0))))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _request(self, question: str, candidates: list[Candidate]) -> tuple[dict[str, JevJudgment], dict]:
        limited = candidates[: max(1, self.settings.jev_max_candidates)]
        state_candidates = []
        questions: dict[str, dict] = {}
        for index, candidate in enumerate(limited):
            state_candidates.append({
                "id": str(candidate.chunk_id),
                "document": candidate.document_title,
                "revision": candidate.revision,
                "authority": candidate.authority,
                "section": candidate.section_path,
                "page": candidate.page_from,
                "existing_rank_method": candidate.rank_method,
                "existing_score": candidate.final_retrieval_score,
                "text": retrieval_text(candidate)[: self.settings.jev_candidate_chars],
            })
            questions[f"relevance_{index}"] = {
                "type": "noul",
                "instructions": "Does this candidate directly help answer the user's question?",
            }
            questions[f"applicability_{index}"] = {
                "type": "noul",
                "instructions": (
                    "Is this candidate applicable to the scope expressed by the question, including any role, "
                    "line, rolling-stock, document, revision, or operational-condition constraints?"
                ),
            }
            questions[f"support_{index}"] = {
                "type": "noul",
                "instructions": "Does this passage contain concrete source evidence that can support a factual answer?",
            }

        payload = {
            "model": self.settings.jev_model,
            "state": {"question": question, "candidates": state_candidates},
            "questions": questions,
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=self.settings.jev_timeout_seconds)
        try:
            response = client.post(
                self.settings.jev_base_url.rstrip("/") + "/v1/systemone",
                headers={"Authorization": f"Bearer {self.settings.typesafe_api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        finally:
            if owns_client:
                client.close()
        answers = body.get("answers", {})
        judgments: dict[str, JevJudgment] = {}
        for index, candidate in enumerate(limited):
            judgments[str(candidate.chunk_id)] = JevJudgment(
                relevance=self._noul(answers, f"relevance_{index}"),
                applicability=self._noul(answers, f"applicability_{index}"),
                support=self._noul(answers, f"support_{index}"),
            )
        return judgments, {
            "model": body.get("model", self.settings.jev_model),
            "candidate_count": len(limited),
            "usage": body.get("usage", {}),
        }

    def judge_and_apply(self, question: str, candidates: list[Candidate]) -> tuple[list[Candidate], dict]:
        if not self.configured or not candidates:
            return candidates, {"mode": self.settings.jev_mode, "called": False}
        try:
            judgments, meta = self._request(question, candidates)
        except Exception as exc:
            logger.exception("jev_evidence_judgment_failed")
            if not self.settings.jev_fail_open:
                raise
            return candidates, {
                "mode": self.settings.jev_mode,
                "called": True,
                "failed": True,
                "error": str(exc)[:500],
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
                "relevance": judgment.relevance,
                "applicability": judgment.applicability,
                "support": judgment.support,
            }
            candidate.sources.add("jev_judged" if self.settings.jev_mode == "apply" else "jev_shadow")
            if denom > 1e-9:
                base = (candidate.final_retrieval_score - lo) / denom
            else:
                base = 1.0 - index / total
            if self.settings.jev_mode == "apply":
                weight = max(0.0, min(1.0, self.settings.jev_weight))
                candidate.final_retrieval_score = (1.0 - weight) * base + weight * judgment.score
                candidate.rank_method = f"{candidate.rank_method}+jev"
            details.append({
                "chunk_id": str(candidate.chunk_id),
                "score": round(judgment.score, 6),
                **candidate.judge_details,
            })
        if self.settings.jev_mode == "apply":
            candidates = sorted(candidates, key=lambda c: c.final_retrieval_score, reverse=True)

        return candidates, {
            "mode": self.settings.jev_mode,
            "called": True,
            **meta,
            "judgments": details,
        }
