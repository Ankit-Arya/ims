from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx

from ike.core.config import get_settings
from ike.services.query_control import QueryCancelled


@dataclass(slots=True)
class InferenceTiming:
    queue_wait_ms: int = 0
    execution_ms: int = 0
    details: dict | None = None


@dataclass(slots=True)
class RerankResult:
    index: int
    score: float


@dataclass(slots=True)
class RerankGroupResult:
    group_id: str
    results: list[RerankResult]
    candidate_count: int = 0


class InferenceClient:
    """HTTP client for one local inference plane.

    The query and ingestion planes use different base URLs in 0.5.0, so document
    embedding can never block live query embedding through a shared process lock.
    """

    def __init__(self, *, plane: Literal["query", "ingest"] = "query") -> None:
        settings = get_settings()
        self.plane = plane
        self.base_url = (
            settings.query_inference_url if plane == "query" else settings.ingestion_inference_url
        ).rstrip("/")
        self.headers = {"X-Internal-Token": settings.internal_service_token}
        self.timeout = httpx.Timeout(300.0, connect=10.0)
        self.client = httpx.Client(timeout=self.timeout, headers=self.headers)

    @classmethod
    def for_query(cls) -> "InferenceClient":
        return cls(plane="query")

    @classmethod
    def for_ingestion(cls) -> "InferenceClient":
        return cls(plane="ingest")

    def close(self) -> None:
        self.client.close()

    def _post(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.post(f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise last_error

    def embed_queries(self, texts: list[str]) -> tuple[list[list[float]], InferenceTiming]:
        if not texts:
            return [], InferenceTiming()
        data = self._post("/embed/query", {"texts": texts})
        return data["vectors"], InferenceTiming(
            queue_wait_ms=int(data.get("queue_wait_ms", 0) or 0),
            execution_ms=int(data.get("execution_ms", 0) or 0),
        )

    def embed_query(self, text: str) -> list[float]:
        vectors, _timing = self.embed_queries([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], InferenceTiming]:
        if not texts:
            return [], InferenceTiming()
        data = self._post("/embed/documents", {"texts": texts})
        return data["vectors"], InferenceTiming(
            queue_wait_ms=int(data.get("queue_wait_ms", 0) or 0),
            execution_ms=int(data.get("execution_ms", 0) or 0),
        )


    def rerank_many(
        self,
        groups: list[tuple[str, str, list[str], int]],
        *,
        request_id: str | None = None,
    ) -> tuple[dict[str, RerankGroupResult], InferenceTiming]:
        """Rerank multiple independent evidence goals in one inference-plane request."""
        if not groups:
            return {}, InferenceTiming()
        payload_groups = []
        for group_id, query, candidates, top_k in groups:
            if not candidates:
                continue
            payload_groups.append(
                {
                    "group_id": group_id,
                    "query": query,
                    "candidates": candidates,
                    "top_k": min(max(1, top_k), len(candidates)),
                }
            )
        if not payload_groups:
            return {}, InferenceTiming()
        try:
            data = self._post("/rerank/batch", {"groups": payload_groups, "request_id": request_id})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise QueryCancelled("Query cancelled during batch reranking") from exc
            if exc.response.status_code in {404, 405}:
                # Rolling-deploy compatibility: a 0.9 API can temporarily coexist with the
                # previous inference-query image. Preserve correctness with serial reranks
                # until the new batch endpoint is available.
                grouped: dict[str, RerankGroupResult] = {}
                queue_wait_ms = 0
                execution_ms = 0
                for group_id, query, candidates, top_k in groups:
                    results, timing = self.rerank(
                        query, candidates, top_k=top_k, request_id=request_id
                    )
                    grouped[group_id] = RerankGroupResult(
                        group_id=group_id,
                        candidate_count=len(candidates),
                        results=results,
                    )
                    queue_wait_ms += timing.queue_wait_ms
                    execution_ms += timing.execution_ms
                return grouped, InferenceTiming(
                    queue_wait_ms=queue_wait_ms,
                    execution_ms=execution_ms,
                    details={"fallback": "serial_rerank_legacy_inference"},
                )
            raise
        grouped: dict[str, RerankGroupResult] = {}
        for group in data.get("groups", []):
            grouped[str(group.get("group_id"))] = RerankGroupResult(
                group_id=str(group.get("group_id")),
                candidate_count=int(group.get("candidate_count", 0) or 0),
                results=[RerankResult(index=item["index"], score=item["score"]) for item in group.get("results", [])],
            )
        return grouped, InferenceTiming(
            queue_wait_ms=int(data.get("queue_wait_ms", 0) or 0),
            execution_ms=int(data.get("execution_ms", 0) or 0),
            details={
                key: data.get(key)
                for key in (
                    "backend", "group_count", "pair_count", "batch_size", "batch_count",
                    "candidate_tokens_p50", "candidate_tokens_p95", "tokenization_ms",
                )
                if key in data
            },
        )

    def cancel(self, request_id: str) -> None:
        if self.plane != "query":
            return
        try:
            self.client.post(f"{self.base_url}/cancel/{request_id}").raise_for_status()
        except Exception:
            pass

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int | None = None,
        request_id: str | None = None,
    ) -> tuple[list[RerankResult], InferenceTiming]:
        if not candidates:
            return [], InferenceTiming()
        payload = {"query": query, "candidates": candidates, "top_k": top_k or len(candidates), "request_id": request_id}
        try:
            data = self._post("/rerank", payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise QueryCancelled("Query cancelled during reranking") from exc
            raise
        results = [RerankResult(index=item["index"], score=item["score"]) for item in data["results"]]
        timing = InferenceTiming(
            queue_wait_ms=int(data.get("queue_wait_ms", 0) or 0),
            execution_ms=int(data.get("execution_ms", 0) or 0),
            details={
                key: data.get(key)
                for key in (
                    "backend", "candidate_count", "top_k", "batch_size", "batch_count",
                    "candidate_tokens_min", "candidate_tokens_p50", "candidate_tokens_p95",
                    "candidate_tokens_max", "tokenization_ms",
                )
                if key in data
            },
        )
        return results, timing
