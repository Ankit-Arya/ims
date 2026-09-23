from __future__ import annotations

import time
import math
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal
from threading import Lock

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import CrossEncoder, SentenceTransformer

from inference_service.backends import load_embedding_model, load_rerank_model
from inference_service.runtime import TimedGate


class InferenceSettings(BaseSettings):
    """Configuration for one isolated local ML plane."""

    model_config = SettingsConfigDict(extra="ignore")

    internal_service_token: str = Field(min_length=24)
    inference_role: Literal["query", "ingest"] = "query"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_max_seq_length: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 1024
    rerank_backend: Literal["torch", "onnx", "openvino"] = "torch"
    ml_device: str = "cpu"
    ml_batch_size: int = 8
    ml_num_threads: int = 0
    embed_concurrency: int = 1
    rerank_concurrency: int = 1

    @field_validator("embedding_dim")
    @classmethod
    def fixed_schema_dimension(cls, value: int) -> int:
        if value != 1024:
            raise ValueError(
                "This release's database schema is fixed at 1024 dimensions. "
                "Rebuild the vector migration before changing EMBEDDING_DIM."
            )
        return value


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()


settings = get_inference_settings()
embedding_model: SentenceTransformer | None = None
rerank_model: CrossEncoder | None = None
embed_gate = TimedGate(stage="embed", role=settings.inference_role, concurrency=settings.embed_concurrency)
rerank_gate = TimedGate(stage="rerank", role=settings.inference_role, concurrency=settings.rerank_concurrency)
cancelled_requests: set[str] = set()
cancel_lock = Lock()


class TextBatch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


class VectorBatch(BaseModel):
    vectors: list[list[float]]
    model: str
    dimension: int
    queue_wait_ms: int = 0
    execution_ms: int = 0


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=6000)
    candidates: list[str] = Field(min_length=1, max_length=200)
    top_k: int = Field(default=20, ge=1, le=200)
    request_id: str | None = Field(default=None, max_length=80)


class RerankGroup(BaseModel):
    group_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=6000)
    candidates: list[str] = Field(min_length=1, max_length=200)
    top_k: int = Field(default=20, ge=1, le=200)


class RerankBatchRequest(BaseModel):
    groups: list[RerankGroup] = Field(min_length=1, max_length=32)
    request_id: str | None = Field(default=None, max_length=80)


def internal_auth(x_internal_token: str | None = Header(default=None)) -> None:
    if x_internal_token != settings.internal_service_token:
        raise HTTPException(status_code=401, detail="Invalid internal service token")


def require_role(role: Literal["query", "ingest"]) -> None:
    if settings.inference_role != role:
        raise HTTPException(
            status_code=404,
            detail=f"Endpoint is not enabled on the {settings.inference_role} inference plane",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedding_model, rerank_model
    if settings.ml_num_threads > 0:
        torch.set_num_threads(settings.ml_num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, settings.ml_num_threads)))
        except RuntimeError:
            pass

    embedding_model = load_embedding_model(
        settings.embedding_model,
        device=settings.ml_device,
        max_seq_length=settings.embedding_max_seq_length,
    )
    dimension = embedding_model.get_sentence_embedding_dimension()
    if dimension != settings.embedding_dim:
        raise RuntimeError(
            f"Configured embedding dimension {settings.embedding_dim} does not match model dimension {dimension}"
        )

    if settings.inference_role == "query":
        rerank_model = load_rerank_model(
            settings.rerank_model,
            device=settings.ml_device,
            max_length=settings.rerank_max_length,
            backend=settings.rerank_backend,
        )
    else:
        rerank_model = None
    yield


app = FastAPI(title="IKE Local Inference", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    embedding_ready = embedding_model is not None
    rerank_ready = settings.inference_role == "ingest" or rerank_model is not None
    return {
        "status": "ready" if embedding_ready and rerank_ready else "loading",
        "role": settings.inference_role,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model if settings.inference_role == "query" else None,
        "rerank_backend": settings.rerank_backend if settings.inference_role == "query" else None,
        "device": settings.ml_device,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _embed(texts: list[str]) -> VectorBatch:
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    with embed_gate.acquire() as timing:
        started = time.perf_counter()
        vectors = embedding_model.encode(
            texts,
            batch_size=settings.ml_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        timing.execution_ms = int((time.perf_counter() - started) * 1000)
    return VectorBatch(
        vectors=vectors,
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
        queue_wait_ms=timing.queue_wait_ms,
        execution_ms=timing.execution_ms,
    )


@app.post("/embed/query", response_model=VectorBatch, dependencies=[Depends(internal_auth)])
def embed_query(payload: TextBatch) -> VectorBatch:
    require_role("query")
    return _embed(payload.texts)


@app.post("/embed/documents", response_model=VectorBatch, dependencies=[Depends(internal_auth)])
def embed_documents(payload: TextBatch) -> VectorBatch:
    require_role("ingest")
    return _embed(payload.texts)


@app.post("/cancel/{request_id}", dependencies=[Depends(internal_auth)])
def cancel_request(request_id: str) -> dict:
    require_role("query")
    with cancel_lock:
        cancelled_requests.add(request_id)
    return {"status": "cancel_requested", "request_id": request_id}


def _is_cancelled(request_id: str | None) -> bool:
    if not request_id:
        return False
    with cancel_lock:
        return request_id in cancelled_requests


@app.post("/rerank", dependencies=[Depends(internal_auth)])
def rerank(payload: RerankRequest) -> dict:
    require_role("query")
    if rerank_model is None:
        raise HTTPException(status_code=503, detail="Reranker not loaded")

    # Estimate candidate lengths using the same tokenizer. Sorting similarly sized
    # sequences into batches reduces padding waste on CPU while preserving exact model
    # scores and final ranking semantics.
    token_started = time.perf_counter()
    tokenizer = getattr(rerank_model, "tokenizer", None)
    token_lengths: list[int] = []
    for candidate in payload.candidates:
        if tokenizer is None:
            token_lengths.append(min(settings.rerank_max_length, max(1, len(candidate) // 4)))
            continue
        try:
            encoded = tokenizer(
                payload.query,
                candidate,
                truncation=True,
                max_length=settings.rerank_max_length,
                add_special_tokens=True,
            )
            token_lengths.append(len(encoded.get("input_ids", [])))
        except Exception:
            token_lengths.append(min(settings.rerank_max_length, max(1, len(candidate) // 4)))
    tokenization_ms = int((time.perf_counter() - token_started) * 1000)

    ordered = sorted(range(len(payload.candidates)), key=lambda idx: token_lengths[idx])
    scores_by_index: dict[int, float] = {}
    batch_count = 0
    with rerank_gate.acquire() as timing:
        model_started = time.perf_counter()
        batch_size = max(1, settings.ml_batch_size)
        for offset in range(0, len(ordered), batch_size):
            if _is_cancelled(payload.request_id):
                if payload.request_id:
                    with cancel_lock:
                        cancelled_requests.discard(payload.request_id)
                raise HTTPException(status_code=409, detail="Query cancelled")
            indices = ordered[offset : offset + batch_size]
            pairs = [(payload.query, payload.candidates[idx]) for idx in indices]
            batch_scores = rerank_model.predict(
                pairs,
                batch_size=len(pairs),
                show_progress_bar=False,
            )
            batch_count += 1
            for idx, score in zip(indices, batch_scores, strict=True):
                scores_by_index[idx] = float(score)
        model_execution_ms = int((time.perf_counter() - model_started) * 1000)
        timing.execution_ms = model_execution_ms

    ranked = sorted(scores_by_index.items(), key=lambda item: item[1], reverse=True)
    top = ranked[: min(payload.top_k, len(ranked))]
    lengths = sorted(token_lengths)
    def percentile(p: float) -> int:
        if not lengths:
            return 0
        index = min(len(lengths) - 1, max(0, math.ceil(p * len(lengths)) - 1))
        return int(lengths[index])

    if payload.request_id:
        with cancel_lock:
            cancelled_requests.discard(payload.request_id)
    return {
        "model": settings.rerank_model,
        "backend": settings.rerank_backend,
        "candidate_count": len(payload.candidates),
        "top_k": min(payload.top_k, len(payload.candidates)),
        "batch_size": max(1, settings.ml_batch_size),
        "batch_count": batch_count,
        "candidate_tokens_min": min(lengths) if lengths else 0,
        "candidate_tokens_p50": percentile(0.50),
        "candidate_tokens_p95": percentile(0.95),
        "candidate_tokens_max": max(lengths) if lengths else 0,
        "tokenization_ms": tokenization_ms,
        "queue_wait_ms": timing.queue_wait_ms,
        "execution_ms": timing.execution_ms,
        "results": [{"index": idx, "score": score} for idx, score in top],
    }

@app.post("/rerank/batch", dependencies=[Depends(internal_auth)])
def rerank_batch(payload: RerankBatchRequest) -> dict:
    """Rerank independent query groups in one cross-encoder scheduling window.

    All query/candidate pairs are length-sorted together so CPU batches spend less time on
    padding and multiple evidence goals no longer queue as serial HTTP/model invocations.
    """
    require_role("query")
    if rerank_model is None:
        raise HTTPException(status_code=503, detail="Reranker not loaded")

    pairs_meta: list[tuple[int, int, str, str]] = []
    for group_index, group in enumerate(payload.groups):
        for candidate_index, candidate in enumerate(group.candidates):
            pairs_meta.append((group_index, candidate_index, group.query, candidate))
    if not pairs_meta:
        return {"groups": [], "queue_wait_ms": 0, "execution_ms": 0, "tokenization_ms": 0}

    token_started = time.perf_counter()
    tokenizer = getattr(rerank_model, "tokenizer", None)
    lengths: list[int] = []
    for _group_index, _candidate_index, query, candidate in pairs_meta:
        if tokenizer is None:
            lengths.append(min(settings.rerank_max_length, max(1, (len(query) + len(candidate)) // 4)))
            continue
        try:
            encoded = tokenizer(
                query, candidate, truncation=True, max_length=settings.rerank_max_length, add_special_tokens=True
            )
            lengths.append(len(encoded.get("input_ids", [])))
        except Exception:
            lengths.append(min(settings.rerank_max_length, max(1, (len(query) + len(candidate)) // 4)))
    tokenization_ms = int((time.perf_counter() - token_started) * 1000)

    ordered = sorted(range(len(pairs_meta)), key=lambda idx: lengths[idx])
    scores_by_pair: dict[int, float] = {}
    batch_count = 0
    with rerank_gate.acquire() as timing:
        model_started = time.perf_counter()
        batch_size = max(1, settings.ml_batch_size)
        for offset in range(0, len(ordered), batch_size):
            if _is_cancelled(payload.request_id):
                if payload.request_id:
                    with cancel_lock:
                        cancelled_requests.discard(payload.request_id)
                raise HTTPException(status_code=409, detail="Query cancelled")
            indices = ordered[offset : offset + batch_size]
            model_pairs = [(pairs_meta[idx][2], pairs_meta[idx][3]) for idx in indices]
            batch_scores = rerank_model.predict(model_pairs, batch_size=len(model_pairs), show_progress_bar=False)
            batch_count += 1
            for idx, score in zip(indices, batch_scores, strict=True):
                scores_by_pair[idx] = float(score)
        timing.execution_ms = int((time.perf_counter() - model_started) * 1000)

    grouped: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(payload.groups))}
    for pair_index, score in scores_by_pair.items():
        group_index, candidate_index, _query, _candidate = pairs_meta[pair_index]
        grouped[group_index].append((candidate_index, score))

    output_groups = []
    for group_index, group in enumerate(payload.groups):
        ranked = sorted(grouped[group_index], key=lambda item: item[1], reverse=True)
        top = ranked[: min(group.top_k, len(ranked))]
        output_groups.append(
            {
                "group_id": group.group_id,
                "candidate_count": len(group.candidates),
                "top_k": min(group.top_k, len(group.candidates)),
                "results": [{"index": idx, "score": score} for idx, score in top],
            }
        )

    if payload.request_id:
        with cancel_lock:
            cancelled_requests.discard(payload.request_id)
    ordered_lengths = sorted(lengths)
    def percentile(p: float) -> int:
        if not ordered_lengths:
            return 0
        index = min(len(ordered_lengths) - 1, max(0, math.ceil(p * len(ordered_lengths)) - 1))
        return int(ordered_lengths[index])

    return {
        "model": settings.rerank_model,
        "backend": settings.rerank_backend,
        "group_count": len(payload.groups),
        "pair_count": len(pairs_meta),
        "batch_size": max(1, settings.ml_batch_size),
        "batch_count": batch_count,
        "candidate_tokens_p50": percentile(0.50),
        "candidate_tokens_p95": percentile(0.95),
        "tokenization_ms": tokenization_ms,
        "queue_wait_ms": timing.queue_wait_ms,
        "execution_ms": timing.execution_ms,
        "groups": output_groups,
    }

