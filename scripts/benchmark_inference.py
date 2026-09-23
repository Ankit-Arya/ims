#!/usr/bin/env python3
"""Benchmark an IMS inference-query endpoint without changing retrieval quality.

Use the same query/candidate payload against torch, ONNX or OpenVINO deployments and
compare p50/p95, queue wait, execution time and ranking output before changing the
production backend.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[index]


def load_case(path: str | None, candidate_count: int) -> tuple[str, list[str]]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        query = str(data["query"])
        candidates = [str(item) for item in data["candidates"]]
        return query, candidates[:candidate_count]
    query = "station controller responsibilities during an emergency"
    templates = [
        "The Station Controller shall handle emergent situations and public announcements.",
        "Maintenance staff shall report before starting work at the station.",
        "The Train Operator shall secure the train and contact the Traffic Controller.",
        "The Station Controller shall supervise passenger flow and assist during evacuation.",
        "General administrative information unrelated to station operation.",
    ]
    candidates = [templates[i % len(templates)] + f" Candidate {i + 1}." for i in range(candidate_count)]
    return query, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--token", default=os.getenv("INTERNAL_SERVICE_TOKEN"), help="INTERNAL_SERVICE_TOKEN (defaults to environment)")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--case-json", help="JSON with query and candidates fields")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token is required unless INTERNAL_SERVICE_TOKEN is set")

    query, candidates = load_case(args.case_json, args.candidates)
    headers = {"X-Internal-Token": args.token}
    latencies: list[float] = []
    queue_waits: list[float] = []
    executions: list[float] = []
    last_order: list[int] = []
    backend = "unknown"

    with httpx.Client(timeout=600) as client:
        health = client.get(f"{args.url.rstrip('/')}/health").json()
        print("health:", json.dumps(health, indent=2))
        for run in range(args.warmup + args.runs):
            started = time.perf_counter()
            response = client.post(
                f"{args.url.rstrip('/')}/rerank",
                headers=headers,
                json={"query": query, "candidates": candidates, "top_k": len(candidates)},
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - started) * 1000
            payload = response.json()
            backend = str(payload.get("backend") or backend)
            if run >= args.warmup:
                latencies.append(elapsed_ms)
                queue_waits.append(float(payload.get("queue_wait_ms") or 0))
                executions.append(float(payload.get("execution_ms") or 0))
            last_order = [int(item["index"]) for item in payload.get("results", [])]

    print(json.dumps({
        "backend": backend,
        "candidate_count": len(candidates),
        "runs": len(latencies),
        "wall_ms": {"p50": round(percentile(latencies, .50), 2), "p95": round(percentile(latencies, .95), 2), "mean": round(statistics.mean(latencies), 2)},
        "queue_wait_ms": {"p50": round(percentile(queue_waits, .50), 2), "p95": round(percentile(queue_waits, .95), 2)},
        "execution_ms": {"p50": round(percentile(executions, .50), 2), "p95": round(percentile(executions, .95), 2)},
        "last_rank_order": last_order,
    }, indent=2))


if __name__ == "__main__":
    main()
