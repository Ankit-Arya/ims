#!/usr/bin/env python3
"""Run a paced, heterogeneous IMS Q&A load test against the deployed HTTPS endpoint.

Example:
  python scripts/load_test_qa.py --base-url https://ims.dmrc.org \\
    --username loadtest --questions eval/load_questions.example.jsonl \\
    --requests 100 --duration 300

The script uses the same streaming endpoint as the browser and reports total latency and
resolved-mode latency. It never certifies capacity by itself; correlate the output with
Docker/host/PostgreSQL/inference metrics and ingestion progress.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[idx]


def load_questions(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        item = json.loads(raw)
        if item.get("question"):
            rows.append(item)
    if len(rows) < 5:
        raise SystemExit("Use at least five different representative questions; do not benchmark one repeated prompt.")
    return rows


async def one_request(client: httpx.AsyncClient, payload: dict[str, Any], slot: int, planned_at: float) -> dict[str, Any]:
    now = time.perf_counter()
    if planned_at > now:
        await asyncio.sleep(planned_at - now)
    started = time.perf_counter()
    resolved_mode = payload.get("mode", "auto")
    queue_step_seen = False
    try:
        async with client.stream("POST", "/api/v1/query/stream", json=payload, timeout=900) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") == "progress" and event.get("stage") == "queue":
                    queue_step_seen = True
                if event.get("type") == "result":
                    resolved_mode = event.get("data", {}).get("resolved_mode") or resolved_mode
                if event.get("type") == "error":
                    raise RuntimeError(event.get("message") or "stream error")
        return {"ok": True, "slot": slot, "seconds": time.perf_counter() - started, "mode": resolved_mode, "queue_step": queue_step_seen}
    except Exception as exc:  # benchmark should record, not abort, individual failures
        return {"ok": False, "slot": slot, "seconds": time.perf_counter() - started, "mode": resolved_mode, "error": str(exc), "queue_step": queue_step_seen}


async def run(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    verify: bool | str = args.ca_file if args.ca_file else (not args.insecure)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), verify=verify, limits=limits, timeout=900) as client:
        login = await client.post("/api/v1/auth/login", json={"username": args.username, "password": args.password})
        login.raise_for_status()
        overall_started = time.perf_counter()
        interval = args.duration / max(1, args.requests)
        tasks = []
        for index in range(args.requests):
            template = dict(questions[index % len(questions)])
            payload = {
                "question": template["question"],
                "mode": template.get("mode", "auto"),
                "document_ids": template.get("document_ids"),
            }
            tasks.append(asyncio.create_task(one_request(client, payload, index, overall_started + index * interval)))
        results = await asyncio.gather(*tasks)

    ok = [row for row in results if row["ok"]]
    failed = [row for row in results if not row["ok"]]
    by_mode: dict[str, list[float]] = defaultdict(list)
    for row in ok:
        by_mode[str(row["mode"])].append(float(row["seconds"]))

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "p50_s": round(percentile(values, .50), 3),
            "p95_s": round(percentile(values, .95), 3),
            "mean_s": round(statistics.mean(values), 3) if values else 0.0,
        }

    output = {
        "requests": len(results),
        "success": len(ok),
        "failed": len(failed),
        "scheduled_duration_s": args.duration,
        "wall_duration_s": round(time.perf_counter() - overall_started, 3),
        "overall": summary([float(row["seconds"]) for row in ok]),
        "by_resolved_mode": {mode: summary(values) for mode, values in sorted(by_mode.items())},
        "requests_that_reported_admission_wait": sum(1 for row in results if row.get("queue_step")),
        "sample_failures": [row.get("error") for row in failed[:10]],
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", help="Password; omit to enter it interactively without shell history")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--max-connections", type=int, default=60)
    parser.add_argument("--ca-file", help="Custom CA bundle for an internal PKI")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for a temporary lab test only")
    args = parser.parse_args()
    if not args.password:
        args.password = getpass.getpass("IMS password: ")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
