#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import httpx


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
    return cases


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a golden-set retrieval/answer evaluation")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--username", default=os.environ.get("BOOTSTRAP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("BOOTSTRAP_ADMIN_PASSWORD"))
    args = parser.parse_args()
    if not args.password:
        raise SystemExit("Provide --password or BOOTSTRAP_ADMIN_PASSWORD")

    cases = load_cases(args.cases)
    rows = []
    with httpx.Client(base_url=args.base_url, timeout=300.0) as client:
        r = client.post("/api/v1/auth/login", json={"username": args.username, "password": args.password})
        r.raise_for_status()
        for case in cases:
            question = case["question"]
            debug = client.get("/api/v1/debug/retrieval", params={"q": question})
            debug.raise_for_status()
            retrieved = debug.json()["evidence"]
            answer_response = client.post("/api/v1/query", json={"question": question, "mode": case.get("mode", "auto")})
            answer_response.raise_for_status()
            answer = answer_response.json()

            expected_docs = [normalized(x) for x in case.get("expected_documents", [])]
            expected_pages = {int(x) for x in case.get("expected_pages", [])}
            required_terms = [normalized(x) for x in case.get("required_terms", case.get("must_include", []))]

            retrieved_docs = [normalized(x["document"]) for x in retrieved]
            cited_docs = [normalized(x["document_title"]) for x in answer.get("citations", [])]
            retrieved_pages = {x["page_from"] for x in retrieved if x.get("page_from") is not None}
            answer_norm = normalized(answer["answer"])

            retrieval_doc_hit = not expected_docs or all(any(exp in got for got in retrieved_docs) for exp in expected_docs)
            citation_doc_hit = not expected_docs or all(any(exp in got for got in cited_docs) for exp in expected_docs)
            page_hit = not expected_pages or bool(expected_pages & retrieved_pages)
            term_hit = not required_terms or all(term in answer_norm for term in required_terms)
            passed = retrieval_doc_hit and citation_doc_hit and page_hit and term_hit
            row = {
                "id": case.get("id", question[:40]),
                "passed": passed,
                "retrieval_doc_hit": retrieval_doc_hit,
                "citation_doc_hit": citation_doc_hit,
                "page_hit": page_hit,
                "required_terms_hit": term_hit,
                "confidence": answer.get("confidence"),
                "requested_mode": answer.get("mode"),
                "resolved_mode": answer.get("resolved_mode"),
                "latency_ms": answer.get("latency_ms"),
                "query_id": answer.get("query_id"),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    passed = sum(1 for row in rows if row["passed"])
    summary = {"cases": len(rows), "passed": passed, "pass_rate": passed / len(rows) if rows else 0.0}
    print("SUMMARY", json.dumps(summary))
    raise SystemExit(0 if passed == len(rows) else 2)


if __name__ == "__main__":
    main()
