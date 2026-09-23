#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the running IKE stack")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--username", default=os.environ.get("BOOTSTRAP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("BOOTSTRAP_ADMIN_PASSWORD"))
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--question", default="What is this document about?")
    args = parser.parse_args()
    if not args.password:
        raise SystemExit("Provide --password or BOOTSTRAP_ADMIN_PASSWORD")

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        print("ready:", ready.json())
        login = client.post("/api/v1/auth/login", json={"username": args.username, "password": args.password})
        login.raise_for_status()
        print("login: ok")

        if args.pdf:
            with args.pdf.open("rb") as handle:
                upload = client.post(
                    "/api/v1/documents",
                    files={"file": (args.pdf.name, handle, "application/pdf")},
                    data={"title": args.pdf.stem},
                )
            upload.raise_for_status()
            document_id = upload.json()["id"]
            print("uploaded:", document_id)
            for _ in range(180):
                info = client.get(f"/api/v1/documents/{document_id}")
                info.raise_for_status()
                status = info.json()["ingestion_status"]
                print("ingestion:", status)
                if status == "ready":
                    break
                if status == "failed":
                    raise SystemExit(info.json().get("ingestion_error"))
                time.sleep(2)
            else:
                raise SystemExit("Ingestion did not reach a terminal state during the smoke-test polling window")

        answer = client.post("/api/v1/query", json={"question": args.question, "mode": "auto"})
        answer.raise_for_status()
        payload = answer.json()
        print("answer:\n", payload["answer"])
        print("citations:", len(payload["citations"]), "confidence:", payload["confidence"])


if __name__ == "__main__":
    main()
