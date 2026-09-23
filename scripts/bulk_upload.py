#!/usr/bin/env python3
"""Queue a directory tree of PDFs through the normal authenticated upload API."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def upload_one(
    base_url: str,
    token: str,
    path: Path,
    *,
    access_scope: str,
    department: str | None,
    allow_duplicate: bool,
    timeout: float,
) -> tuple[Path, str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "access_scope": access_scope,
        "allow_duplicate": "true" if allow_duplicate else "false",
    }
    if department:
        data["department"] = department
    try:
        with path.open("rb") as handle, httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.post(
                "/api/v1/documents",
                headers=headers,
                files={"file": (path.name, handle, "application/pdf")},
                data=data,
            )
        if response.status_code == 409 and not allow_duplicate:
            return path, "duplicate", response.json().get("detail", "already exists")
        response.raise_for_status()
        return path, "queued", response.json()["id"]
    except Exception as exc:  # noqa: BLE001 - CLI should continue and report every file
        return path, "failed", str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue all PDFs in a folder tree for IKE ingestion")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--username", default=os.environ.get("BOOTSTRAP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("BOOTSTRAP_ADMIN_PASSWORD"))
    parser.add_argument("--parallel", type=int, default=4, help="simultaneous HTTP uploads; not ingestion concurrency")
    parser.add_argument("--access-scope", choices=["organization", "department", "restricted"], default="organization")
    parser.add_argument("--department")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("Provide --password or BOOTSTRAP_ADMIN_PASSWORD")
    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")
    if args.parallel < 1 or args.parallel > 16:
        raise SystemExit("--parallel must be between 1 and 16")

    pattern = "*.pdf" if args.no_recursive else "**/*.pdf"
    files = sorted(path for path in args.folder.glob(pattern) if path.is_file())
    if not files:
        raise SystemExit("No PDFs found")

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        login = client.post("/api/v1/auth/login", json={"username": args.username, "password": args.password})
        login.raise_for_status()
        token = login.json()["access_token"]

    print(f"Found {len(files)} PDF(s); queuing with {args.parallel} parallel HTTP upload(s).")
    counts = {"queued": 0, "duplicate": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                upload_one,
                args.base_url,
                token,
                path,
                access_scope=args.access_scope,
                department=args.department,
                allow_duplicate=args.allow_duplicate,
                timeout=args.timeout,
            ): path
            for path in files
        }
        for index, future in enumerate(as_completed(futures), start=1):
            path, status, detail = future.result()
            counts[status] += 1
            print(f"[{index}/{len(files)}] {status.upper():9s} {path} {detail}")

    print("SUMMARY", counts)
    raise SystemExit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
