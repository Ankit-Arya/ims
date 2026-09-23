#!/usr/bin/env python3
"""Capture a secret-free acceptance manifest from a running Docker Compose deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "__pycache__",
    "data",
    "acceptance",
}
EXCLUDED_NAMES = {".env"}


def run(*args: str) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def source_manifest(root: Path) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if path.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = relative.as_posix()
        rows.append({"path": rel, "sha256": digest})
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture acceptance-build provenance without copying .env secrets")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--inference-url", default="http://localhost:8090/health")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or root / "acceptance" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_sha256, files = source_manifest(root)
    manifest: dict[str, object] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_tree_sha256": source_sha256,
        "source_file_count": len(files),
        "docker_version": run("docker", "version", "--format", "{{.Server.Version}}"),
        "compose_version": run("docker", "compose", "version", "--short"),
        "compose_images": run("docker", "compose", "images"),
    }

    with urlopen(args.inference_url, timeout=5) as response:  # nosec B310 - operator-supplied local health URL
        manifest["inference_health"] = json.loads(response.read().decode("utf-8"))

    manifest["pgvector_version"] = run(
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "sh",
        "-lc",
        "psql -At -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -c \"SELECT extversion FROM pg_extension WHERE extname='vector'\"",
    )

    for service in ("api", "worker", "inference"):
        freeze = run("docker", "compose", "exec", "-T", service, "python", "-m", "pip", "freeze")
        (output / f"pip-freeze-{service}.txt").write_text(freeze + "\n", encoding="utf-8")

    (output / "source-files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Acceptance manifest written to: {output}")
    print(f"Source tree SHA-256: {source_sha256}")
    print("No .env values are captured by this script.")


if __name__ == "__main__":
    main()
