#!/usr/bin/env python3
"""Upgrade an existing IMS .env to the 0.6.0 production profile without touching secrets.

The script is intentionally line-oriented instead of shell-sourcing the file: IMS values such
as APP_NAME contain spaces and a .env file is not necessarily valid shell syntax.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

RELEASE = "0.6.0"

# These values are safe, non-secret release controls for the current 8-vCPU/~31-GiB VM.
# Existing secrets, credentials, URLs containing credentials, and unknown custom settings are
# deliberately preserved exactly as they appear in the source .env.
MANAGED: dict[str, str] = {
    "APP_ENV": "production",
    "APP_NAME": "IMS - Incident Management System",
    "ORGANIZATION_NAME": "Delhi Metro Rail Corporation (DMRC)",
    "ORGANIZATION_SHORT_NAME": "DMRC",
    "SESSION_COOKIE_SECURE": "true",
    "LOG_LEVEL": "INFO",
    "TLS_CERT_DIR": "/etc/ims/tls",
    "QUERY_INFERENCE_URL": "http://inference-query:8090",
    "INGESTION_INFERENCE_URL": "http://inference-ingest:8090",
    "INFERENCE_URL": "http://inference-query:8090",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "EMBEDDING_DIM": "1024",
    "EMBEDDING_MAX_SEQ_LENGTH": "1024",
    "RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANK_MAX_LENGTH": "1024",
    "ML_DEVICE": "cpu",
    "ML_BATCH_SIZE": "8",
    "RERANK_BACKEND": "torch",
    "INSTALL_OPTIMIZED_RERANK": "0",
    "QUERY_ML_NUM_THREADS": "6",
    "QUERY_EMBED_CONCURRENCY": "1",
    "QUERY_RERANK_CONCURRENCY": "1",
    "INGEST_ML_NUM_THREADS": "2",
    "INGEST_EMBED_CONCURRENCY": "1",
    "DENSE_TOP_K": "80",
    "LEXICAL_TOP_K": "80",
    "EXACT_TOP_K": "30",
    "FUSED_TOP_K": "80",
    "RERANK_TOP_K": "16",
    "ANSWER_EVIDENCE_K": "12",
    "RRF_K": "60",
    "HNSW_EF_SEARCH": "100",
    "HNSW_MAX_SCAN_TUPLES": "20000",
    "RERANK_SCORE_THRESHOLD": "0.12",
    "NEIGHBOR_RADIUS": "1",
    "DIRECT_DENSE_TOP_K": "30",
    "DIRECT_LEXICAL_TOP_K": "40",
    "DIRECT_EXACT_TOP_K": "30",
    "DIRECT_FUSED_TOP_K": "36",
    "DIRECT_RERANK_TOP_K": "12",
    "DIRECT_EVIDENCE_K": "10",
    "LOOKUP_SCAN_TOP_K": "240",
    "LOOKUP_RERANK_TOP_K": "16",
    "LOOKUP_EVIDENCE_K": "16",
    "COVERAGE_SCAN_TOP_K": "800",
    "COVERAGE_MAX_DOCUMENTS": "16",
    "COVERAGE_EVIDENCE_PER_DOCUMENT": "2",
    "COVERAGE_MAX_EVIDENCE_K": "32",
    "ROLE_ALIAS_SCAN_TOP_K": "120",
    "ROLE_ALIAS_FALLBACK_SCAN_TOP_K": "240",
    "ROLE_COVERAGE_MAX_ALIASES": "4",
    "ROLE_COVERAGE_SCAN_TOP_K": "240",
    "ROLE_COVERAGE_MAX_DOCUMENTS": "24",
    "ROLE_COVERAGE_MAX_SECTIONS": "32",
    "ROLE_COVERAGE_SECTIONS_PER_DOCUMENT": "8",
    "ROLE_COVERAGE_MAX_EVIDENCE_K": "32",
    "ROLE_COVERAGE_RERANK_POOL": "40",
    "ROLE_COVERAGE_RERANK_TOP_K": "20",
    "ROLE_ALIAS_CACHE_TTL_SECONDS": "600",
    "ROLE_ALIAS_CACHE_MAX_ITEMS": "512",
    "QUERY_MAX_ACTIVE_PER_API": "4",
    "QUERY_MAX_WAITING_PER_API": "48",
    # Rich answers remain evidence-bound. Six related sections is a ceiling, not a quota.
    "HELPFUL_CONTEXT_MODE": "rich",
    "HELPFUL_CONTEXT_MAX_SECTIONS": "6",
    "DIRECT_MAX_OUTPUT_TOKENS": "3200",
    "OVERVIEW_RETRIEVAL_ENABLED": "true",
    "OVERVIEW_DOCUMENT_PATTERN": "MRGR",
    "OVERVIEW_DENSE_TOP_K": "12",
    "OVERVIEW_LEXICAL_TOP_K": "12",
    "OVERVIEW_EVIDENCE_K": "4",
    "IDENTIFIER_VARIANT_EXPANSION": "true",
    "IDENTIFIER_VARIANT_MAX_QUERIES": "8",
    "TABLE_CONTEXT_BOOST_ENABLED": "true",
    "TABLE_CONTEXT_FUSION_BONUS": "0.12",
    "TABLE_RETRIEVAL_TOP_K": "16",
    "TABLE_RETRIEVAL_WEIGHT": "1.35",
    "HIERARCHICAL_RETRIEVAL_ENABLED": "true",
    "HIERARCHICAL_WINDOW": "4",
    "HIERARCHICAL_MAX_CHUNKS_PER_ANCHOR": "6",
    "RERANK_PREFILTER_MAX_CANDIDATES": "40",
    "VERIFICATION_RISK_THRESHOLD": "3",
    "VISUAL_EVIDENCE_ENABLED": "true",
    "VISUAL_MAX_ITEMS": "3",
    "VISUAL_NEARBY_PAGES": "1",
    "VISUAL_RENDER_DPI": "130",
    "VISUAL_CACHE_DIR_NAME": "visual-cache",
    "VISUAL_TOKEN_TTL_SECONDS": "3600",
    "SELECTIVE_VISION_ENABLED": "false",
    "CHUNK_MAX_TOKENS": "500",
    "DOCLING_OCR": "true",
    "DOCLING_OCR_LANGUAGES": "eng",
    "DOCLING_OCR_MODE": "pdf_aware_layout_regions",
    "DOCLING_TABLE_STRUCTURE": "true",
    "DOCLING_TABLE_MODE": "accurate",
    "DOCLING_TIMEOUT_SECONDS": "3600",
    "DOCLING_ARTIFACTS_PATH": "/models/docling",
    "INGESTION_EMBEDDING_BATCH_SIZE": "32",
    "INGEST_WORKER_CONCURRENCY": "1",
    "INGEST_CPU_TARGET_PERCENT": "75",
    "INGEST_MAX_CONCURRENCY": "1",
    "INGEST_MEMORY_GB_PER_PROCESS": "2.5",
    "INGEST_MEMORY_RESERVE_GB": "3.0",
    "DOCLING_NUM_THREADS_PER_DOCUMENT": "2",
    "DOCLING_OCR_BATCH_SIZE": "2",
    "DOCLING_LAYOUT_BATCH_SIZE": "2",
    "DOCLING_TABLE_BATCH_SIZE": "2",
    "TESSERACT_OMP_THREAD_LIMIT": "1",
    "INGEST_MAX_TASKS_PER_CHILD": "20",
    "REPORT_WORKER_CONCURRENCY": "1",
    "REPORT_MAX_TASKS_PER_CHILD": "10",
    "LLM_MAX_OUTPUT_TOKENS": "7000",
    "VERIFY_MODE": "selective",
    "REPORT_MAX_DOCUMENTS": "20",
    "REPORT_PACK_MAX_CHARS": "24000",
    "REPORT_MAX_PACKS": "120",
    "REPORT_PARALLELISM": "3",
    "REPORT_REDUCE_BATCH_CHARS": "60000",
    "REPORT_SYNTHESIS_MAX_CHARS": "120000",
    "REPORT_REDUCE_MAX_ROUNDS": "4",
    "ONLINE_CPUSET": "0-5",
    "BACKGROUND_CPUSET": "6-7",
    "POSTGRES_MEMORY_LIMIT": "4g",
    "VALKEY_MEMORY_LIMIT": "512m",
    "QUERY_INFERENCE_MEMORY_LIMIT": "5g",
    "INGEST_INFERENCE_MEMORY_LIMIT": "3500m",
    "API_MEMORY_LIMIT": "1500m",
    "WORKER_MEMORY_LIMIT": "8g",
    "REPORT_WORKER_MEMORY_LIMIT": "2g",
    "NGINX_MEMORY_LIMIT": "128m",
    "MODEL_BOOTSTRAP_MEMORY_LIMIT": "4g",
    "MIGRATE_MEMORY_LIMIT": "1g",
}

NEVER_TOUCH = {
    "APP_SECRET_KEY",
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "INTERNAL_SERVICE_TOKEN",
    "OPENAI_API_KEY",
    "BOOTSTRAP_ADMIN_PASSWORD",
}


def key_of(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        return None
    key = line.split("=", 1)[0].strip()
    if key and all(ch.isalnum() or ch == "_" for ch in key):
        return key
    return None


def upgrade(path: Path, *, dry_run: bool = False) -> tuple[Path | None, list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Environment file not found: {path}")

    original = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    changed: list[str] = []
    output: list[str] = []

    for line in original:
        key = key_of(line)
        if key:
            seen.add(key)
        if key in NEVER_TOUCH:
            output.append(line)
            continue
        if key in MANAGED:
            replacement = f"{key}={MANAGED[key]}"
            if line != replacement:
                changed.append(key)
            output.append(replacement)
        else:
            output.append(line)

    missing = [key for key in MANAGED if key not in seen]
    if missing:
        if output and output[-1].strip():
            output.append("")
        output.extend(
            [
                "# ============================================================",
                f"# IMS {RELEASE} managed upgrade values",
                "# Added by scripts/upgrade_env_060.py; secrets above were preserved.",
                "# ============================================================",
            ]
        )
        output.extend(f"{key}={MANAGED[key]}" for key in missing)

    backup: Path | None = None
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.name}.before-0.6.0.{timestamp}")
        shutil.copy2(path, backup)
        path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
        path.chmod(0o600)

    return backup, sorted(set(changed)), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".env", help="Existing .env path")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    backup, changed, added = upgrade(path, dry_run=args.dry_run)
    print(f"IMS {RELEASE} environment upgrade {'preview' if args.dry_run else 'complete'}: {path}")
    if backup:
        print(f"Backup: {backup}")
    print(f"Managed values changed: {len(changed)}")
    print(f"Managed values added: {len(added)}")
    print("Secrets preserved: " + ", ".join(sorted(NEVER_TOUCH)))
    print("Run: docker compose config -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
