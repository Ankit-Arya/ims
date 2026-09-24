# IMS 0.13.4 - OKF + Jev local experiment

Base commit: `3e29df47f647c5d9c533bced44908615a23f9cdc`.

This patch is fail-safe by design:
- No database migration.
- OKF and Jev are disabled by default.
- Current BGE-M3/PostgreSQL/RRF behavior remains the fallback.
- `JEV_MODE=shadow` calls Jev and records judgment but does not reorder evidence.
- `JEV_MODE=apply` blends Jev judgment into `final_retrieval_score`.

## Environment

Add to your existing `.env`:

```env
OKF_ENABLED=true
OKF_PROFILE_ENRICHMENT_ENABLED=true
JEV_MODE=shadow
TYPESAFE_API_KEY=replace-with-your-key
JEV_BASE_URL=https://api.typesafe.ai
JEV_MODEL=jev-latest
JEV_TIMEOUT_SECONDS=6
JEV_MAX_CANDIDATES=16
JEV_CANDIDATE_CHARS=2600
JEV_WEIGHT=0.55
JEV_FAIL_OPEN=true
```
## Existing documents

Because your local database already mirrors production, you do not need to re-upload or re-embed existing ready documents. After rebuilding, run:

```bash
docker compose exec api-1 python scripts/backfill_okf.py
```

This reads the existing `documents` and `chunks`, generates OKF sidecars, and enriches missing operational profiles without changing embeddings or chunk text. It will create:

```
/data/okf/index.md
/data/okf/documents/<document-id>.md
```

and record status/path in `documents.extra_metadata.okf`.

When OKF is enabled and a document does not already have an `operational_profile`, ingestion also derives the existing IMS operational profile (line, rolling stock, document type) from title/filename/chunks. Current IMS applicability scoring already consumes that profile. The authoritative PDF/chunks remain the factual answer source.

## Build

```bash
python -m pytest -q tests/test_0134_okf_jev.py
docker compose build
docker compose up -d
```
## Controlled A/B sequence

1. Baseline: `OKF_ENABLED=false`, `JEV_MODE=off`.
2. OKF only: `OKF_ENABLED=true`, `JEV_MODE=off`; re-ingest selected PDFs.
3. Jev shadow: `OKF_ENABLED=true`, `JEV_MODE=shadow`.
4. Jev apply: `OKF_ENABLED=true`, `JEV_MODE=apply`.

Use exactly the same question set in every run.

Compare:
- correct document and section
- correct line/rolling-stock/role applicability
- citation correctness
- answer completeness
- unsupported claims
- p50/p95 latency
- Jev input-token usage/cost

Jev details are attached under the existing retrieval `rerank_details` trace. In shadow mode the candidate order is unchanged.

## Rollback

Set:

```env
OKF_ENABLED=false
JEV_MODE=off
```

and restart API/worker containers. No schema rollback is required. Existing `/data/okf` files and JSON metadata are inert when disabled.
