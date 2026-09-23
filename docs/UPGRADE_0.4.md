# Upgrade from 0.3.0 to 0.4.0

Release 0.4.0 adds a real schema migration. Existing organisation documents become `workspace_scope=organization`; no existing source PDFs/chunks are deleted.

## Before upgrading

1. Let currently running/queued ingestion jobs finish if practical.
2. Keep the existing `.env`.
3. Do not delete Docker volumes.
4. Back up PostgreSQL/source data before an organisational deployment.

## Apply files

Extract the 0.4 upgrade ZIP into the existing project directory and overwrite matching project files. The upgrade package does not include `.env`.

## Optional environment tuning

Existing `.env` files continue to work because defaults are defined in code/Compose. To make the new tuning explicit, add:

```text
INGEST_WORKER_CONCURRENCY=auto
INGEST_CPU_TARGET_PERCENT=75
INGEST_MAX_CONCURRENCY=8
INGEST_MEMORY_GB_PER_PROCESS=2.5
INGEST_MEMORY_RESERVE_GB=3.0
DOCLING_NUM_THREADS_PER_DOCUMENT=2
DOCLING_OCR_BATCH_SIZE=2
DOCLING_LAYOUT_BATCH_SIZE=2
DOCLING_TABLE_BATCH_SIZE=2
TESSERACT_OMP_THREAD_LIMIT=1
REPORT_MAX_DOCUMENTS=20
```

## Build and migrate

From the project folder:

```powershell
docker compose build api worker report-worker migrate model-bootstrap
docker compose run --rm migrate
docker compose up -d --force-recreate api worker report-worker
```

You normally do not need to rerun `model-bootstrap` if `/models/docling` is already populated from 0.3. If the worker reports missing Docling artifacts, run:

```powershell
docker compose run --rm model-bootstrap
docker compose up -d --force-recreate worker
```

## Verify

```powershell
docker compose ps
curl.exe http://localhost:8080/health/ready
docker compose logs --tail=100 worker
```

Look for a worker startup line like:

```text
IKE ingestion worker: concurrency=..., CPUs=..., target=75%, memory=...GiB, Docling threads/document=2
```

Then test:

1. Ask page at phone-width and desktop-width.
2. Light/dark theme toggle.
3. Direct, Research and Auto Q&A.
4. Deep Analysis from the same Ask screen.
5. Personal PDF upload by a normal `user`.
6. Share that PDF with User B.
7. Confirm User B can view/download/query it.
8. Confirm User C and an unshared admin cannot access it via normal routes.
9. Confirm My Questions remains private per account.
10. Upload several representative PDFs and observe parallel ingestion CPU/RAM before raising concurrency limits.

## Rollback warning

The migration adds `workspace_scope`, `document_shares` and report progress fields. Rolling application code back to 0.3 while leaving schema at 0.4 is not an endorsed production rollback plan. Restore a pre-upgrade DB backup if a full rollback is required.
