# Upgrade from 0.1.1 to 0.2.0

0.2.0 does **not** require a database migration. Existing users, documents, chunks, query logs and model-cache volumes can be preserved.

## What changes

- Q&A API accepts `direct` (legacy `qa` remains accepted)
- new streaming endpoint `/api/v1/query/stream`
- new history endpoint `/api/v1/query/history`
- Direct Q&A uses the fast model and smaller retrieval pool
- Research expands first and performs broader retrieval + verification
- acronym/definition lookups receive a document-diverse protected retrieval path
- OCR defaults to `eng` + `pdf_aware_layout_regions`
- Docling pipeline is cached per worker process
- report work moves to a dedicated `reports` Celery queue/service
- ingestion worker listens to the dedicated `ingestion` queue
- browser UI is replaced with the 0.2 workflow/history/progress design

## Safe upgrade procedure

From the existing project directory after replacing source files with 0.2.0:

```powershell
docker compose build api worker report-worker migrate

docker compose run --rm migrate

docker compose up -d --force-recreate api worker report-worker
```

The inference image is unchanged by this release. Rebuild it only if you deliberately change inference code/model dependencies.

Check:

```powershell
docker compose ps
docker compose logs --tail=120 api worker report-worker
curl.exe http://localhost:8080/health/ready
```

Do **not** run:

```powershell
docker compose down -v
```

unless you intentionally want to delete PostgreSQL data, source documents, Valkey state and model cache.

## Existing `.env`

The release has safe code/default fallbacks for the new variables, so an existing `.env` can boot unchanged. For explicit configuration, add:

```text
DIRECT_DENSE_TOP_K=30
DIRECT_LEXICAL_TOP_K=40
DIRECT_EXACT_TOP_K=30
DIRECT_FUSED_TOP_K=36
DIRECT_RERANK_TOP_K=12
DIRECT_EVIDENCE_K=8
LOOKUP_SCAN_TOP_K=240
LOOKUP_RERANK_TOP_K=16
LOOKUP_EVIDENCE_K=16

DOCLING_OCR_LANGUAGES=eng
DOCLING_OCR_MODE=pdf_aware_layout_regions
DOCLING_TABLE_MODE=accurate
INGESTION_EMBEDDING_BATCH_SIZE=32

INGEST_WORKER_CONCURRENCY=1
INGEST_MAX_TASKS_PER_CHILD=20
REPORT_WORKER_CONCURRENCY=1
REPORT_MAX_TASKS_PER_CHILD=10
```

## Do existing documents need reindexing?

No for the Q&A/UI changes. Existing chunks remain usable.

Reindex a document only if you want it to benefit from the updated Docling/OCR configuration. Reindexing changes extracted/chunked evidence, so evaluate representative documents before bulk reindexing the whole corpus.

## Queue compatibility

0.2.0 routes new ingestion tasks to `ingestion` and report tasks to `reports`. Start both `worker` and `report-worker` after the upgrade.

If an old queued task existed in the unnamed/default Celery queue during upgrade, it will not be consumed by the new queue-specific workers. For a pilot, inspect the affected document/report status and requeue it explicitly via the API after confirming it is not running.
