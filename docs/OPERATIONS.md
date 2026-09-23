# Operations — 0.5.0

Primary services: `nginx`, `api-1`, `api-2`, `postgres`, `valkey`, `inference-query`, `inference-ingest`, `worker`, `report-worker`. Useful loopback checks: API replicas on 8081/8082, query inference on 8090, ingestion inference on 8091. Public access belongs on HTTPS 443 through Nginx.

For ingestion maintenance, keep `INGEST_WORKER_CONCURRENCY=1` on the current VM. To drain before a deployment, cancel the `ingestion` consumer and wait for the current `processing` document to finish; queued tasks stay in Valkey. Never use `docker compose down -v`.

For performance diagnosis, use API `/metrics`, query-inference `/metrics`, ingestion-inference `/metrics`, retrieval traces, `docker stats`, host memory/CPU and PostgreSQL `EXPLAIN (ANALYZE, BUFFERS)`. Queue wait and model execution are separate metrics.

---

## Historical / earlier-release notes


## Service checks

```powershell
docker compose ps
curl.exe http://localhost:8080/health/live
curl.exe http://localhost:8080/health/ready
```

Useful logs:

```powershell
docker compose logs --tail=100 api
docker compose logs --tail=150 worker
docker compose logs --tail=100 report-worker
docker compose logs --tail=100 inference
```

## Verify ingestion concurrency

```powershell
docker compose logs worker | Select-String "IKE ingestion worker"
```

If the machine becomes memory constrained, reduce one or more of:

```text
INGEST_CPU_TARGET_PERCENT
INGEST_MAX_CONCURRENCY
INGEST_WORKER_CONCURRENCY (set explicit lower integer)
```

or increase the memory estimate/reserve so auto mode chooses fewer processes.

Restart only the ingestion worker after changing those settings:

```powershell
docker compose up -d --force-recreate worker
```

## Processing many PDFs

The browser supports multi-file upload. Admin/analyst bulk organisation ingestion can also use:

```powershell
python scripts/bulk_upload.py "D:\PDFs" --password "<password>" --parallel 4
```

`--parallel` controls upload HTTP concurrency; the Celery ingestion worker independently controls actual PDF-processing concurrency.

Monitor:

```powershell
docker stats
docker compose logs -f worker
```

For 1,000-document onboarding, ingest a representative batch first. Record throughput, peak RAM, CPU utilization, extraction failures and average document size before deciding final concurrency.

## Failed documents

A terminal failed document can be retried from its library row. Re-uploading identical bytes also retries an existing failed record within the relevant deduplication scope.

A PDF cannot be deleted while queued/processing.

## Personal PDFs

Personal documents are stored using the same source/parsed volume conventions as organisation PDFs, but access is controlled by workspace ownership and `document_shares`.

Deleting a personal PDF permanently removes its DB document/chunks/shares plus source/parsed files.

## Deep Analysis

Deep Analysis jobs are on the `reports` queue. If normal Q&A works but Deep Analysis stalls:

```powershell
docker compose ps report-worker
docker compose logs --tail=150 report-worker
```

The browser stores no report result locally; job progress/result is persisted in PostgreSQL.

## Backups

Back up both:

- PostgreSQL volume/database;
- shared source/parsed document storage.

The Valkey queue is not the system of record.

## Acceptance snapshot

After you approve a concrete build on the target machine:

```powershell
python scripts/capture_acceptance.py
```

Store that manifest with your release/acceptance records so later dependency/image changes are detectable.