# Upgrade from 0.2.0 to 0.3.0

Release 0.3.0 requires **no database schema migration**. Existing PostgreSQL data, uploaded source PDFs, parsed JSON, chunks, model cache, query history and reports can remain in place.

## What changes

- new `model-bootstrap` one-shot service for persistent Docling artifacts;
- ingestion worker uses local Docling `artifacts_path`;
- transient internal-inference connection failures are retried;
- Celery automatic retry status remains `queued` until terminal failure;
- identical failed uploads retry the existing document record;
- document `Retry`, `Reprocess`, `View PDF`, `Download`, and permanent `Delete` actions;
- separate **Ask** and **My Questions** screens;
- private per-user Q&A/report product surfaces;
- administrator Users page.

## Before upgrading

Allow any currently `processing` 0.2 ingestion jobs to finish if practical.

Do not delete volumes.

Optional backup:

```powershell
docker compose exec postgres sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f /tmp/ike-before-030.dump'
docker cp institutional-knowledge-engine-postgres-1:/tmp/ike-before-030.dump .\ike-before-030.dump
```

## Apply the upgrade package

Extract the 0.3 upgrade ZIP into the existing project folder and overwrite matching source/config/docs files. The upgrade package does not contain `.env`.

Then:

```powershell
docker compose build api worker report-worker migrate model-bootstrap
```

Initialize local Docling artifacts:

```powershell
docker compose run --rm model-bootstrap
```

Then recreate application services:

```powershell
docker compose run --rm migrate
docker compose up -d --force-recreate api worker report-worker inference
```

`migrate` is safe even though 0.3 has no new migration; it confirms the existing schema is current.

## Confirm model bootstrap

```powershell
docker compose logs --tail=100 model-bootstrap
```

Expected result includes `docling_model_cache_bootstrap_complete` or `docling_model_cache_ready`.

If the first download encounters DNS/internet failure, restore connectivity and rerun:

```powershell
docker compose run --rm model-bootstrap
```

The bootstrap sentinel prevents repeated external downloads on normal future starts.

## Confirm services

```powershell
docker compose ps
```

Expected:

- postgres healthy
- valkey healthy
- inference healthy
- model-bootstrap exited 0
- api healthy
- worker up
- report-worker up

## Retry an existing failed document

Open **Documents** and press **Retry processing** on the failed PDF.

You may also re-upload the exact same failed PDF. SHA-256 deduplication will reuse/retry the existing failed record instead of creating a duplicate.

Renaming the local file does not change its content hash.

## Tester accounts

Open **Users** as an administrator and create a separate account for each tester.

For ordinary acceptance testing choose role `user`. These users can:

- see documents permitted by document ACL;
- view/download permitted PDFs;
- ask questions;
- see only their own question history;
- create/see only their own reports.

Use `analyst` only when a tester should upload and manage their own documents.

## Destructive command warning

Do not run:

```powershell
docker compose down -v
```

unless the intention is to destroy PostgreSQL, model cache and source/parsed document volumes.
