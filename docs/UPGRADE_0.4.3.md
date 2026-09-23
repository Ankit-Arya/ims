# Upgrade from 0.4.2 to 0.4.3

Release 0.4.3 fixes a silent ingestion-integrity defect. Docling can return `PARTIAL_SUCCESS` when a document reaches its processing timeout or when pages fail. Earlier IMS releases consumed that partial `document`, embedded the available chunks, and marked the database record `ready`.

## What changes

- IMS now accepts only Docling `SUCCESS` conversions for indexing.
- Timeout/partial conversions become failed ingestion jobs instead of incomplete `ready` documents.
- Existing chunks are deleted only after a complete parse and complete embedding pass succeeds, so a failed reprocess does not destroy the previous index.
- Fresh installs default `DOCLING_TIMEOUT_SECONDS` to 3600 seconds.
- `DOCLING_TIMEOUT_SECONDS=0` disables Docling's internal document timeout if an administrator intentionally wants no per-document limit.

## Existing `.env`

Existing `.env` files are preserved and may still contain `DOCLING_TIMEOUT_SECONDS=900`. For large DMRC manuals on CPU infrastructure, change it to:

```text
DOCLING_TIMEOUT_SECONDS=3600
```

For recovery/reindexing, also consider temporarily limiting parallel ingestion, for example:

```text
INGEST_WORKER_CONCURRENCY=2
```

Increase concurrency again only after representative large manuals complete reliably within the configured timeout and memory envelope.

## Apply

No database migration is required. Rebuild/recreate the ingestion worker:

```bash
docker compose build --no-cache worker
docker compose up -d --force-recreate worker
```

The API does not need rebuilding unless you deploy the full 0.4.3 source release for version consistency.

## Reprocess suspect documents

Any document indexed under an earlier release that is `ready` but has materially fewer chunks/pages than a known-good index should be reprocessed. Use the Documents UI **Reprocess** action. The source PDF is retained and successful reprocessing replaces the old chunks atomically at the final database commit stage.

After reprocessing, compare page/chunk coverage again before evaluating Q&A quality.
