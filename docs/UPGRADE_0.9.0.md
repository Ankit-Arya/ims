# Upgrade IMS 0.8.0 -> 0.9.0

## Compatibility

0.9.0 is source-compatible with the existing chunk corpus but adds one database table and one inference-query endpoint.

Required:
- apply Alembic revision `0004_retrieval_intelligence`;
- rebuild/recreate `inference-query`, `api-1`, `api-2`, and `worker`;
- backfill `retrieval_nodes` from existing ready chunks.

Not required:
- PDF re-upload;
- OCR rerun;
- Docling rerun;
- chunk regeneration;
- chunk re-embedding;
- PostgreSQL restart;
- Valkey restart;
- Nginx restart.

The backfill uses `inference-ingest`, so heavy embedding work remains on the background CPU set rather than the online query inference process.

## Rolling compatibility

A 0.9 API knows how to fall back to legacy serial `/rerank` if it temporarily reaches an older inference-query image during deployment. This keeps deployment ordering flexible. Batch reranking activates automatically once the 0.9 inference-query image is healthy.

`CorpusIntelligence.discover()` also fails open if the new `retrieval_nodes` table is unavailable during a mixed rollout; ordinary chunk retrieval remains functional.

## Backfill strategy

The backfill is resumable. Without `--rebuild`, the script selects only ready documents that do not yet have retrieval nodes. Therefore repeated bounded batches progress through the corpus:

```bash
docker compose run --rm --no-deps api-1 \
  python scripts/build_retrieval_intelligence.py --limit 100
```

Re-run until it prints `indexed_documents=0`.

A full one-shot run is also supported:

```bash
docker compose run --rm --no-deps api-1 \
  python scripts/build_retrieval_intelligence.py
```

For a 1000-PDF production corpus, bounded batches are preferable because they give operators explicit control over background load and rollback timing.

## Rollback

Application rollback does not require dropping `retrieval_nodes`; 0.8 ignores the additive table. Restore the source backup, rebuild/recreate the previous API/inference/worker images, and leave PostgreSQL/chunks/volumes intact.

Only run `alembic downgrade 0003_query_performance_indexes` if there is a specific reason to remove the secondary table. It is not required for a normal code rollback.
