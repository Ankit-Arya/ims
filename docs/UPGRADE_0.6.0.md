# Upgrade to IMS 0.6.0 — Operational Quick Reference retrieval

IMS 0.6.0 evolves the 0.5.0 architecture without rebuilding the indexed corpus.

## Data impact

- Database migration: **NO new migration from 0.5.0**. Existing `documents.extra_metadata` stores optional derived operational profiles and Valkey stores short-lived structured user context.
- PDF reprocessing: **NO**.
- OCR rerun: **NO**.
- Re-chunking: **NO**.
- Re-embedding: **NO**.

The reason table-aware retrieval can be improved query-side is that the existing ingestion already uses Docling `HybridChunker(... repeat_table_header=True)`, stores `contextual_text`, marks table chunks, persists `source_metadata`, and retains canonical Docling JSON. 0.6.0 therefore uses those stored structures instead of changing ingestion.

## Major changes

1. Dedicated generic overview lane configured with `OVERVIEW_DOCUMENT_PATTERN=MRGR`.
2. Operational retrieval is separated from overview evidence; overview evidence is explanatory only.
3. Separator-tolerant technical identifier expansion: `RS10`, `RS-10`, and `RS 10` are complementary search forms.
4. Coverage-sensitive Research now uses semantic expansion **and** deterministic coverage.
5. Dedicated table lexical lane, small table fusion protection, and table-aware rerank/prompt text.
6. Section-aware reconstruction uses existing `section_path` and chunk ordinal metadata.
7. Optional document metadata enrichment writes derived profiles into `documents.extra_metadata.operational_profile`; raw chunks are untouched.
8. Structured Line / rolling-stock context is retained in Valkey for eight hours and can be cleared in the Ask UI.
9. Risk-based verification avoids an extra verification LLM call solely because a query is Research.
10. Final answer text streams progressively; Ask supports real Stop/cancel checkpoints and one-click Copy.
11. Reranker telemetry now records candidate/token/batch information and uses length-bucketed batches to reduce CPU padding waste.

## Optional existing-corpus enrichment

After deployment, run:

```bash
python -m scripts.enrich_existing_documents
```

This reads existing titles, filenames, and the first stored chunks and writes derived metadata only. It does not call Docling, OCR, chunking, or embedding.

## Local use

The full 0.6.0 archive is self-contained application source. It does not require applying previous patch ZIPs. A patch ZIP is only a convenient delta for upgrading an existing 0.5.0 tree.

For local HTTP use, do not blindly reuse production HTTPS settings. Create `.env.local` from the existing `.env`:

```bash
python scripts/make_local_env.py
```

Then, on a machine with at least four logical CPUs:

```bash
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml \
  up -d postgres valkey inference-query inference-ingest model-bootstrap migrate api-1 worker report-worker
```

Open `http://127.0.0.1:8081`.

The OpenAI API key, application secret, database password, and internal service token may be copied from the production `.env` into `.env.local`, but `APP_ENV`, `SESSION_COOKIE_SECURE`, CPU affinity, and local HTTPS settings should use the local profile.
