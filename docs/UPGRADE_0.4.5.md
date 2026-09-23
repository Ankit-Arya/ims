# Upgrade from 0.4.4 to 0.4.5

Release 0.4.5 is a web/API-only workspace update. It does **not** change ingestion, chunking, Docling/OCR, embedding generation, Celery worker behavior, Valkey queue semantics or database schema.

## UI changes

- Desktop left navigation rail for Ask, My Questions, Knowledge, My PDFs and Users.
- Right activity rail with signed-in user summary, total question count, available/ready/indexing PDF counts and the five most recent questions.
- Chat-style Ask workspace with a persistent bottom composer.
- Mode and source scope live inside the composer.
- Selected-PDF chooser is anchored to the composer and includes PDF search.
- Processing stages auto-follow while a query is running; when the final answer arrives, the answer header is brought into view.
- Multiple questions asked during the current browser session remain visible in the Ask transcript, but each request remains independent and no prior Q&A is sent as conversational context.
- Knowledge and My PDFs have title/filename/metadata search, status filters and fixed-height scrollable lists suitable for large libraries.
- Responsive behavior keeps the desktop three-column layout where space permits, collapses the right rail on narrower screens and converts navigation to a compact horizontal rail on phones.

## API change

New authenticated endpoint:

```text
GET /api/v1/dashboard/summary
```

It returns user-scoped counts and recent questions for the activity rail. Document counts use the same row-level access predicate as normal document retrieval, so personal PDFs are still visible only to owners and explicitly shared users.

## No migration / reindex

No Alembic migration is required. Existing documents, chunks, embeddings and queued ingestion tasks remain valid.

## Safe deployment while PDFs are queued

If ingestion is currently active, rebuild/recreate **only the API**:

```bash
cd ~/institutional-knowledge-engine

docker compose build api

docker compose run --rm --no-deps api \
  python -c "import ike; print(ike.__version__)"

docker compose up -d --no-deps --force-recreate api
```

Expected version:

```text
0.4.5
```

Do not restart `worker`, `inference`, `valkey` or `postgres` merely to apply this release.
