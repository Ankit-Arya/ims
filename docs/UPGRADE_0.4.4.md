# Upgrade from 0.4.3 to 0.4.4

Release 0.4.4 fixes broad role/responsibility queries such as `All duties of SC` without hard-coding any organisation-specific acronym or filename.

## What changes

- Broad duties/responsibilities/functions/role questions are recognized as coverage-sensitive Research queries.
- Short role identifiers are resolved from corpus evidence. For example, if the corpus contains a heading such as `Responsibilities of Station Controller`, the initials can resolve `SC` to `Station Controller`; no fixed `SC = ...` map exists in code.
- Role coverage is section-aware as well as document-aware, so multiple responsibility sections from a governing document can survive alongside specialised SOPs.
- Canonical responsibility headings and `ROLE shall/must/...` clauses are protected before ordinary global reranking.
- Role queries skip non-deterministic LLM search expansion.
- The expensive CPU rerank pool is bounded specifically for role-coverage queries because deterministic structural discovery protects recall.
- `coverage_complete` now reflects role alias/section coverage. A high semantic score alone cannot imply exhaustive coverage.
- Retrieval traces include role subject/aliases, discovered sections/documents, and fused/reranked candidate previews.

## Ingestion impact

None. This release does not modify Docling, OCR, chunking, embedding generation, Celery ingestion jobs, Valkey queue semantics, or database schema.

If PDFs are currently queued or processing, deploy this patch by rebuilding/recreating **only the API**. Leave `worker`, `inference`, `postgres`, and `valkey` running.

## Apply

```bash
docker compose build --no-cache api
docker compose up -d --force-recreate api
```

No database migration and no document reprocessing are required.

## Optional configuration

All new values have safe defaults, so no `.env` change is required:

```text
ROLE_ALIAS_SCAN_TOP_K=1200
ROLE_COVERAGE_MAX_ALIASES=4
ROLE_COVERAGE_SCAN_TOP_K=240
ROLE_COVERAGE_MAX_DOCUMENTS=24
ROLE_COVERAGE_MAX_SECTIONS=32
ROLE_COVERAGE_SECTIONS_PER_DOCUMENT=8
ROLE_COVERAGE_MAX_EVIDENCE_K=32
ROLE_COVERAGE_RERANK_POOL=40
ROLE_COVERAGE_RERANK_TOP_K=20
```

## Verification

After deployment, repeat the same query and inspect the trace:

```sql
SELECT jsonb_pretty(retrieval_trace)
FROM query_logs
WHERE lower(question)=lower('All duties of SC')
ORDER BY created_at DESC
LIMIT 1;
```

Expected trace characteristics include:

```text
resolved_mode = research
coverage_sensitive = true
coverage_kind = role
role_coverage_sensitive = true
role_subject = SC
role_aliases = [...]            # corpus-derived, e.g. Station Controller if supported
role_sections_discovered > 0
role_documents_discovered > 0
```

If the configured section/document bounds are exceeded, `coverage_complete` is false and answer confidence is not allowed to imply exhaustive coverage.
