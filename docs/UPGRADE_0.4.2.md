# Upgrade from 0.4.1 to 0.4.2

Release 0.4.2 fixes cross-document completeness and reproducibility for procedure/requirements Q&A.

## What changes

- procedure/requirements questions use deterministic topic extraction for document coverage discovery;
- one relevant seed passage is reserved from each matching accessible document (up to the configured cap);
- evidence budget grows with the number of matching document variants, within a hard maximum;
- synthesis is instructed to separate materially different procedures by document / rolling-stock variant;
- reset, post-procedure movement and exceptional methods are kept subordinate to the correct source variant;
- verification now treats omission/merging of represented procedural variants as a failure;
- high reranker score no longer implies high corpus-completeness confidence when the coverage pass is incomplete;
- short coverage-sensitive procedure questions no longer depend on non-deterministic LLM query expansion.

## Database/index impact

No migration and no reindex are required. Existing documents, chunks and embeddings remain valid.

## Apply

Overlay the 0.4.2 files over 0.4.1, preserve `.env`, then rebuild/recreate the API:

```bash
docker compose build --no-cache api
docker compose up -d --force-recreate api
```

Verify:

```bash
curl -fsS http://127.0.0.1:8080/health/ready
docker compose logs --tail=100 api
```

## New optional settings

```text
COVERAGE_SCAN_TOP_K=800
COVERAGE_MAX_DOCUMENTS=16
COVERAGE_EVIDENCE_PER_DOCUMENT=2
COVERAGE_MAX_EVIDENCE_K=32
```

The code provides these defaults even if they are not added to an existing `.env`.
