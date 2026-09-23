# Troubleshooting — 0.5.0 quick checks

If Q&A slows while ingestion runs, first verify the worker is using `INGESTION_INFERENCE_URL=http://inference-ingest:8090` and APIs use `QUERY_INFERENCE_URL=http://inference-query:8090`; check container CPU sets and inference metrics. If exact/role search remains slow, verify migration `0003_query_performance_indexes` is applied and the trigram index exists. If the host approaches OOM, do not raise ingestion concurrency; inspect cgroup memory and the specific PDF.

If Nginx returns upstream errors after API recreation, check both `api-1` and `api-2` health and run `nginx -t`; Nginx upstream names are Docker service names. If a source visual fails, verify the original PDF and `parsed_path` still exist under shared storage and that the requesting user can access the document.

---

## Historical / earlier-release notes


## API container is `Restarting (1)` and `localhost:8080` is unreachable

First inspect the startup traceback:

```bash
docker compose logs --tail=200 api
```

Release 0.1.1 fixes an API/worker dependency-boundary defect present in the initial 0.1.0 handoff: the API route module imported worker task implementations, which transitively imported Docling even though Docling is intentionally installed only in the worker image. The corrected API submits Celery jobs by stable task name through `ike.services.task_dispatch`, while only the worker imports ingestion implementations.

After updating from 0.1.0, rebuild **both** API and worker so the sender and worker share the explicit task-name contract:

```bash
docker compose build api worker
docker compose up -d --force-recreate api worker
docker compose ps
docker compose logs --tail=100 api worker
```

Expected result: `api` remains `Up`/healthy and port `8080` is published. Verify:

```bash
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
```

## `OPENAI_API_KEY is not configured`

Set `OPENAI_API_KEY` in `.env` and recreate the API/worker containers:

```bash
docker compose up -d --force-recreate api worker
```

## Inference container repeatedly restarts

Inspect:

```bash
docker compose logs --tail=300 inference
```

Likely classes:

- model download/network failure
- insufficient memory
- embedding dimension mismatch
- corrupted model cache

The configured BGE-M3 embedding output must remain 1024 dimensions with the initial schema.

## Worker downloads models after every build

Normal image rebuilds use a persistent `model_cache` Docker volume at `/models` for Hugging Face and Docling artifacts. Confirm you have not used:

```bash
docker compose down -v
```

Also confirm `HF_HOME=/models/huggingface` is present in `.env`.

## NVIDIA packages appear on CPU build

The provided Dockerfiles install PyTorch/torchvision from:

```text
https://download.pytorch.org/whl/cpu
```

before the remaining dependencies. If you changed the Dockerfile or dependency resolution order, restore that step. Use:

```bash
docker history <image>
```

and build logs to identify the layer pulling CUDA libraries.

## `Embedding dimension mismatch`

Do not force-cast or truncate vectors. Either restore `BAAI/bge-m3`/1024 or create a deliberate vector-schema migration and re-embed every chunk.

## PDF is ready but answer misses known text

1. Use `/api/v1/debug/retrieval`.
2. If the text is absent, inspect `/data/parsed/<document-id>/canonical_document.json` in the worker/shared volume.
3. If extraction contains it but retrieval does not, add the query/evidence to the golden set and inspect dense/lexical/exact candidate counts.
4. If retrieval contains it but answer does not cite it, the problem is downstream generation/completeness.

## Scanned PDF extraction is poor

The default uses Docling + Tesseract CLI. Possible experiments, measured against extraction fixtures/golden questions:

- higher-quality OCR engine supported by Docling
- full-page OCR for specific scanner profiles
- pre-rotation/de-skew pipeline
- page-image/VLM extraction on only the pages where classical OCR fails

Do not switch the entire corpus to expensive visual extraction before measuring which pages need it.

## Duplicate document upload rejected

The same active SHA-256 file already exists. This prevents duplicate evidence from distorting ranking. A ready/processing duplicate remains rejected. An identical failed document is retried in place in 0.3. If a genuinely separate copy is intentional for controlled testing, the API still exposes `allow_duplicate=true`, but normal UI workflows should keep deduplication enabled.

## Uploaded document not shown in Q&A

A document participates in Q&A only if:

- lifecycle is active
- ingestion is ready
- effective date window includes today
- user passes ACL

The Documents management list can still display queued/failed/future documents to authorized users.

---

## 0.2: a simple Q&A takes minutes

First confirm which route actually ran. The browser shows `Direct Q&A` or `Research`; query history also stores `resolved_mode`.

For a focused lookup, explicitly select **Direct Q&A** and compare.

Relevant components:

1. query embedding + SQL retrieval
2. local CPU cross-encoder reranker
3. external answer model

Direct mode in 0.2 uses the fast model and smaller retrieval/rerank budgets. Research is intentionally slower because it adds expansion, broader retrieval, the strong model and verification.

If Direct is still slow, use retrieval diagnostics and query trace timings before reducing quality settings.

## 0.2: acronym answer misses another documented meaning

Use `/api/v1/debug/retrieval?q=<ACRONYM>` and inspect whether the missing source chunk is present.

0.2 includes a document-diverse acronym/definition retrieval arm. If a meaning is still missing:

1. confirm the text exists in stored chunks;
2. confirm the missing document is accessible/effective/ready;
3. confirm the definition chunk appears in lookup/retrieval evidence;
4. if retrieved but not answered, classify as generation/completeness failure;
5. if absent from chunks, fix extraction before retrieval.

Do not hardcode the acronym into routing or prompts.

## 0.2: OCR still logs `OSD failed` / `Too few characters`

These are often region-level Tesseract orientation-detection warnings. Judge the ingestion job by its final status and extraction coverage, not the presence of the word `ERROR` in a third-party OCR log line.

0.2 reduces unnecessary OCR with:

```text
DOCLING_OCR_LANGUAGES=eng
DOCLING_OCR_MODE=pdf_aware_layout_regions
```

If a specific rotated/scanned page is actually missing text, inspect the canonical Docling output and treat that as an extraction-quality defect.

## 0.2: bulk ingestion is too slow

Check host RAM/CPU and one-document baseline first. Then:

```powershell
docker compose up -d --scale worker=2
```

or set:

```text
INGEST_WORKER_CONCURRENCY=2
```

Do not combine high container replicas and high per-container concurrency until memory has been measured.

The `worker` queue handles ingestion only. `report-worker` handles reports separately.

## Report form shows objective validation

Reports require a meaningful `objective` (minimum 5 characters). The 0.2 browser validates this before calling the API and formats backend validation errors into readable field messages instead of displaying the raw Pydantic error JSON.

## 0.2: queue hundreds/thousands of PDFs without browser selection

Use the recursive folder uploader:

```powershell
python scripts/bulk_upload.py "D:\Organisation PDFs" --password "<admin-password>" --parallel 4
```

This only parallelizes upload/queuing. It does not create four Docling processes. Control ingestion throughput separately with `INGEST_WORKER_CONCURRENCY` or multiple `worker` replicas after measuring RAM/CPU.

## Failed PDF: `[Errno -3] Temporary failure in name resolution`

0.3 handles two common transient network/DNS surfaces:

1. Docling model artifacts are prefetched by `model-bootstrap` before ingestion workers start.
2. Calls from ingestion/retrieval to the internal inference container retry transient connect/name-resolution failures.

Check bootstrap:

```powershell
docker compose logs --tail=150 model-bootstrap
```

If first-time model bootstrap failed:

```powershell
docker compose run --rm model-bootstrap
```

After connectivity is healthy, press **Retry processing** on the failed document. Do not rename the PDF to bypass deduplication.

## Renamed failed PDF still says duplicate

This is expected because deduplication is based on SHA-256 content, not filename.

In 0.3 an identical **failed** accessible document is retried automatically if re-uploaded. The simpler path is the document-row **Retry processing** button.

An identical ready document remains rejected to avoid duplicate evidence.

## Delete an uploaded PDF and processed data

Use **Delete** on a ready/failed document. Active queued/processing documents cannot be deleted to avoid racing the ingestion worker.

Delete removes the document/chunks plus original and parsed filesystem directories.

## Worker uses less CPU than 75%

`INGEST_CPU_TARGET_PERCENT=75` is an input to the concurrency planner, not a host CPU guarantee. The memory guard may choose fewer processes, and some PDF stages can be I/O/model-service bound.

Check the chosen plan:

```powershell
docker compose logs worker | Select-String "IKE ingestion worker"
```

Then inspect:

```powershell
docker stats
```

If RAM is comfortable and the worker chose a low memory-limited concurrency, calibrate `INGEST_MEMORY_GB_PER_PROCESS` from observed peak memory. Do not simply set a very high explicit concurrency.

## Worker/container runs out of memory after 0.4

Reduce processing concurrency:

```text
INGEST_WORKER_CONCURRENCY=2
```

or lower `INGEST_MAX_CONCURRENCY` / increase `INGEST_MEMORY_GB_PER_PROCESS`, then recreate only the worker.

## My PDF is visible to the wrong user

Treat this as a security issue. Capture the document ID and accounts involved, stop sharing new content, and test the API directly:

```text
GET /api/v1/library
GET /api/v1/documents/{id}
```

An unshared user should not receive the document. Personal access is owner-or-explicit-share only; admin role is not intended to bypass this product-level boundary.

## Deep Analysis says too many PDFs

The Ask screen uses `REPORT_MAX_DOCUMENTS`. Switch Sources to **Choose PDFs** and select a smaller set, or raise the limit only after considering context/token cost and report-worker latency.

## Mobile layout looks desktop-sized

Confirm the browser is loading the current 0.4 static files and is not serving a cached 0.3 asset through a proxy. Hard-refresh/clear site cache and verify the HTML contains the viewport meta tag. The phone layout switches navigation to the bottom at the mobile breakpoint.