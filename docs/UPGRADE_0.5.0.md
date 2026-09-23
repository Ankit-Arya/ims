# Upgrade to IMS 0.5.0 — production-safe Q&A plane, richer answers, and lazy visuals

IMS 0.5.0 is a coherent minor release for the current 8-vCPU / ~31-GiB CPU-only VM. It preserves the existing PDF corpus and 1024-dimensional BGE-M3 vectors while changing the service topology around them.

## Compatibility decision

| Item | Required? | Reason |
|---|---:|---|
| Database migration | **Yes** | Additive search/provenance indexes only. |
| Reprocess existing PDFs | **No** | Existing chunks, page provenance, canonical Docling JSON and source PDFs remain valid. |
| Re-embed existing chunks | **No** | Both ML planes use the same BAAI/bge-m3 model, 1024 dimensions and normalized vectors. |
| Re-upload PDFs | **No** | Storage and document IDs are unchanged. |
| Restart ingestion worker | **Yes, once during cutover** | It must switch from the old shared inference endpoint to `inference-ingest`. Queued Valkey tasks survive. |

## 1. Online/background inference isolation

**CURRENT:** one inference process serves query embeddings, document embeddings and reranking. Query/document embeddings share one model lock.

**PROBLEM:** long background document embedding can directly block an interactive query embedding and consume the same CPUs/memory.

**NEW:** `inference-query` serves `/embed/query` and `/rerank`; `inference-ingest` serves `/embed/documents` and does not load a reranker. Query and background planes use separate processes, locks, CPU sets and memory ceilings.

**IMPACT:** background embedding can no longer acquire the online query embedding gate. Existing vectors remain compatible.

**REPROCESSING:** none.

**TEST:** `tests/test_050_architecture_contract.py`; health-role checks during deployment; compare query latency with worker active vs stopped.

## 2. CPU and memory isolation

**CURRENT:** ingestion and inference can contend for the whole host; the VM has previously suffered host-wide OOM/starvation.

**PROBLEM:** a pathological OCR/table-heavy PDF can degrade SSH, API, database and Q&A simultaneously.

**NEW:** the release starts with `ONLINE_CPUSET=0-5`, `BACKGROUND_CPUSET=6-7`, `INGEST_WORKER_CONCURRENCY=1`, and explicit Compose `mem_limit` ceilings. These are benchmark starting points, not a capacity certification.

**IMPACT:** the operating system keeps headroom and background work is cgroup-confined. A worker/container failure is preferable to a host OOM; Celery late-ack/redelivery remains enabled.

**REPROCESSING:** none.

**TEST:** representative large/image-heavy PDF under load; monitor Docker/host OOM and restart counters.

## 3. Batched query embeddings

**CURRENT:** complex search formulations can issue multiple `/embed/query` requests sequentially.

**PROBLEM:** repeated HTTP round trips, lock acquisitions and inference overhead.

**NEW:** all compatible formulations for one retrieval pass are sent in one `texts[]` request. Search still preserves the original user wording as its own formulation.

**IMPACT:** fewer inference requests and explicit `query_embedding_queue_wait` / `query_embedding_execution` timing.

**REPROCESSING:** none.

**TEST:** `tests/test_050_query_intelligence_contract.py` and retrieval traces.

## 4. Role/coverage latency

**CURRENT:** short-role resolution can scan a large generic duty/responsibility pool before the actual role coverage queries execute.

**PROBLEM:** broad role questions can spend tens of seconds in pre-search work unrelated to OpenAI or embeddings.

**NEW:** subject-specific lookup runs first; generic structural fallback runs only when needed and is bounded. Resolved aliases are cached only as metadata with a key containing user identity/ACL attributes, explicit document scope and accessible-corpus revision. After alias resolution, role section discovery compiles the supported names into one structural PostgreSQL FTS query instead of issuing exact+lexical SQL repeatedly for many formulations. The release also adds a trigram index for the remaining substring exact-search path.

**IMPACT:** repeated short-role questions should avoid repeated broad alias scans while preserving ACL isolation, and each role coverage pass eliminates dozens of avoidable SQL round-trips. Exact phrase search elsewhere still has an indexable trigram path.

**REPROCESSING:** none.

**TEST:** trace `role_alias_resolution.total_ms`, `role_coverage_discovery`, `role_coverage_sql_queries` and `exact_search`; run `EXPLAIN (ANALYZE, BUFFERS)` on representative structural/exact SQL after migration.

## 5. CPU reranker runtime

**CURRENT:** PyTorch CrossEncoder is the default and measured CPU reranking can dominate query latency.

**PROBLEM:** expected multi-user throughput cannot be inferred from API replica count while one expensive local reranker remains slow.

**NEW:** the same reranker can be built with optional ONNX/OpenVINO dependencies and selected by `RERANK_BACKEND`. **Torch remains the default until the same golden set proves an optimized backend retains acceptable ranking/citation quality.** `scripts/benchmark_inference.py` measures p50/p95, queue wait, execution time and resulting order.

**IMPACT:** optimization is evaluation-gated rather than achieved by silently shrinking recall budgets.

**REPROCESSING:** none.

**TEST:** benchmark identical query/candidate fixtures at 10/20/40/80 candidates plus retrieval golden set.

## 6. Bounded Q&A admission and two API replicas

**CURRENT:** the streaming route can create unbounded per-request worker threads.

**PROBLEM:** a burst can turn into many simultaneous inference calls and memory stacks even though local ML remains finite.

**NEW:** each API replica owns a bounded Q&A executor (`QUERY_MAX_ACTIVE_PER_API`, `QUERY_MAX_WAITING_PER_API`). Nginx uses `least_conn` across `api-1` and `api-2`. The inference-query service supplies a second global model gate. Queue waiting is surfaced to the browser and Prometheus.

**IMPACT:** predictable backpressure; API replication improves HTTP/DB/OpenAI concurrency without pretending to duplicate ML capacity.

**REPROCESSING:** none.

**TEST:** `scripts/load_test_qa.py`, 100 varied requests over 300 seconds with one PDF processing.

## 7. Question understanding and helpful-answer policy

**CURRENT:** retrieval may depend too heavily on vocabulary entered by the user, while a short definition can lead to an answer that is technically correct but not sufficiently useful.

**PROBLEM:** users should not have to know the exact manual phrase or all the follow-up questions they ought to ask.

**NEW:** every request creates a corpus-generic query plan that preserves the literal original text and supplements it with normalized/semantic formulations, supported acronym/role expansion and deterministic coverage logic. `HELPFUL_CONTEXT_MODE=rich` tells the drafting contract to answer the direct need first and then proactively include **only evidence-supported, materially useful** definitions, applicability, prerequisites, limits, exceptions, scenario differences, immediate consequences and likely next-needed context. `HELPFUL_CONTEXT_MAX_SECTIONS=6` is a ceiling rather than a quota. Generic knowledge is never used to fill documentary gaps.

**IMPACT:** short/vague prompts can produce a richer professional answer without turning into an unrelated document dump.

**REPROCESSING:** none.

**TEST:** golden-set categories: single/multiple acronym meanings, vague wording, poor terminology, no-answer, multi-scenario and coverage queries.

## 8. Answer planning and targeted verification

**CURRENT:** long answers can become flat bullet lists, and a verifier rewrite can destroy good grouping.

**PROBLEM:** correctness without professional organization makes operational answers harder to use.

**NEW:** Research adds a compact answer-plan stage before drafting. The draft adapts headings, lifecycle/scenario grouping and Markdown tables to the evidence. Verification returns issues/repair instructions rather than a replacement answer; a targeted repair pass is invoked only when needed and must preserve structure.

**IMPACT:** better synthesis, scenario separation and readability; Direct avoids the extra planning call.

**REPROCESSING:** none.

**TEST:** answer-structure and verification contracts plus human/golden evaluation.

## 9. Ask-page redesign

**CURRENT:** a permanent right activity rail consumes width needed for technical answers.

**PROBLEM:** long procedures, tables, citations and figures need a wide primary reading surface.

**NEW:** navigation, Mode, Sources and PDF selection are on the left. The answer occupies the rest of the page. Activity/user stats/recent questions move to a slide-in drawer. The composer remains at the bottom of the Ask workspace and progress auto-scroll remains enabled.

**IMPACT:** answer-first desktop UX while keeping tablet/mobile responsiveness.

**REPROCESSING:** none.

**TEST:** UI contract + browser smoke test in light/dark desktop/mobile layouts.

## 10. Lazy figure/table visual evidence

**CURRENT:** source PDFs/canonical Docling JSON are stored, but the Ask answer does not resolve relevant source visuals.

**PROBLEM:** some operational questions depend on a diagram, layout, panel, figure or visually structured table.

**NEW:** after text retrieval, visual cues are detected only in retrieved evidence/user intent. IMS reuses chunk page provenance and existing canonical Docling JSON to find a likely bounding box; if no reliable box exists it renders the relevant page. The original PDF is rendered lazily with PyMuPDF, cached deterministically and served through a short-lived signed token. The endpoint reuses the same document ACL clause and never accepts a filesystem path from the browser.

**IMPACT:** no wholesale image extraction, no image embeddings and no corpus-wide reprocessing. Relevant source visuals can appear beside the answer.

**REPROCESSING:** none.

**TEST:** `tests/test_050_visual_contract.py` plus an ACL test using two users with different document access.

### Selective vision reasoning

The 0.5.0 foundation keeps `SELECTIVE_VISION_ENABLED=false` by default. Page/figure rendering is production-ready; sending a crop to an external vision model remains evaluation-gated because it changes external data transfer/cost and must be explicitly approved. Do not enable it merely for benchmark numbers.

## 11. Observability

0.5.0 separates:

- API Q&A admission wait;
- query embedding queue wait and execution;
- dense, lexical, exact and lookup search;
- procedure/role coverage discovery;
- fusion;
- rerank queue wait and execution;
- evidence construction;
- query expansion;
- answer planning;
- answer generation;
- verification;
- targeted repair;
- total query latency.

Inference containers expose their own Prometheus metrics on loopback-only ports; Nginx does not expose `/metrics` publicly.

## Migration safety

`0003_query_performance_indexes`:

1. installs `pg_trgm` if available to the DB user;
2. creates a GIN trigram index on `chunks.contextual_text`;
3. creates `(document_id,page_from,page_to)` provenance index;
4. creates `(user_id,created_at DESC)` query-history index.

It does not update any chunk, vector or document content. Indexes use `CREATE INDEX CONCURRENTLY` to reduce table blocking, but creation can still consume CPU/I/O; run the migration before the load test, not during peak usage.

## Upgrade workflow

See root `APPLY.md` for the exact safe cutover commands. The release includes `scripts/upgrade_env_050.py`, which upgrades non-secret 0.5.0 settings while preserving the existing application secret, database password/URL, internal-service token, OpenAI key and bootstrap-admin password exactly. Do **not** use `docker compose down -v`.

## Capacity statement

0.5.0 is designed for the requested 8-vCPU topology, but the release does **not** claim that the VM supports 100 Q&A requests/5 minutes until the included load test passes on the actual VM while one representative PDF is continuously processing.
