# Configuration — 0.8.0

Important current-VM production defaults are in `.env.example`. In particular: `QUERY_INFERENCE_URL`, `INGESTION_INFERENCE_URL`, `INGEST_WORKER_CONCURRENCY=1`, CPU/memory isolation, bounded Q&A admission, `HELPFUL_CONTEXT_MODE=rich`, and lazy-visual controls. Real secrets belong only in `.env`.

Do not raise ingestion concurrency on the current VM without a representative OCR/table-heavy benchmark; concurrency 3 previously caused severe host pressure. Do not switch `RERANK_BACKEND` from `torch` until `scripts/benchmark_inference.py` plus the golden set show acceptable ranking quality.

---

## Historical / earlier-release notes


Configuration comes from `.env`. `scripts/init_env.py` starts from `.env.example` and generates local secrets.

## Ingestion throughput

### Recommended laptop defaults

```text
INGEST_WORKER_CONCURRENCY=auto
INGEST_CPU_TARGET_PERCENT=75
INGEST_MAX_CONCURRENCY=8
INGEST_MEMORY_GB_PER_PROCESS=2.5
INGEST_MEMORY_RESERVE_GB=3.0
DOCLING_NUM_THREADS_PER_DOCUMENT=2
DOCLING_OCR_BATCH_SIZE=2
DOCLING_LAYOUT_BATCH_SIZE=2
DOCLING_TABLE_BATCH_SIZE=2
TESSERACT_OMP_THREAD_LIMIT=1
```

### `INGEST_WORKER_CONCURRENCY`

- `auto`: compute safe document-level Celery prefork concurrency.
- integer: explicit override.

Use an integer only after measuring representative PDFs. An override bypasses the automatic CPU/memory concurrency calculation, so it can overload a small machine.

### `INGEST_CPU_TARGET_PERCENT`

Percentage of Docker-visible logical CPUs used to derive the document-process CPU budget. Default `75`.

It is a target. Memory can lower actual concurrency.

### `INGEST_MAX_CONCURRENCY`

Hard upper bound on ingestion document processes. Default `8`.

### Memory guard

`INGEST_MEMORY_GB_PER_PROCESS` is a planning estimate, not an allocator. `INGEST_MEMORY_RESERVE_GB` keeps headroom for the OS/Docker/other services.

For table-heavy or image-heavy manuals, observe real peak memory and raise the per-process estimate if needed.

### Docling per-document CPU

`DOCLING_NUM_THREADS_PER_DOCUMENT` controls `AcceleratorOptions.num_threads` for the Docling pipeline.

Batch-size settings control the threaded pipeline's OCR/layout/table queues. Larger values are not automatically faster on a CPU laptop and may use more memory.

### Tesseract

`TESSERACT_OMP_THREAD_LIMIT=1` constrains each OCR subprocess. This is important when multiple PDFs are processed concurrently.

The Docker worker uses a wrapper at `/usr/local/bin/tesseract` that sets `OMP_THREAD_LIMIT` only for Tesseract instead of globally throttling Docling/PyTorch.

## OCR

```text
DOCLING_OCR=true
DOCLING_OCR_LANGUAGES=eng
DOCLING_OCR_MODE=pdf_aware_layout_regions
DOCLING_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate
```

Keep only languages your corpus needs. Examples:

```text
DOCLING_OCR_LANGUAGES=eng
DOCLING_OCR_LANGUAGES=eng,hin
```

`accurate` table extraction remains the default because technical manuals often encode operational information in tables. Benchmark `fast` against an extraction golden set before changing it globally.

## Deep Analysis

```text
REPORT_MAX_DOCUMENTS=20
REPORT_PACK_MAX_CHARS=24000
REPORT_MAX_PACKS=120
REPORT_PARALLELISM=3
REPORT_REDUCE_BATCH_CHARS=60000
REPORT_SYNTHESIS_MAX_CHARS=120000
REPORT_REDUCE_MAX_ROUNDS=4
```

Deep Analysis is deliberately bounded. Increasing these settings can increase token use and latency materially.

## Retrieval

The defaults define separate Direct and Research candidate budgets. Do not tune them from one anecdotal question; use the evaluation set and compare Recall@K / rerank quality / answer completeness.

The current answer-usefulness policy is deliberately rich but evidence-bound:

```text
HELPFUL_CONTEXT_MODE=rich
HELPFUL_CONTEXT_MAX_SECTIONS=6
DIRECT_MAX_OUTPUT_TOKENS=3200
LLM_MAX_OUTPUT_TOKENS=7000
```

The six related sections are a ceiling rather than a target. A generic/vague prompt may receive definitions, applicability, prerequisites, conditions, limits, exceptions and scenario differences when the retrieved corpus supports them; a simple prompt is not padded with unrelated material.

Important values include:

```text
DENSE_TOP_K=80
LEXICAL_TOP_K=80
EXACT_TOP_K=30
FUSED_TOP_K=80
RERANK_TOP_K=16
ANSWER_EVIDENCE_K=12

DIRECT_DENSE_TOP_K=30
DIRECT_LEXICAL_TOP_K=40
DIRECT_EXACT_TOP_K=30
DIRECT_FUSED_TOP_K=36
DIRECT_RERANK_TOP_K=12
DIRECT_EVIDENCE_K=10
```

## Compositional evidence planning (0.7.0)

```text
COMPOSITIONAL_PLANNING_ENABLED=true
COMPOSITIONAL_SEMANTIC_PLANNING_ENABLED=true
COMPOSITIONAL_MAX_GOALS=12
COMPOSITIONAL_MAX_QUERIES=14
COMPOSITIONAL_QUERIES_PER_GOAL=3
COMPOSITIONAL_GOAL_CANDIDATE_RESERVE=2
COMPOSITIONAL_GOAL_EVIDENCE_PER_GOAL=2
COMPOSITIONAL_MAX_EVIDENCE_K=32
COMPOSITIONAL_RERANK_POOL=28
COMPOSITIONAL_RERANK_TOP_K=16
COMPOSITIONAL_LOOKUP_SCAN_TOP_K=160
COMPOSITIONAL_PLANNER_MAX_OUTPUT_TOKENS=1800
COMPOSITIONAL_AUDIT_ENABLED=true
COMPOSITIONAL_AUDIT_MAX_EVIDENCE=24
COMPOSITIONAL_AUDIT_MAX_OUTPUT_TOKENS=1800
COMPOSITIONAL_RECOVERY_ENABLED=true
COMPOSITIONAL_RECOVERY_MAX_GOALS=4
```

The planner creates bounded evidence requirements before retrieval. `COMPOSITIONAL_MAX_GOALS` and `COMPOSITIONAL_MAX_QUERIES` prevent arbitrary wording from causing unbounded fan-out. Required goals reserve candidates before the global rerank prefilter. `COMPOSITIONAL_RERANK_POOL` is intentionally lower than the generic Research pool because goal-level reservation protects branch recall before the expensive CPU cross-encoder.

Semantic planning and goal auditing use the fast LLM and are evidence-planning/verification steps only; they are not allowed to answer from model knowledge. Disable either feature for benchmarking with the corresponding flag, but expect weaker handling of indirect, conditional and multi-hop wording. Recovery is limited to one workflow pass and at most the configured number of unresolved goals.

Do not increase these values from one anecdotal question. Evaluate required-goal recall, goal-completeness, relationship correctness, recovery rate and latency using `eval/compositional_070_cases.json` plus corpus-specific golden questions.

## Retrieval control plane (0.8.0)

```text
PRIORITY_RETRIEVAL_ENABLED=true
PRIORITY_DOCUMENT_PATTERNS=
PRIORITY_MAX_QUERIES=8
PRIORITY_DENSE_TOP_K=56
PRIORITY_LEXICAL_TOP_K=56
PRIORITY_RELAXED_LEXICAL_TOP_K=48
PRIORITY_SECTION_TOP_K=36
PRIORITY_EXACT_TOP_K=24
PRIORITY_FUSED_TOP_K=48
PRIORITY_RERANK_TOP_K=20
PRIORITY_EVIDENCE_K=18
PRIORITY_RECOVERY_ENABLED=true
PRIORITY_RECOVERY_MAX_QUERIES_PER_GOAL=4

RELAXED_LEXICAL_ENABLED=true
RELAXED_LEXICAL_TOP_K=36
RELAXED_LEXICAL_WEIGHT=0.92
SECTION_NAVIGATION_ENABLED=true
SECTION_NAVIGATION_TOP_K=28
SECTION_NAVIGATION_WEIGHT=1.30
SECTION_NAVIGATION_SCAN_MULTIPLIER=4

GOAL_LOCAL_RERANK_ENABLED=true
GOAL_LOCAL_RERANK_CANDIDATES=10
GOAL_LOCAL_RERANK_TOP_K=4
COMPOSITIONAL_DEFINITION_EVIDENCE_PER_GOAL=6

RETRIEVAL_QUALITY_GATE_ENABLED=true
RETRIEVAL_REPAIR_MAX_OUTPUT_TOKENS=1200
```

`PRIORITY_DOCUMENT_PATTERNS` is an administrator-controlled title/filename family list. When blank, it inherits `OVERVIEW_DOCUMENT_PATTERN` for upgrade compatibility. Priority documents are searched as a separate stage and Auto broadens only after the required evidence goals remain incomplete. Explicit `document_ids` are always hard scope.

Relaxed lexical retrieval is an additional recall lane; it does not weaken the existing precise lexical query. Section navigation uses existing `section_path`, `contextual_text` and generated FTS vectors and therefore does not require corpus reprocessing. Goal-local reranking protects required evidence branches from global crowd-out. Retrieval repair is search-only: hypotheses can help locate evidence but are not facts.

Do not tune these budgets around one regression question. Use source/page/section recall, required-goal recall, answer/citation quality and latency/token measurements across the golden set.

## Models

```text
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANK_MODEL=BAAI/bge-reranker-v2-m3
ML_DEVICE=cpu
```

The current DB vector schema is fixed at 1024 dimensions. Changing embedding dimensionality requires a deliberate schema/reindex migration.

## UI

The light/dark preference is browser-local and has no `.env` setting. The UI uses the OS/browser preferred scheme on first use.

## Security-sensitive values

Never commit:

- `APP_SECRET_KEY`
- `POSTGRES_PASSWORD`
- `INTERNAL_SERVICE_TOKEN`
- `OPENAI_API_KEY`
- bootstrap administrator password

Use proper external secret management for organisational production deployment.

## Cross-document procedure coverage (0.4.2)

```text
COVERAGE_SCAN_TOP_K=800
COVERAGE_MAX_DOCUMENTS=16
COVERAGE_EVIDENCE_PER_DOCUMENT=2
COVERAGE_MAX_EVIDENCE_K=32
```

Procedure/requirements questions perform a deterministic document-discovery pass before synthesis. The goal is to prevent globally high-scoring chunks from one manual from crowding out a materially different procedure in another accessible manual. `COVERAGE_MAX_DOCUMENTS` and `COVERAGE_MAX_EVIDENCE_K` are hard safety bounds; if discovery exceeds them, answer confidence is not allowed to imply complete corpus coverage.

## Exhaustive role/responsibility coverage (0.4.4)

```text
ROLE_ALIAS_SCAN_TOP_K=120
ROLE_ALIAS_FALLBACK_SCAN_TOP_K=240
ROLE_COVERAGE_MAX_ALIASES=4
ROLE_COVERAGE_SCAN_TOP_K=240
ROLE_COVERAGE_MAX_DOCUMENTS=24
ROLE_COVERAGE_MAX_SECTIONS=32
ROLE_COVERAGE_SECTIONS_PER_DOCUMENT=8
ROLE_COVERAGE_MAX_EVIDENCE_K=32
ROLE_COVERAGE_RERANK_POOL=40
ROLE_COVERAGE_RERANK_TOP_K=20
```

Broad duties/responsibilities/functions/role queries first resolve short identifiers from subject-specific corpus evidence, using the bounded structural fallback only when needed. Section discovery then compiles the evidence-derived role names into **one structural PostgreSQL FTS query** instead of issuing exact+lexical SQL repeatedly for many formulations. Multiple responsibility sections from one governing document can survive alongside specialised documents. The role-specific rerank pool is intentionally smaller than generic Research because structurally discovered sections are protected before reranking. If alias resolution or configured coverage bounds are incomplete, confidence is capped and the answer must not claim exhaustiveness.

## Ingestion integrity and timeout (0.4.3)

`DOCLING_TIMEOUT_SECONDS` is now `3600` by default for fresh installations. A value of `0` disables Docling's internal per-document timeout.

IMS 0.4.3 rejects any Docling conversion result that is not `SUCCESS`. A timeout or page-level partial conversion therefore cannot be marked `ready`; the job fails and preserves the previous chunk set until a complete reprocess succeeds.

For large CPU-only manuals, tune timeout and document-level concurrency together. A higher number of simultaneous Docling processes can increase aggregate throughput but also increase per-document wall-clock time and memory pressure.