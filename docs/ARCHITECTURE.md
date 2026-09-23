# Architecture — 0.9.0

## Runtime topology

```text
Users -> Nginx/HTTPS -> api-1 + api-2 -> PostgreSQL/Valkey -> inference-query (BGE-M3 + reranker)
PDF queue -> worker(concurrency=1) -> Docling/OCR -> inference-ingest (BGE-M3 only)
```

The online and background ML planes use the same embedding model/version but different processes, locks, CPU sets and memory ceilings. Background ingestion cannot acquire the live query embedding gate. APIs remain lightweight and share no conversational Q&A memory. ACL filtering still occurs before evidence reaches an LLM.

The current VM starting split is CPUs `0-5` online and `6-7` background; treat it as a measured starting configuration rather than universal tuning. Existing vectors remain `BAAI/bge-m3`, dimension 1024.

Ask is answer-first: navigation and query controls are on the left; activity is an optional drawer. Visual evidence is lazily rendered only from retrieved/authorised source pages.

---

## Q&A retrieval control plane in 0.8.0

Online Q&A now uses an evidence-driven staged control flow. Administrator-configured or explicitly selected priority sources are searched before the wider corpus. Retrieval combines dense, precise lexical, relaxed lexical, section-navigation and exact/lookup evidence, then reranks required evidence goals independently. A semantic evidence-quality gate decides whether the source-local result is sufficient; incomplete goals receive a materially different search-only repair pass before any broad fallback.

```text
question -> evidence plan -> source policy
         -> priority retrieval -> goal-local rerank -> evidence audit
              | complete                         | incomplete
              v                                  v
            answer                 source-local query repair
                                                   |
                                                   v
                                      broad unresolved goals only
                                                   |
                                                   v
                                      synthesis -> verification
```

Search hypotheses are never evidence. ACL filtering remains inside every retrieval SQL path. Existing `chunks.section_path` and `search_vector` provide section navigation without a schema migration.


## Q&A industrial retrieval fabric in 0.9.0

0.9 adds a secondary hierarchy and adaptive query orchestration without replacing the source chunk store.

```text
PostgreSQL
├── documents
├── chunks                 source evidence + BGE-M3 + FTS
└── retrieval_nodes        routing hierarchy
    ├── document
    ├── section
    └── concept
```

At query time the hierarchy supplies corpus vocabulary and likely source documents/families. The API then chooses a retrieval effort:

```text
FAST     -> bounded lookup/fact/primary definition
FOCUSED  -> procedure/condition/relationship with routed source scope
RESEARCH -> enumeration, variants, multi-entity, comparison, multi-goal and coverage
```

Configured priority sources are probed first with a small profile. Incomplete Fast/Focused queries use corpus-routed hybrid retrieval before escalation. Research questions keep independent evidence branches and broad coverage.

Independent goal reranks can be sent to `/rerank/batch`, which performs one cross-encoder scheduling window and reconstructs rankings per goal. During rolling deployment the 0.9 inference client falls back to legacy serial `/rerank` if the batch endpoint is not yet available.

The evidence lifecycle is monotonic at the level that matters: passages explicitly relied on by a prior semantic goal audit are protected. Recovery for unresolved goals is admitted before unused residual candidates, so a full earlier top-k does not block better evidence.

Final drafting uses adaptive goal-balanced evidence budgets rather than the entire retrieval pool. The trace records exactly which evidence IDs/chunks/documents entered the final prompt.

The secondary hierarchy is ACL-aware and fail-open: absence/failure of `retrieval_nodes` disables routing intelligence but does not bypass authorization or invalidate normal chunk retrieval.


## Historical / earlier-release notes


## Design rules

IKE 0.4 follows these rules:

1. **Parsing quality precedes retrieval quality.** PDF structure, tables, OCR and page provenance are first-class.
2. **Retrieval is deterministic and measurable.** Hybrid candidate generation + learned reranking, not agent improvisation.
3. **Workflow complexity is conditional.** Focused questions stay cheap; broader questions can expand/verify; Deep Analysis runs a separate coverage-oriented workflow.
4. **Every question is independent.** History is not conversational context.
5. **Authorization is applied before evidence enters the model context.**
6. **Organisation knowledge and personal PDFs are distinct workspaces with different ownership semantics.**
7. **Deep Analysis is an answer mode in the product, but remains an asynchronous backend workflow.**
8. **Ingestion throughput is bounded by both CPU and memory.**
9. **UI adapts to viewport and colour preference rather than assuming a desktop.**

## Document workspace model

### Organisation Knowledge

`documents.workspace_scope = organization`

Access follows organisation/department/restricted ACL metadata. Admins can manage the organisation corpus; analyst uploaders can manage their own organisation uploads.

### Personal library

`documents.workspace_scope = personal`

Access is:

```text
owner
OR
EXISTS(document_shares WHERE user_id = current_user)
```

There is deliberately no `admin OR ...` bypass inside the normal product predicate for personal documents.

`document_shares` records explicit user grants. Shares cascade away with the document.

## Retrieval authorization

The access predicate is embedded into document/chunk queries before dense/lexical/exact evidence is returned. This avoids the unsafe design:

```text
retrieve everything -> filter forbidden results later
```

and instead uses:

```text
current user ACL -> candidate retrieval -> reranking -> model evidence
```

## Q&A workflows

### Direct

```text
question
  -> hybrid candidate retrieval
  -> RRF fusion
  -> local cross-encoder rerank
  -> compact evidence set
  -> fast LLM
  -> citations
```

### Research

```text
question
  -> query expansion
  -> broader hybrid retrieval
  -> rerank
  -> structural evidence expansion
  -> strong LLM synthesis
  -> support/citation verification
```

### Auto

Obvious broad/complex prompts route directly to Research. Otherwise Auto first takes the efficient evidence path and upgrades only when the workflow rules indicate broader work is required.

## Deep Analysis workflow

Deep Analysis is surfaced beside Auto/Direct/Research in Ask, but uses `/reports` and a background worker because its workload is fundamentally different:

```text
selected accessible PDFs
  -> enumerate source chunks/sections
  -> build bounded analysis packs
  -> parallel map analysis
  -> hierarchical reduction
  -> final synthesis
  -> citation validation
  -> result in My Questions
```

This prevents a long document comparison from being reduced to ordinary top-k RAG.

The browser polls persisted progress (`stage`, `percent`, `message`) and renders it in the same current-request area used by normal Q&A.

## Responsive UI architecture

The server ships a small dependency-free HTML/CSS/JS UI. Important properties:

- viewport meta configured for phone screens;
- fluid container widths;
- desktop/tablet grid layouts collapse to a single column;
- navigation becomes a fixed bottom bar on phone widths;
- answer-mode buttons become horizontally scrollable where needed;
- tables use their own horizontal overflow container;
- dialogs use viewport-bounded width/height;
- light/dark design tokens are CSS custom properties;
- theme preference is stored in browser `localStorage`.

No server-side theme state is required.

## Ingestion architecture

```text
PDF upload
   -> immutable source storage
   -> Celery ingestion queue
   -> CPU/RAM-aware worker bootstrap
   -> N prefork document processes
       -> cached Docling pipeline per process
       -> Tesseract OCR (1 OpenMP thread per invocation)
       -> canonical document + hybrid chunks
       -> batched embedding calls
       -> PostgreSQL/pgvector
```

### Concurrency calculation

With `INGEST_WORKER_CONCURRENCY=auto`, the worker computes:

```text
target_threads = floor(logical_cpus * target_percent)
cpu_limit      = floor(target_threads / docling_threads_per_document)
memory_limit   = floor((memory - reserve) / estimated_memory_per_process)
concurrency    = min(max_concurrency, cpu_limit, memory_limit)
```

All values are clamped to at least one process.

This is intentionally safer than mapping “80% CPU” directly to Celery processes, because each Docling worker process can consume substantial RAM and each process also has internal CPU parallelism.

## Local inference

Embeddings and reranking run through a private inference service so models are loaded once per service rather than separately into the FastAPI process.

The application uses:

- BAAI/bge-m3 embeddings;
- BAAI/bge-reranker-v2-m3 cross-encoder reranking;
- PostgreSQL + pgvector for persistence/search.

## Queues

- `ingestion` — CPU-heavy PDF parsing/indexing.
- `reports` — LLM-heavy Deep Analysis jobs.

Separating the queues prevents a large report from blocking PDF processing or vice versa.

## Data ownership

PostgreSQL is the system of record for metadata, ACLs, shares, chunks, queries, reports and feedback. Source/parsed document files live on the shared persistent data volume. Valkey is coordination/queue infrastructure, not the primary record.