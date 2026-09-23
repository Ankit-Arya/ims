# Models and retrieval — 0.8.0

`inference-query` loads BAAI/bge-m3 plus BAAI/bge-reranker-v2-m3 and serves live Q&A only. `inference-ingest` loads BAAI/bge-m3 only and serves document embeddings. Both preserve 1024-dimensional normalized-vector compatibility with the existing corpus.

## 0.8.0 retrieval control plane

Retrieval no longer relies on one globally competing candidate pool for every question. Natural-language requests can receive a fast semantic evidence plan. Each required goal has its own search formulations and coverage contract. Configured priority sources are searched and audited before broad corpus search; if incomplete, an LLM retrieval-repair controller generates search-only conceptual/heading/role hypotheses from the unresolved goals and retained corpus snippets.

The indexed search stack now includes separate precise FTS and relaxed OR-based FTS lanes plus section navigation over existing chunk hierarchy. The learned reranker scores bounded candidates against each required goal before fan-in. Table-specific boosting is gated by structural relevance so tables do not dominate unrelated responsibility/relationship questions.

These changes do not replace BGE-M3 or the BGE reranker and do not require re-embedding.

All query formulations in one retrieval pass are embedded as one batch. Candidate recall still comes from dense + FTS lexical + exact substring search, followed by RRF and bounded learned reranking. Exact substring search is backed by `pg_trgm` after migration 0003. Coverage-sensitive procedure/role questions retain dedicated document/section coverage instead of simply shrinking candidate budgets.

Short-role alias resolution is subject-specific first, bounded fallback second, with a short-lived ACL/corpus-revision-aware metadata cache. Final answers are not cached across users in 0.5.0.

---

## Historical / earlier-release notes


## Default embedding model

```text
BAAI/bge-m3
```

Configured dimension: `1024`.

It runs inside the private inference container and is used for both document embeddings and query embeddings. The database vector column is fixed at `vector(1024)` in this release; a different-dimensional model requires a schema migration and full re-embedding.

## Default reranker

```text
BAAI/bge-reranker-v2-m3
```

The reranker evaluates `(question, passage)` jointly after candidate generation. It is deliberately separate from the embedding model: candidate generation aims for recall; reranking spends more compute only on a bounded pool.

## Retrieval arms

### Dense

Normalized embedding cosine distance over pgvector HNSW. Filtered HNSW queries enable pgvector iterative scans to preserve recall under ACL/document filters.

### Lexical

PostgreSQL `websearch_to_tsquery('simple', ...)` over a generated `tsvector` column. This is essential for organisation-specific vocabulary, codes and exact terms.

### Exact substring

Case-insensitive substring match over contextual text. It is weighted more strongly in RRF because exact identifiers can be semantically unremarkable to an embedding model.

### Definition/acronym lookup arm

The current release includes a specialized **retrieval** path for conservative short acronym/identifier questions. It is not a hardcoded glossary.

The path:

1. extracts a likely term only from uppercase single-token identifiers or explicit lookup cues;
2. performs broad token-matching across accessible documents;
3. scores source chunks for definition-like forms (`ABC = ...`, `Expansion (ABC)`, abbreviation/nomenclature sections);
4. guarantees an initial breadth pass across documents;
5. contributes the ranking to RRF with additional weight;
6. protects likely definition-bearing chunks during evidence construction.

Why: an acronym may be repeated many times in one manual but have a different expansion in another. Ordinary top-k ranking can silently lose the second meaning.

## Mode-specific budgets

### Direct Q&A

Default candidate depths are intentionally smaller:

```text
DIRECT_DENSE_TOP_K=30
DIRECT_LEXICAL_TOP_K=40
DIRECT_EXACT_TOP_K=30
DIRECT_FUSED_TOP_K=36
DIRECT_RERANK_TOP_K=12
DIRECT_EVIDENCE_K=8
```

This is the latency-oriented path.

### Research

Uses the larger base settings:

```text
DENSE_TOP_K=80
LEXICAL_TOP_K=80
EXACT_TOP_K=30
FUSED_TOP_K=80
RERANK_TOP_K=16
ANSWER_EVIDENCE_K=10
```

and adds 2–4 retrieval-only query expansions.

### Lookup override

Short definition lookups increase exact/lookup depth and evidence breadth without forcing a full Research workflow:

```text
LOOKUP_SCAN_TOP_K=240
LOOKUP_RERANK_TOP_K=16
LOOKUP_EVIDENCE_K=16
```

This keeps `BIC`-style questions fast while reducing definition omissions.

## Fusion

Every retrieval source produces an ordered chunk-ID list. Weighted Reciprocal Rank Fusion combines them without trying to numerically calibrate incomparable raw scores.

Exact and lookup arms receive additional weight, while research-expansion queries receive slightly lower weight than the original user wording.

## Structural neighbours

For ordinary questions, bounded neighbouring chunks can be included after reranking to recover split procedures/table continuation/context.

For short definition lookups, neighbour expansion is disabled; it would consume evidence slots with adjacent unrelated glossary/table material. Those slots are instead used for alternative definition-bearing chunks.

## Generation model choice

### Direct

Uses `LLM_FAST_MODEL` with `LLM_FAST_REASONING`.

### Research

Uses `LLM_STRONG_MODEL` with `LLM_STRONG_REASONING`, followed by verification when `VERIFY_MODE=selective`.

This distinction is deliberate. A one-word acronym lookup should not pay the same reasoning/latency cost as a cross-document comparative research request.

## Chunking

Docling `HybridChunker` is aligned to the embedding tokenizer and defaults to 500 tokens. It operates on structured Docling output, merges peers and repeats table headers when tables are split.

## OCR

0.2 defaults:

```text
DOCLING_OCR_LANGUAGES=eng
DOCLING_OCR_MODE=pdf_aware_layout_regions
```

This avoids processing four language models by default and reduces redundant OCR over reliable embedded PDF text. OCR remains enabled so scanned/image regions are still available.

## What to benchmark before changing models

For retrieval:

- Recall@5/10/20
- MRR / nDCG
- expected document recall
- expected page recall
- acronym-definition coverage
- procedure completeness
- reranker latency

For Q&A:

- source-grounded correctness
- completeness
- citation correctness
- Direct p50/p95 latency
- Research p50/p95 latency
- token usage by mode

A model swap is an experiment, not a config tweak.

## Local parser artifacts in 0.3

Docling layout/table model artifacts are deployment dependencies rather than per-document network dependencies. The `model-bootstrap` service prefetches them into the persistent model volume and the PDF pipeline passes that location as `artifacts_path`.

This does not change retrieval ranking; it removes a reliability failure mode where first-use model download could fail during a user's ingestion job.


## CPU-aware PDF ingestion in 0.4

Retrieval quality begins with ingestion quality. Release 0.4 retains accurate table extraction by default but improves corpus throughput with two bounded levels of CPU parallelism:

- Celery prefork processes handle multiple documents concurrently.
- Each Docling document pipeline receives a configured `AcceleratorOptions.num_threads` budget.
- Tesseract OCR subprocesses are capped to one OpenMP thread by default so document-level parallelism does not multiply OCR threads unpredictably.

The worker chooses prefork concurrency from both CPU and memory limits when `INGEST_WORKER_CONCURRENCY=auto`. This should be benchmarked against representative manuals because parser/OCR throughput and RAM are document-dependent.

---

## 0.9.0 industrial retrieval additions

### Corpus intelligence is a routing index, not an answer index

`retrieval_nodes` stores embeddings/FTS for document metadata, section-level text and high-signal corpus terms. Its purpose is to answer **where and how should IMS search?** The final answer is still grounded only in ordinary authorized `chunks`.

This separation lets IMS learn organization vocabulary and likely source families without allowing generated/derived terminology to become factual evidence.

### Adaptive retrieval effort

The Q&A controller assigns `fast`, `focused` or `research` effort from the evidence plan. This replaces the previous tendency to run a near-research workload for simple terms.

`all_supported_variants`, enumerations, comparisons and multi-entity coverage use Research. Procedures/conditions/relationships use Focused unless additional decomposition/coverage requires Research. Bounded primary definitions/facts use Fast.

### Candidate generation and ranking

The factual retrieval stack remains hybrid. Corpus routing reduces the documents that compete on Fast/Focused requests. Research retains broad retrieval and goal-based candidate reservation.

Reranking remains cross-encoder based but independent evidence goals can be batched. This does not change the relevance semantics of each branch; it reduces serial scheduling and padding overhead.

### Dynamic drafting context

Generation-context size is separate from retrieval-candidate count and rerank count. Default draft budgets are 8/16/28 evidence units for Fast/Focused/Research, with per-goal minima and per-document diversity caps. Semantic-audit evidence is protected before score-based filling.

### Secondary-index embeddings

The secondary hierarchy uses the same BGE-M3 embedding dimension (1024) as the existing corpus. Backfill calls `inference-ingest`, not `inference-query`, so expensive bulk embedding remains on the background inference plane.
