# Institutional Knowledge Engine / IMS

## IMS 0.10.1 - Stateless recall-safe institutional retrieval

IMS 0.10.1 builds on the 0.10 retrieval-control foundation and makes Q&A requests explicitly stateless across questions. Each new question is interpreted only from the current request, explicit current-request document scope, ACLs, and indexed institutional evidence. Previous questions, answers, rolling-stock scope, line scope, citations, and search results are not injected into later queries. The original-query preservation, global-corpus safety lane, routed-document boost, and bounded governing-reference behavior from 0.10 remain unchanged.

The release also adds corpus-aware terminology resolution, lane-reserved reranking, strict structured-controller output paths, source-authority metadata, a separate query-inference CPU-set hook, and an OpenVINO benchmark profile. Existing PDF chunks and BGE-M3 embeddings remain valid.

This is deliberately a **foundation release**: true speculative fan-out, persisted Fact/Claim atoms, deterministic table-row backfill, generalized applicability/conflict resolution, OpenTelemetry exporters, and search-backend challenger adapters are specified but not falsely presented as complete.

See:

- [`docs/RETRIEVAL_CONTROL_PLANE_0.10.0.md`](docs/RETRIEVAL_CONTROL_PLANE_0.10.0.md)
- [`docs/UPGRADE_0.10.0.md`](docs/UPGRADE_0.10.0.md)
- [`docs/UPGRADE_0.10.1.md`](docs/UPGRADE_0.10.1.md)
- [`deploy/env/0.10-query-benchmark.env`](deploy/env/0.10-query-benchmark.env)

### 0.10 invariants

1. The exact original query is always retained as Q0.
2. Only explicit user document selection may become a hard retrieval scope.
3. Likely-document routing is a soft boost and cannot eliminate the global ACL corpus.
4. Governing references are always considered when configured/marked and can never end broad retrieval.
5. Corpus terminology/fuzzy matches are search hints, never answer evidence.
6. Important retrieval lanes reserve candidate capacity before bounded cross-encoder reranking.
7. Expensive parallel/speculative execution remains disabled until independent-session concurrency is implemented and measured.

## IMS 0.9.1 — Industrial Retrieval Fabric + Optimized Corpus Intelligence

IMS 0.9.1 retains the 0.9 backend retrieval redesign for a large internal corpus where answers may live in one paragraph, several sections, several documents, or a chain of compatible rules. The release keeps the existing Docling/OCR/chunk/embedding corpus as the source of truth, but adds a lightweight **corpus-intelligence hierarchy** over those chunks so query planning can use the organisation's own terminology and route to likely documents before expensive passage retrieval.

The design is domain-generic. Production code contains no expected answers, document UUIDs, DMRC-specific regression terms, or question-specific shortcuts.


### 0.9.1 production hardening

Production-scale backfill measurements showed that 0.9.0 created too many vector-bearing concept rows. 0.9.1 keeps terminology discovery but stores high-signal terms on their parent section node, derives document/section vectors from the **existing chunk embeddings**, and adds a coverage gate so a partial secondary index can never hard-restrict factual retrieval. The backfill therefore performs no new document-embedding inference.

See [`docs/CORPUS_INTELLIGENCE_0.9.1.md`](docs/CORPUS_INTELLIGENCE_0.9.1.md) and [`docs/UPGRADE_0.9.1.md`](docs/UPGRADE_0.9.1.md).

### What changes

```text
                              OFFLINE / INGESTION
Existing PDFs -> chunks/embeddings ------------------------------------+
        |                                                               |
        +-> retrieval_nodes: document / section / corpus-term hierarchy |
                                                                         v
USER QUESTION -> semantic frame -> adaptive effort (FAST / FOCUSED / RESEARCH)
                         |                  |
                         |                  +-> corpus terminology + likely documents/families
                         v
                 cheap priority-source probe
                         |
          +--------------+------------------+
          |                                 |
      sufficient                         incomplete
          |                                 |
          v                                 v
       answer                    routed hybrid retrieval
                                  lexical + dense + exact
                                  + section navigation
                                           |
                                  goal-local BATCH rerank
                                           |
                                  monotonic evidence ledger
                                           |
                                  coverage / relation audit
                                           |
                           unresolved goals only -> repair
                                           |
                                  dynamic draft context
                                           |
                                answer + verification
```

### Industry patterns adopted

0.9 follows well-established enterprise-retrieval patterns rather than treating every question as one global vector search:

- adaptive retrieval effort: cheap path for simple lookups, planned multi-query retrieval only when the question needs it;
- hybrid lexical + dense retrieval with reciprocal-rank-style fusion;
- hierarchical document/section retrieval before chunk-level ranking;
- organisation-specific terminology discovery from the corpus itself;
- multi-query / multi-goal retrieval with batched reranking;
- evidence preservation across later recovery passes;
- dynamic generation-context budgets based on the information need;
- explicit query/evidence/draft-context telemetry for evaluation;
- map/reduce remains the separate Deep Analysis path for genuinely corpus-wide report synthesis.

The architecture rationale and public product research are documented in [`docs/INDUSTRIAL_RETRIEVAL_0.9.0.md`](docs/INDUSTRIAL_RETRIEVAL_0.9.0.md).

## Upgrade characteristics

Existing PDFs, parsed artifacts, chunks and BGE-M3 chunk embeddings remain valid. **Do not re-OCR, re-chunk or reprocess the PDF corpus.**

0.9 does require one **additive Alembic migration** creating `retrieval_nodes`, followed by a resumable secondary-index backfill from the existing chunks. The secondary index is rebuildable and can fail independently without invalidating normal chunk retrieval.

For a production VM already on 0.9.0 use the 0.9.1 patch ZIP and follow [`APPLY.md`](APPLY.md). The 0.9.1 backfill automatically replaces older 0.9.0 secondary nodes without touching PDFs or factual chunk embeddings.

## Quick validation

```bash
PYTHONPATH=src:. python3 scripts/check_091_corpus_intelligence.py
python3 -m compileall -q src scripts inference_service
```

Run the full Python suite in the Python 3.12 API image, as documented in `APPLY.md`.

## 0.9 retrieval invariants

1. **A search hypothesis is never evidence.** Corpus-derived terms can improve recall but cannot enter an answer unless a source chunk supports them.
2. **The original semantic frame cannot silently drift.** Recovery must preserve entities, material conditions, qualifiers and the requested relation.
3. **Supported evidence is monotonic.** Later searches may add stronger or contradictory evidence but may not silently evict evidence already relied on by a semantic audit.
4. **Multi-document questions retain source diversity.** Required goals are reranked separately and draft context is goal-balanced/document-diverse.
5. **Candidate presence is not answer completeness.** `goal_candidate_represented` and `evidence_requirements_complete` are distinct trace states.
6. **The final LLM context is observable.** The trace records exactly which evidence IDs, chunks, documents and approximate tokens were sent for drafting.
7. **Corpus-wide absence requires coverage.** A top-k miss cannot be converted into a corpus-negative assertion.

## Documentation

- [0.9.1 corpus intelligence optimization](docs/CORPUS_INTELLIGENCE_0.9.1.md)
- [Upgrade 0.9.0 -> 0.9.1](docs/UPGRADE_0.9.1.md)
- [0.9.0 industrial retrieval fabric](docs/INDUSTRIAL_RETRIEVAL_0.9.0.md)
- [Upgrade 0.8.0 -> 0.9.0](docs/UPGRADE_0.9.0.md)
- [0.8.0 retrieval control plane](docs/RETRIEVAL_CONTROL_PLANE_0.8.0.md)
- [0.7.0 compositional capability model](docs/COMPOSITIONAL_QA_0.7.0.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Model and retrieval](docs/MODEL_AND_RETRIEVAL.md)
- [Evaluation](docs/EVALUATION.md)
- [Operations](docs/OPERATIONS.md)
- [Security](docs/SECURITY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Branding note

The user-facing product name is **IMS — Incident Management System** for **Delhi Metro Rail Corporation (DMRC)**. Internal Python package names, Docker service names, task identifiers, metrics prefixes and compatibility keys remain `ike` intentionally to avoid a risky non-functional rename.
