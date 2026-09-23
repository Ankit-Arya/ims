# IMS 0.9.1 — Corpus Intelligence Efficiency and Coverage Safety

## Why 0.9.1 exists

The 0.9.0 architecture was directionally correct, but the first production backfill exposed two implementation issues at the real corpus scale:

1. every extracted concept became a separate 1024-dimensional BGE-M3 row containing largely duplicated section context; and
2. a partially built secondary index could return document IDs before the index covered the full accessible corpus.

The production measurement that triggered this patch was a 70-page document with 121 original chunks and 81 distinct section paths. 0.9.0 expanded that single document into 346 retrieval nodes: 1 document node, 81 section nodes and 264 concept nodes. The concept rows therefore represented more than three quarters of the secondary vectors while adding substantial duplicated text.

0.9.1 keeps the industrial retrieval architecture but changes the physical representation.

## New representation

```text
existing factual chunks + existing BGE-M3 vectors
              |
              +--> document routing node
              |      vector = normalized centroid of existing chunk vectors
              |
              +--> section routing node(s)
                     vector = normalized centroid of section chunk vectors
                     text   = bounded section context
                     terms  = headings / definitions / acronym-full-form aliases
                              stored as lexical metadata on the section node
```

There are no new standalone `concept` rows in 0.9.1. The existing database constraint still accepts the legacy type for rolling compatibility, but a current rebuild creates only `document` and `section` nodes.

## No document-embedding backfill

The backfill no longer calls `inference-ingest:/embed/documents`. It reuses the already persisted BGE-M3 chunk vectors and derives normalized centroids in-process.

This removes the dominant CPU cost observed in 0.9.0 while keeping the semantic routing signal consistent with the chunk embedding space.

## Safer terminology extraction

0.9.0 extracted broad uppercase phrases from section bodies. Real operational manuals contain many uppercase states, colours and table cells, which can become noisy query vocabulary.

0.9.1 prefers:

- section headings;
- explicitly defined terms (`means`, `refers to`, `is defined as`, `shall mean`);
- acronym/full-form pairs.

Those terms are attached to the parent section and included in PostgreSQL FTS. They are search hints, never factual answer evidence.

## Coverage gate

Corpus intelligence is secondary routing infrastructure, not the source of truth. A partial routing index must never hide an unindexed document.

For every query, 0.9.1 records:

- `scope_documents`
- `indexed_documents`
- `coverage_ratio`
- `routing_safe`

Search hints are disabled below `RETRIEVAL_INTELLIGENCE_MIN_HINT_COVERAGE_RATIO` (default 0.90). More importantly, hard document narrowing is allowed only when **every accessible document in the requested scope is indexed**.

Therefore:

```text
partial global index
  -> no hard document routing
  -> ordinary chunk retrieval remains corpus-wide

fully indexed global scope
  -> routed document narrowing allowed

explicit selected-document scope
  -> routing allowed when every selected accessible document is indexed
```

## Upgrade behavior

The backfill version is `0.9.1`. Existing documents marked with retrieval-intelligence version `0.9.0` are automatically treated as needing an upgrade. `--limit 100` can therefore be repeated safely: each completed document is marked `0.9.1`, and the next invocation advances to the next outdated/missing batch.

A document is replaced atomically: old secondary nodes are deleted and new nodes inserted in the same transaction. Interrupted work does not expose half-indexed documents.

## Expected production effect

For the first measured production document, 0.9.0 created 346 vectors. The same 81 section paths imply approximately 82 semantic routing vectors in 0.9.1 (1 document + 81 sections), a 76% reduction before considering the removal of all document-embedding inference calls.

The factual retrieval layer is unchanged: original chunks, their BGE-M3 vectors, PostgreSQL FTS, exact/table/section lanes, reranking, evidence ledger, dynamic drafting context and answer verification remain intact.
