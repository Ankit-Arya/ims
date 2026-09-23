# IMS 0.9.0 — Industrial Retrieval Fabric

## Purpose

The target workload is not "chat with one PDF". It is enterprise retrieval across a large, preprocessed internal corpus where:

- the user's wording may differ from the terminology used in governing documents;
- one answer may require several documents or sections;
- an acronym may have multiple supported meanings;
- procedures may span prerequisites, ordered actions, exceptions and post-actions;
- different lines/systems/revisions can legitimately contain different rules;
- exhaustive/list questions cannot be answered safely from an ordinary top-k;
- a simple definition should not pay the same latency as a cross-document research question.

0.9 changes the backend around those constraints.

## Public industry research used for the design

The release deliberately uses public, documented retrieval patterns rather than attempting to copy proprietary internals.

### Microsoft Azure AI Search / Foundry IQ

Public documentation describes agentic retrieval that uses an LLM to decompose complex questions into focused subqueries, executes subqueries in parallel, applies keyword/vector/hybrid search and semantic reranking, then merges grounding data with source references and execution details. Azure also exposes retrieval-reasoning effort so simple requests can skip expensive planning while harder requests receive deeper retrieval.

Relevant public documentation:
- https://learn.microsoft.com/azure/search/agentic-retrieval-overview
- https://learn.microsoft.com/azure/search/search-features-list
- https://learn.microsoft.com/azure/search/hybrid-search-how-to-query

**IMS adaptation:** `fast | focused | research` effort, independent evidence goals, routed hybrid retrieval, batched goal reranking, and explicit trace telemetry.

### Elastic

Elastic recommends hybrid full-text + vector search, commonly combined with Reciprocal Rank Fusion (RRF), followed by semantic reranking on a relatively small candidate set. Expensive reranking belongs after recall-oriented first-stage retrieval rather than over the full corpus.

Relevant public documentation:
- https://www.elastic.co/docs/solutions/search/hybrid-search
- https://www.elastic.co/docs/solutions/search/ranking

**IMS adaptation:** existing dense + lexical + exact/section lanes remain, but 0.9 reduces the search space through corpus/document routing before the cross-encoder and batches independent rerank groups.

### Glean

Glean's public enterprise-search material explicitly highlights organisation-specific terminology, acronyms, synonyms and a knowledge graph over enterprise content/interactions.

Relevant public documentation:
- https://docs.glean.com/administration/search/about

**IMS adaptation:** a local `retrieval_nodes` hierarchy learns document titles, section headings, defined terms and technical vocabulary from the already-ingested corpus. These are search hints, not answer facts.

### Microsoft GraphRAG

GraphRAG documents different query modes rather than one retrieval strategy for everything: Basic, Local, Global and DRIFT. Global search uses map/reduce over corpus-level community reports for holistic dataset questions, while Local combines graph/entity context with source text.

Relevant public documentation:
- https://microsoft.github.io/graphrag/query/overview/
- https://microsoft.github.io/graphrag/query/global_search/

**IMS adaptation:** ordinary Q&A remains hybrid/hierarchical because a full graph pipeline would add unnecessary ingestion and runtime cost. Truly corpus-wide report analysis continues to use IMS Deep Analysis, which already performs parallel section analysis and hierarchical reduce/synthesis. 0.9 adds a lightweight hierarchy/vocabulary layer for Q&A instead of forcing GraphRAG onto every lookup.

## Architecture

### 1. Corpus-intelligence index

`retrieval_nodes` is a secondary, rebuildable index derived from existing chunks:

```text
document node
  title / filename / family / authority / revision / section inventory

section node
  section_path + bounded section text + pages + ordinal range

concept node
  high-signal heading / defined term / technical corpus phrase
```

Each node has BGE-M3 embedding + PostgreSQL FTS. Dense and lexical node rankings are fused to return:

- corpus terminology useful for query formulation;
- likely documents;
- likely source families/revisions;
- likely governing sections.

The original chunks remain the evidence source. A retrieval node can guide search but is never cited as factual answer evidence.

### 2. Adaptive effort router

```text
FAST
  bounded definition/fact/lookup
  small routed document set
  8-unit draft budget

FOCUSED
  procedure / condition / relationship
  larger routed document/family scope
  16-unit draft budget

RESEARCH
  enumeration / all-supported-variants / multi-entity / comparison /
  calculation / decomposed multi-goal / explicit Research
  broad coverage with 28-unit draft budget
```

The budgets are defaults and remain capped by overall compositional safety settings.

### 3. Priority source becomes a probe, not a second full research job

Configured governing sources are still checked first, but 0.9 uses a much smaller `priority_probe` profile. If that evidence is sufficient, Auto stops. If not, routed source-family retrieval continues. Priority-source AI repair is disabled by default because corpus routing should happen before spending a second expensive pass on the same source.

### 4. Corpus-language query planning

Semantic planning receives corpus-index hints such as relevant headings/terms/documents. The prompt explicitly labels them **SEARCH HINTS, not facts**. This lets a user's natural language map toward terminology actually used in the organisation without hardcoding domain synonyms in production code.

### 5. Hybrid retrieval remains the factual recall layer

Chunk retrieval still combines:

- dense BGE-M3 retrieval;
- precise PostgreSQL FTS;
- recall-oriented relaxed lexical retrieval;
- exact/identifier lookup;
- section navigation;
- table retrieval when structurally relevant;
- procedure/role/entity coverage lanes where the information need requires them.

Corpus intelligence narrows/routs; it does not replace these evidence-bearing lanes.

### 6. Batched goal-local reranking

A compositional question may have several independent evidence goals. 0.8 reranked each branch separately and paid serial inference scheduling. 0.9 adds `/rerank/batch`: query/candidate pairs from independent goals are length-sorted together and evaluated in one inference scheduling window. The API client falls back to legacy serial `/rerank` during a rolling deployment if the batch endpoint is not available yet.

### 7. Monotonic evidence ledger

Later retrieval cannot silently overwrite earlier semantic support.

The merge order is:

1. evidence explicitly cited by the prior supported/partial audit;
2. newly recovered evidence targeted at unresolved goals;
3. remaining earlier evidence;
4. optional new context.

Primary evidence IDs remain stable. New recovery evidence receives new IDs above the current ledger maximum.

### 8. Semantic-drift guard

Retrieval repair may hypothesize different document vocabulary, but it must preserve material qualifiers and at least one explicit entity anchor for the goal. New numeric thresholds and technical identifiers are rejected unless the technical identifier is present in trusted corpus hints/snippets.

This prevents a question about condition `A + B` from silently turning into an easier but different rule `A + C` merely because `A + C` retrieves better passages.

### 9. Dynamic generation context

The final answer model no longer depends on a universal "four chunks per goal" concept. It receives a goal-balanced, document-diverse evidence context whose size depends on retrieval effort. Semantic-audit evidence is protected first.

The query log now records:

- `draft_context_count`
- `draft_context_chars`
- `draft_context_token_estimate`
- `draft_context_evidence_ids`
- `draft_context_chunk_ids`
- `draft_context_document_ids`
- per-goal evidence IDs

This makes "retrieval found it but drafting never saw it" directly diagnosable.

## Why 0.9 does not send more and more chunks

Enterprise retrieval quality is primarily a **selection** problem, not a context-window maximisation problem. More irrelevant context increases reranking cost, prompt tokens and conflict risk. The target is:

```text
large corpus
  -> high-recall first stage
  -> small semantically strong candidate pool
  -> rerank
  -> coherent parent/section expansion
  -> enough diverse evidence for the information need
  -> final LLM
```

Simple facts should terminate early. Enumeration/research workloads receive wider coverage.

## Multi-document synthesis rules

- Each required evidence goal is independently represented and ranked.
- Document/family diversity is preserved in the generation context.
- A relationship is not inferred from two unrelated definitions.
- Different line/system/revision scopes remain attached to their claims.
- Later evidence can contradict or supersede earlier evidence only when the answer explicitly explains the conflict/scope; it cannot silently replace earlier support.
- A corpus-negative conclusion requires a coverage-capable route; top-k failure is not proof of absence.

## Operational compatibility

- existing PDFs: unchanged
- existing parsed files: unchanged
- existing chunks: unchanged
- existing chunk embeddings: unchanged
- BGE-M3 model/dimension: unchanged
- reranker model: unchanged
- PostgreSQL receives one additive `retrieval_nodes` table
- the secondary index is built from existing chunks using the isolated ingestion inference plane
- if the secondary index is missing or fails, query-time discovery falls back safely to ordinary chunk retrieval

## What 0.9 intentionally does not do

- It does not train Learning-to-Rank from sparse feedback. LTR is appropriate after enough labelled relevance/feedback data exists.
- It does not force a full GraphRAG knowledge graph over every PDF. Q&A gets a lighter hierarchy; Deep Analysis already supplies map/reduce for whole-corpus analytical workloads.
- It does not hardcode organization-specific synonyms or known regression answers.
- It does not claim that an architecture change guarantees every possible question. Acceptance is measured with unseen queries and stage-specific retrieval metrics.
