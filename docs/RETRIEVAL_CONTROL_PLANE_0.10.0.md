# IMS 0.10 retrieval/control-plane redesign

Status: **production foundation + implementation blueprint**  
Baseline reviewed: **IMS 0.9.1**  
Scope: query understanding, corpus retrieval, source authority, reranking, evidence applicability, latency, evaluation, rollout, and future structured evidence.

## 1. Executive decision

IMS should be optimized as an **institutional evidence retrieval system**, not as a chain of increasingly broad RAG fallbacks.

The user contract is intentionally forgiving: users may not know official terminology, document names, subsystem names, exact rules, or whether the premise embedded in their question is correct. The system must bridge ordinary language to corpus language while preserving the original question and the document security boundary.

The recommended architecture has six invariants:

1. **Q0 is immutable.** The exact user question always participates in retrieval. Rewrites, terminology resolution, acronym expansion, and typo candidates only add search hypotheses.
2. **Global recall has a safety lane.** Every ACL-accessible document remains eligible unless the user explicitly scopes the request to selected documents.
3. **Document routing is a boost, never a hidden filter.** Likely documents receive additional candidate budget; they do not replace the global corpus.
4. **Governing references are always considered and never terminate retrieval.** MRGR is the current governing source, but the mechanism is generic and metadata/configuration driven.
5. **Expensive ranking is bounded.** Cheap retrieval/fusion should produce a small diverse pool; the cross-encoder should not be repeatedly applied to hundreds of near-duplicate candidates.
6. **Evidence applicability is distinct from semantic relevance.** A highly relevant conditional rule can still be inapplicable to the user's entity, operating mode, failure state, location, revision, or time.

The 0.10 foundation patch implements the first control-plane invariants and the terminology/ranking foundation. It deliberately does **not** pretend that the full Fact/Claim sidecar, deterministic table-row atoms, generalized applicability resolver, OpenTelemetry rollout, or true parallel speculative LangGraph fan-out are already complete. Those are specified below as the next production phases.

---

## 2. What the 0.9.1 failure tells us

The failed `Max speed at which train can operate` query is diagnostically useful because the corpus contained a direct RS-10 table fact while IMS returned generic/conditional MRGR speed restrictions.

The trace shows a characteristic enterprise-RAG failure pattern:

- one basic fact produced hundreds of candidates;
- table retrieval was underrepresented relative to dense, relaxed lexical, and section lanes;
- MRGR received a special priority path;
- the evidence audit recognized that the general maximum operating speed was still missing;
- expensive reranking/recovery dominated latency;
- answer generation nevertheless proceeded from incomplete evidence.

This is not primarily an embedding-model failure. It is a **control-plane, candidate-allocation, structured-evidence, and applicability problem**.

A key performance observation is that the measured search stages were sub-second while the CPU cross-encoder consumed roughly a minute in the captured query. Therefore search-engine migration is not the first latency intervention.

---

## 3. Current-state architecture assessment

### 3.1 Ingestion and representation

Relevant files:

- `src/ike/ingestion/docling_pipeline.py`
- `src/ike/ingestion/processor.py`
- `src/ike/db/models.py`

Strengths to keep:

- Docling layout/table/OCR parsing.
- `HybridChunker` with `repeat_table_header=True`.
- hierarchical `section_path` metadata.
- page provenance.
- fail-closed behavior for partial Docling conversion.
- canonical Docling JSON persisted for source reconstruction.
- BGE-M3 chunk embeddings.

Current limitation:

- the primary answer index is still chunk-centric;
- a specification row, conditional rule, and prose explanation all compete as chunks;
- tables are marked with `content_kind="table"`, but individual rows are not durable first-class evidence objects;
- overlapping numeric facts do not yet have explicit machine-readable conditions/applicability.

### 3.2 Query planning

Relevant files:

- `src/ike/retrieval/query_plan.py`
- `src/ike/workflows/routing.py`
- `src/ike/workflows/evidence_planning.py`
- `src/ike/workflows/query_understanding.py`

0.9.1 already contains valuable generalized machinery: evidence goals, claim checks, conditions, relationships, coverage contracts, and recovery queries.

The risk is that deterministic routing still relies heavily on grammatical/keyword patterns. Such rules are acceptable for safe structural clues, but they must not evolve into a railway/HR/engineering ontology encoded in regular expressions.

### 3.3 Corpus intelligence

Relevant file:

- `src/ike/retrieval/corpus_intelligence.py`

This subsystem is a strong foundation. It already derives document/section nodes and high-signal concepts from headings, definitions, and acronym/full-form pairs without re-embedding entire documents.

0.10 extends this into a separate `corpus_terms` vocabulary index so misspellings and organization-specific wording can generate search hypotheses without becoming answer evidence.

### 3.4 Retrieval engine

Relevant files:

- `src/ike/retrieval/engine.py`
- `src/ike/retrieval/search_plan.py`
- `src/ike/retrieval/fusion.py`
- `src/ike/retrieval/applicability.py`

Existing useful lanes include:

- dense vector retrieval;
- PostgreSQL full-text retrieval;
- relaxed lexical retrieval;
- exact/phrase retrieval;
- table-chunk retrieval;
- section-navigation retrieval;
- role/entity/coverage helpers;
- RRF-style fusion;
- cross-encoder reranking.

The problem is not lack of retrieval techniques. It is candidate budgeting and orchestration. Before 0.10, likely-document routing could effectively narrow the search scope, and MRGR priority retrieval could become a gate. Both are unsafe in a corpus where users often do not know the correct document family.

### 3.5 Applicability

Relevant file:

- `src/ike/retrieval/applicability.py`

Current applicability scoring is conservative but narrow: primarily line and rolling-stock matching. It does not yet explicitly model operating mode, trigger condition, system/failure state, role, organizational scope, exception, effective date, revision, or authority relationship.

That gap is central to overlapping condition-dependent facts.

### 3.6 Workflow orchestration

Relevant file:

- `src/ike/workflows/qa_graph.py`

The 0.9.1 graph had a separate priority-source stage that could perform its own assessment/recovery and could route directly to answering. This creates both correctness and latency risk.

0.10 removes the priority graph node. Governing references are retrieved inside the broad retrieval operation and are additive only.

### 3.7 LLM controller outputs

Relevant file:

- `src/ike/services/llm.py`

0.9.1 primarily used generic JSON-object generation followed by `json.loads`. That ensures JSON syntax, not domain-schema validity.

0.10 adds `generate_structured()` with a Pydantic schema contract for controller outputs. Existing generic JSON generation remains for staged migration, not as the intended long-term control-plane contract.

### 3.8 Security / ACL

Relevant file:

- `src/ike/retrieval/access.py`

This is a critical strength. The redesign keeps ACL filtering inside retrieval. Broader global retrieval must never mean broader authorization.

---

## 4. Target runtime architecture

```text
                                USER QUESTION
                                      |
                          deterministic safe parse
                                      |
                 +--------------------+--------------------+
                 |                                         |
        ORIGINAL QUERY SEARCH                        QUERY FRAME
        starts with exact Q0                         understanding
        lexical / dense / exact                     entities / relation
        table/fact candidates                       qualifiers / premise
                 |                                  alternate hypotheses
                 |                                         |
                 +--------------------+--------------------+
                                      |
          +---------------------------+---------------------------+
          |                           |                           |
   GLOBAL ACL CORPUS             ROUTED BOOST              GOVERNING LANE
   always eligible              likely documents           MRGR / future
   unless explicit scope        extra budget only          bounded/additive
          |                           |                           |
          +---------------------------+---------------------------+
                                      |
                       per-lane candidate reservation
                                      |
                                  RRF/fusion
                                      |
                        one bounded primary reranker
                                      |
                        applicability / conflict pass
                                      |
                   parent section/table context expansion
                                      |
                            evidence grouping
                                      |
                               answer synthesis
                                      |
                     risk-based verification only
```

For complex requests, query understanding may create 2-6 atomic evidence goals. Those goals should ultimately run in parallel using independent database sessions and merge into one evidence set. The 0.10 foundation does not yet enable this true speculative fan-out; the feature flag is intentionally defaulted to `false` until concurrency/session safety is implemented and measured.

---

## 5. What 0.10 implements now

### 5.1 Q0 preservation

`RetrievalEngine.retrieve()` explicitly adds the original question first:

```python
add_raw_query(question, origin="original")
```

Semantic, lexical-plan, acronym/identifier, corpus vocabulary, and expansion queries are additive.

### 5.2 QueryFrame foundation

New file:

- `src/ike/workflows/query_frame.py`

The frame is domain-neutral and includes:

- answer shape;
- entities;
- relations;
- qualifiers;
- explicit scope;
- unverified premise claims;
- canonical terms;
- alternate phrasings;
- acronym expansions;
- typo candidates;
- ambiguity;
- cross-document reasoning flag.

The deterministic path intentionally avoids inventing domain terms. Confirmation-style user premises are marked `unverified`, not assumed true.

### 5.3 Corpus-aware terminology

New file:

- `src/ike/retrieval/vocabulary.py`

Extended file:

- `src/ike/retrieval/corpus_intelligence.py`

New database table:

- `corpus_terms`

Vocabulary is derived from trusted corpus structure such as titles, family keys, headings, definitions, and acronym/full-form pairs. PostgreSQL `pg_trgm` is used to retrieve spelling/terminology candidates.

Important safety behavior:

- uppercase/identifier-like tokens containing technical codes, digits, slash notation, etc. are excluded from generic fuzzy mutation;
- vocabulary candidates are search hints only;
- exact Q0 remains present;
- answer generation never treats a vocabulary hit as factual evidence.

### 5.4 Routing as soft relevance prior

`qa_graph.py` now uses:

```text
explicit document IDs -> hard scope
routed document IDs   -> boost lane
global corpus         -> remains eligible
```

The former assumption that a complete routing index makes the router safe enough to hard-filter documents is removed from query control. Index completeness is useful observability; it is not proof that document classification recall is perfect.

### 5.5 Governing-reference lane

`RetrievalEngine._governing_document_ids()` resolves governing sources by:

1. explicit `Document.source_role` such as `governing_reference` / `governing_rule`; or
2. administrator-configured filename/title patterns as a compatibility fallback.

MRGR therefore remains mandatory-to-consider without becoming a gate.

### 5.6 Source authority metadata

`Document` now adds:

- `source_role`
- `authority_level`
- `supersedes_document_id`

Existing metadata such as family, revision, department, `effective_from`, and `effective_to` remains available.

The upload API accepts the new authority fields. Existing managed documents can also be updated through `PATCH /documents/{document_id}/authority`, so administrators can assign `source_role`, `authority_level`, and `supersedes_document_id` without reprocessing the document. `source_role` is normalized into a stable slug. `authority_level` is validated both at the API and database constraint layer.

### 5.7 Candidate lane reservations

Before the bounded cross-encoder pool is finalized, retrieval reserves capacity for important evidence lanes such as:

- table;
- exact;
- governing;
- routed boost;
- lexical;
- dense.

The exact numbers remain configuration knobs and must be tuned by evaluation. The invariant is that a noisy lane cannot consume the complete rerank budget.

### 5.8 Table-query activation for generic scalar questions

`search_plan.py` now recognizes generic scalar/superlative information needs rather than requiring a domain word such as `speed`.

This improves table participation for questions such as maximum/minimum/rated/capacity/dimension/value lookups without encoding departmental knowledge.

### 5.9 Adaptive planner behavior

Simple scalar lookup questions no longer require an LLM-planning call merely because they use ordinary language. Ambiguous free-form multi-concept queries and non-English natural-language questions may still use semantic planning so users are not forced to know official terminology.

When a semantic evidence plan requests broader retrieval but the query does not justify full multi-goal research, the effort classifier now chooses a **focused** path rather than silently collapsing the planner signal into a fast direct route.

### 5.10 Strict controller output foundation

`LLMClient.generate_structured()` uses SDK/Pydantic parsing for typed outputs. Evidence-planning and goal-audit payloads now have strict Pydantic transport models.

### 5.11 CPU isolation hook

`docker-compose.yml` introduces a separate `QUERY_CPUSET` for `inference-query`, and moves report generation to background CPUs by default.

The existing topology allowed PostgreSQL, APIs, Valkey, report work, and a six-thread cross-encoder to share the same online CPU set. On an 8-vCPU host this can create severe contention even when SQL retrieval itself is fast.

The default remains backwards compatible. A separate benchmark profile is supplied under `deploy/env/0.10-query-benchmark.env`.

---

## 6. Retrieval algorithm after the foundation patch

For an ordinary query:

1. Parse explicit scope and deterministic query structure.
2. Preserve exact Q0.
3. Retrieve corpus-intelligence hints and corpus-term fuzzy candidates.
4. Build additive search hypotheses.
5. Resolve likely documents as boost targets only.
6. Resolve governing-reference documents under the same ACL.
7. Embed bounded query formulations.
8. Run global dense + lexical + exact + table/section lanes.
9. Run bounded routed-document boost lanes.
10. Run bounded governing lanes.
11. Fuse ranked lists with RRF/weights.
12. Reserve candidate capacity per important lane.
13. Apply the primary cross-encoder to a bounded candidate pool.
14. Build evidence with governing/table/exact/coverage protections.
15. Assess required goals when the question is complex or risky.
16. Run at most one bounded recovery set for ordinary focused failures; deeper exploration is reserved for Research.
17. Generate an evidence-bound answer and risk-based verification.

The desired next optimization is to run independent cheap retrieval branches concurrently. It should not be implemented by sharing the same SQLAlchemy `Session` across threads/tasks because sessions are not thread-safe. Use separate read-only sessions per fan-out branch and merge by IDs/results.

---

## 7. QueryFrame production design

The final QueryFrame should remain a *search-control object*, not a source of facts.

Recommended schema:

```text
original: string                    # immutable Q0
answer_shape: enum                  # fact/scalar/definition/procedure/...
entities: [string]
relations: [string]
qualifiers: [{dimension, value}]
explicit_scope: [string]
premise_claims: [{text, status}]
canonical_terms: [string]
alternate_phrasings: [string]
acronym_expansions: [string]
typo_candidates: [string]
ambiguity: [string]
requires_cross_document_reasoning: bool
```

Rules:

- every generated string is bounded;
- unknown fields are rejected;
- Q0 cannot be omitted or replaced;
- generated identifiers/numbers not anchored in user/corpus evidence are rejected;
- fuzzy candidates cannot become facts;
- if semantic parsing fails, deterministic retrieval still works;
- if several interpretations remain materially different after retrieval, clarification is allowed as a final step.

The semantic QueryFrame and EvidencePlan should eventually be produced in one controller call for complex queries to avoid paying for two independent planner calls.

---

## 8. Fact/Claim sidecar: next schema

This should be implemented after the control-plane/evaluation baseline stabilizes.

Recommended table: `evidence_claims`

```text
id UUID PK
workspace/document ACL inherited through document_id
claim_type                    # table_row / scalar / normative / relation / condition
subject_text
relation_text
value_text
numeric_value nullable
unit nullable
qualifiers JSONB
conditions JSONB
exceptions JSONB
scope JSONB
modality nullable             # shall/must/may/specification/etc.
source_role nullable
authority_level nullable
document_id FK
chunk_id FK nullable
page_from/page_to
section_path text[]
revision/effective dates copied or joined
source_text                   # exact local evidence fragment
embedding optional
search_vector generated
created_at
```

This is not a knowledge graph ontology. It is a normalized evidence sidecar with provenance.

### Why this matters

Values such as 85, 25, 40, and 20 km/h should not be treated as competing numbers. They can coexist as claims with different conditions:

```text
85 km/h -> RS-10 / normal operational specification
25 km/h -> generic degraded RM/ROS condition
20 km/h -> platform-door abnormal condition
```

The applicability resolver can then determine whether evidence is direct, conditional, general, complementary, or conflicting.

---

## 9. Deterministic table-row evidence: next ingestion step

Docling already preserves the document/table structure and the current pipeline writes canonical JSON. The next ingestion version should create table-row evidence objects deterministically rather than asking an LLM to interpret tables during query time.

Recommended flow:

```text
Docling table
   -> identify column headers / row cells
   -> row text = parent headings + header:value pairs
   -> EvidenceClaim(type=table_row)
   -> BM25/FTS + optional embedding
   -> retain table reference/page/provenance
```

Retrieval should rank the small row object, then load the parent table/page/section for answer context.

For complex merged cells, prefer the structured Docling table JSON rather than relying only on Markdown serialization. DataFrame export is useful for simple row extraction but must not discard merged-cell semantics/provenance.

Existing PDFs do **not** need to be OCRed again if their canonical Docling JSON is still available. A backfill job can derive table rows from canonical JSON; only documents missing usable canonical structure should require reprocessing.

---

## 10. Applicability and conflict resolver: next production component

Semantic score answers **relevance**. It does not answer **applicability**.

Add a typed `ApplicabilityAssessment` over the final 8-16 evidence items. Generic dimensions should include:

- entity/equipment/system/subsystem;
- department/organizational scope;
- line/location/station;
- operating mode;
- system/failure state;
- trigger condition;
- user role;
- effective/revision date;
- document family;
- source role/authority;
- exception/override relationship.

Output labels:

```text
directly_applicable
conditionally_applicable
general_governing_information
complementary
not_applicable
genuine_conflict
```

Resolution principles:

1. Never infer authority precedence solely from embedding/reranker score.
2. Prefer explicit metadata (`supersedes`, effective dates, source role, authority level) when known.
3. Specific conditions may narrow a general rule without contradicting it.
4. If two current documents truly conflict under the same scope/condition and metadata cannot resolve precedence, surface the conflict instead of inventing a winner.
5. MRGR can remain governing background even when a technical/manual source supplies the detailed answer.

---

## 11. Lexical retrieval evolution

Keep the current technical lexical lane because `simple` PostgreSQL FTS is valuable for identifiers and exact technical vocabulary.

Add a second natural-language lexical/BM25 lane after benchmark evidence shows its value. Options:

1. PostgreSQL-native FTS with a second language/stemming configuration where appropriate;
2. PostgreSQL + an extension that provides BM25 semantics, subject to licensing review;
3. external OpenSearch/Qdrant/Vespa challenger behind a `SearchBackend` interface.

Do not remove technical lexical retrieval when adding stemming. Technical identifiers and ordinary prose have different retrieval behavior.

---

## 12. SearchBackend abstraction

Do not refactor the whole engine into a new backend during the same release as the control-plane fix. Stabilize metrics first, then introduce:

```python
class SearchBackend(Protocol):
    def lexical(...): ...
    def dense(...): ...
    def fuzzy(...): ...
    def structured(...): ...
```

`PostgresSearchBackend` becomes the baseline adapter. Challenger adapters must preserve:

- ACL filters;
- explicit document scope;
- effective/lifecycle filters;
- document/page/section provenance;
- stable IDs;
- per-lane diagnostics.

The orchestration/fusion/evidence layer should not know which search engine produced a ranked list.

---

## 13. Technology decision

### Keep PostgreSQL/pgvector now

Reason:

- current measured dense/lexical/exact/table search stages are already small compared with reranking/recovery;
- ACL and document metadata are already co-located;
- migration risk would be high while the main correctness problems are control-plane and evidence representation;
- pgvector 0.8+ supports iterative HNSW scans for filtered recall;
- `pg_trgm` supplies indexed corpus-term similarity.

### Qdrant - first vector/multi-stage challenger

Strengths:

- strong multi-stage query API;
- hybrid/vector prefetch flows;
- natural support for dense candidate retrieval followed by multi-vector/ColBERT refinement;
- Apache-2.0 open-source core.

Evaluate if learned sparse / multi-vector retrieval materially raises gold evidence recall or reduces total inference cost.

### OpenSearch - first enterprise text-search challenger

Strengths:

- mature BM25/inverted indexing;
- filters/facets;
- hybrid retrieval/RRF;
- neural sparse options;
- Apache-2.0 core.

Evaluate if the corpus grows enough that advanced lexical search, filtering, distributed scale, and search operations justify another service.

### Vespa - advanced ranking challenger

Strengths:

- explicit cheap first-phase / bounded expensive second/global-phase ranking;
- excellent control over ranking features and learned rankers;
- strong fit if IMS eventually needs sophisticated learning-to-rank at large scale.

Trade-off: greater operational and relevance-engineering complexity.

### ParadeDB / BM25-in-Postgres options

Architecturally attractive because lexical/vector/metadata can stay near PostgreSQL. Current licensing must be reviewed carefully before organization adoption; do not introduce AGPL/commercial obligations accidentally.

### Milvus

Technically capable, but do not add it unless a benchmark shows material benefit versus the simpler candidates for this workload.

**Decision gate:** no engine migration until a shadow benchmark on the same corpus proves a meaningful gain in gold-evidence recall/quality and/or p95 latency after including operational complexity and ACL behavior.

---

## 14. BGE-M3 decision

Keep `BAAI/bge-m3` initially.

The current failure does not prove the dense model is weak. First benchmark:

1. existing dense + improved lexical/table/fact lanes;
2. dense + true BM25;
3. BGE-M3 learned sparse if the chosen inference/search stack supports it cleanly;
4. optional late-interaction/ColBERT only if the evaluation set shows a meaningful gain.

Do not add retrieval stages because they are fashionable. Each stage must earn its latency/operational cost.

---

## 15. Reranker performance plan

The captured trace makes this the immediate optimization target.

Candidate experiments supplied in `deploy/env/0.10-query-benchmark.env`:

```text
A  Torch     1024 tokens  40 candidates
B  OpenVINO  1024 tokens  40 candidates
C  OpenVINO   512 tokens  32 candidates
D  OpenVINO   512 tokens  24 candidates
```

The supplied benchmark profile starts near D but is explicitly **not** a production default until the gold set is run.

Select by:

- `GoldEvidenceRecall@rerank_pool`;
- `GoldEvidenceRecall@8/12`;
- nDCG/MRR;
- citation support;
- p50/p95 rerank execution;
- throughput at realistic concurrent query load.

### CPU topology

For an 8-vCPU host, benchmark isolation such as:

```text
ONLINE_CPUSET=0-1       # PostgreSQL/API/Valkey/nginx
QUERY_CPUSET=2-5        # query inference / reranker
BACKGROUND_CPUSET=6-7   # ingestion/report work
QUERY_ML_NUM_THREADS=4
```

This is a benchmark hypothesis, not a universal recommendation. A six-thread model sharing the same six cores with PostgreSQL can have worse wall-clock and tail latency than a four-thread model on isolated cores.

---

## 16. Candidate service-level objectives

These are acceptance targets to validate on the actual VM, not guarantees:

| Stage/query class | Candidate target |
|---|---:|
| First-stage indexed retrieval p95 | < 1.5 s |
| Primary reranker p95 | < 2.5 s CPU after optimization |
| Simple/fact end-to-end p50 | < 5 s |
| Simple/fact end-to-end p95 | < 10 s |
| Focused conditional/procedure p95 | < 15 s |
| Research/multi-doc p95 | < 30 s, with progress streaming |

If model-provider latency makes end-to-end targets impossible, measure retrieval/ranking and generation separately. Do not hide a 60-second reranker inside a single total number.

---

## 17. Recommended environment policy

### Safe foundation defaults

Keep initially:

```text
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_MAX_SEQ_LENGTH=1024
CHUNK_MAX_TOKENS=500
DOCLING_OCR=true
DOCLING_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate
HNSW_ITERATIVE_SCAN_MODE=strict_order
SPECULATIVE_RETRIEVAL_ENABLED=false
```

The exact original query/global ACL corpus behavior is a code invariant, not a switch that operations should accidentally disable.

### Benchmark profile

Evaluate:

```text
INSTALL_OPTIMIZED_RERANK=1
RERANK_BACKEND=openvino
RERANK_MAX_LENGTH=512
RERANK_PREFILTER_MAX_CANDIDATES=24
RERANK_TOP_K=12
DIRECT_RERANK_TOP_K=8
```

plus the optional CPU partition described above.

### New retrieval controls

The 0.10 foundation adds controls for:

- routed-document boost budget;
- governing-reference patterns/budgets;
- corpus terminology resolution;
- candidate lane reservations;
- bounded recovery query count;
- HNSW iterative scan mode.

Do not treat their initial values as truth. Tune against the evaluation set.

### Legacy priority settings

Do not simply disable old priority retrieval on an unpatched 0.9.1 deployment. Apply the code change first so the governing lane exists and global retrieval remains active. In 0.10 the old priority environment variables remain accepted only for rollback/config compatibility; the graph no longer runs the old priority short-circuit stage.

---

## 18. Evaluation framework

The golden set is part of the product architecture.

### Required query categories

- exact source wording;
- paraphrase;
- typo/noisy spelling;
- colloquial/informal staff language;
- acronym/identifier;
- wrong premise;
- underspecified entity;
- table/scalar fact;
- condition-sensitive fact;
- governing + detailed evidence;
- one-source fact;
- multi-document synthesis;
- conflicting revisions;
- enumeration/completeness;
- negative/prohibition request;
- follow-up query;
- non-English/mixed-language query;
- cross-department terminology collision;
- ACL isolation.

### Retrieval metrics

Record lineage, not only final answer quality:

```text
Gold evidence exists
 -> GlobalCandidateRecall@K
 -> per-lane recall
 -> routed-lane recall (diagnostic only)
 -> governing-source recall
 -> RerankPoolRecall
 -> RerankRecall@8/@12
 -> DraftContextRecall
 -> CitationSupport
```

Also record:

- MRR;
- nDCG;
- correct-document recall;
- condition preservation;
- false governing-source inclusion rate;
- contradiction detection;
- citation precision/support;
- answer completeness;
- abstention/clarification correctness;
- p50/p95 per stage;
- token use and model calls.

### Release gate

No ranking/control-plane change should ship based on one visible failure. Require:

1. no material regression in evidence recall;
2. no ACL regression;
3. measurable p95 improvement for claimed performance work;
4. category-level reports so gains in easy facts cannot hide losses in procedures/conditions;
5. reviewed regressions converted into durable gold cases.

---

## 19. Observability plan

The current JSON trace is valuable for development but too large as the primary production diagnostic.

Add OpenTelemetry spans:

```text
qa.request
  query.frame
  corpus.discovery
  retrieve.global_dense
  retrieve.global_lexical
  retrieve.fuzzy
  retrieve.governing
  retrieve.structured
  retrieve.routed_boost
  fusion
  rerank
  applicability
  context_expand
  verify
  generate
```

Record numeric/identifier metadata only:

- candidate count;
- top-k/budget;
- execution/queue time;
- cache hit;
- inference backend/model version;
- document count;
- retrieval lane;
- top score/rank distributions;
- request/query ID.

Do not emit source document text into general telemetry.

---

## 20. Caching plan

Recommended caches:

- normalized query -> query embedding;
- typo/variant -> corpus-term candidates;
- query concepts -> governing evidence IDs, invalidated on governing-source revision/reingestion;
- query hash + chunk checksum + reranker version -> rerank score for evaluation/repeated queries.

Do not use final-answer caching as the primary optimization. ACLs, effective dates, document revisions, and user context make it much harder to invalidate safely.

---

## 21. Migration and rollout

### Database

Apply Alembic migration:

```text
0005_retrieval_control_foundation
```

It is additive:

- three document authority columns;
- `corpus_terms` table;
- indexes including trigram GIN.

No chunk/embedding rewrite is required for the foundation release.

### Backfill corpus intelligence

After migration/code deployment:

```bash
python scripts/build_retrieval_intelligence.py
```

`INDEX_VERSION=0.10.0` causes existing ready documents with older intelligence metadata to be rebuilt from existing chunk embeddings. This should not re-OCR or re-embed source PDFs.

Run in bounded batches if desired:

```bash
python scripts/build_retrieval_intelligence.py --limit 100
```

Repeat until `pending_documents=0`.

### Shadow/canary rollout

Recommended sequence:

1. freeze 0.9.1 image/config and export a representative gold set;
2. apply database migration (additive);
3. deploy 0.10 to a canary environment;
4. backfill corpus terms/intelligence;
5. run the exact same gold set on 0.9.1 and 0.10;
6. benchmark Torch vs OpenVINO profiles;
7. shadow selected real queries if policy permits, storing only IDs/metrics needed for comparison;
8. canary a small user group;
9. promote only after ACL, retrieval recall, citation, and latency gates pass.

### Rollback

The new schema is additive. A safe application rollback can leave the new columns/table in place while 0.9.1 binaries ignore them. Avoid downgrading the database during an emergency application rollback unless there is a specific reason to remove the additive schema.

---

## 22. Next implementation phases

### Phase A - finish control-plane + latency (highest priority)

- benchmark the 0.10 foundation against the real corpus;
- optimize/replace the current CPU reranker profile;
- ensure primary direct/focused retrieval performs one bounded cross-encoder pass;
- remove remaining repeated rerank work from ordinary recovery paths;
- add per-stage metrics.

### Phase B - true speculative fan-out/fan-in

- split read-only retrieval branches onto independent SQLAlchemy sessions;
- start Q0 retrieval immediately;
- run query understanding concurrently;
- add only novel hypotheses after the frame returns;
- merge/fuse once;
- keep LLM planning off the latency-critical path for straightforward lookups.

### Phase C - table-row evidence

- derive row atoms from canonical Docling table structure;
- backfill from canonical JSON where possible;
- index row text with headers + parent section;
- retrieve row first, expand parent context later.

### Phase D - Fact/Claim + applicability

- add `evidence_claims` migration/model;
- deterministic table/scalar extraction first;
- asynchronous bounded claim extraction from high-value normative prose later;
- build typed applicability/conflict resolver;
- never let LLM extraction overwrite source text/provenance.

### Phase E - natural-language BM25 / backend abstraction

- add `SearchBackend` protocol;
- implement Postgres baseline adapter;
- run shadow adapters for Qdrant/OpenSearch/Vespa or BM25-in-Postgres alternatives;
- decide by metrics, licensing, and operations.

### Phase F - learning to rank

Only after sufficient SME/user relevance labels exist. Candidate features include BM25 rank, dense score, fuzzy/exact match, table/fact lane, heading overlap, document routing score, source role, entity/condition compatibility, revision status, and document diversity.

---

## 23. Exact 0.10 foundation file changes

### New files

- `src/ike/workflows/query_frame.py`
- `src/ike/retrieval/vocabulary.py`
- `migrations/versions/0005_retrieval_control_foundation.py`
- `deploy/env/0.10-query-benchmark.env`
- `docs/RETRIEVAL_CONTROL_PLANE_0.10.0.md`
- `docs/UPGRADE_0.10.0.md`

### Modified production files

- `.env.example`
- `docker-compose.yml`
- `pyproject.toml`
- `src/ike/__init__.py`
- `src/ike/core/config.py`
- `src/ike/db/models.py`
- `src/ike/api/routes/documents.py`
- `src/ike/schemas/documents.py`
- `src/ike/retrieval/corpus_intelligence.py`
- `src/ike/retrieval/engine.py`
- `src/ike/retrieval/search_plan.py`
- `src/ike/services/llm.py`
- `src/ike/services/query_execution.py`
- `src/ike/workflows/evidence_planning.py`
- `src/ike/workflows/query_understanding.py`
- `src/ike/workflows/qa_graph.py`
- `scripts/build_retrieval_intelligence.py`
- `tests/test_080_retrieval_control_plane.py`

---

## 24. Validation performed on the patch

The modified Python tree compiles successfully with `compileall` in the review environment.

The following focused regression set passed:

```text
63 tests passed
```

Covering:

- retrieval control plane;
- industrial retrieval contracts;
- compositional retrieval;
- entity/attribute retrieval;
- query complexity.

New tests explicitly cover:

- governing references cannot short-circuit global retrieval;
- routed documents are boosts rather than hidden hard scope;
- scalar lookup can avoid unnecessary semantic planning;
- exact Q0 precedes semantic hypotheses;
- generic superlative/scalar questions can activate the table lane;
- corpus fuzzy correction protects technical identifiers;
- confirmation-style premises remain unverified;
- speculative fan-out stays disabled until parallel-session safety is implemented.

The review container does not include the full production dependency/runtime stack or live PostgreSQL/pgvector corpus, so this is **not** a substitute for migration/integration/golden-set testing on the deployment environment.

---

## 25. Risks and trade-offs

### Relevance drift from terminology expansion

Mitigation: preserve Q0, exclude identifier-like tokens from generic fuzzy correction, treat term matches as search hints only, cap expansion count, measure drift rate.

### Global retrieval increases candidates

Mitigation: global eligibility does not mean large global top-k. Use indexed search, routed boosts, RRF, lane reservations, and a bounded rerank pool.

### Governing content may appear too often

Mitigation: always **search** governing sources but reserve only a small number of relevant evidence slots. Measure false-governing-inclusion rate.

### Structured claim extraction can introduce errors

Mitigation: deterministic table rows first; provenance is mandatory; extracted fields never replace source text; claims are evidence indexes, not source-of-truth edits.

### Authority metadata may be incomplete

Mitigation: unknown authority must not be invented. Prefer administrator-confirmed metadata for important source families; surface unresolved conflicts.

### More parallelism can overload CPU/DB

Mitigation: independent bounded branches, admission control, inference semaphores, CPU isolation, per-stage tracing, load tests.

### Search-engine migration can add operational risk

Mitigation: backend abstraction + shadow benchmark. No migration on feature checklists alone.

### ACL leakage

Mitigation: every retrieval backend/lane must enforce authorization before evidence leaves the data layer. Add ACL-specific golden and adversarial tests.

---

## 26. Industry-practice alignment

The redesign is intentionally consistent with public production-search patterns rather than a bespoke agent architecture:

- Azure AI Search documents minimal retrieval effort that skips LLM planning for fast requests, and low/medium effort that decomposes complex requests into parallel subqueries.
- Azure hybrid retrieval executes keyword and vector queries in parallel and merges them with RRF.
- Azure knowledge sources can be configured to always participate, which maps well to a mandatory governing-reference lane without making that source a short-circuit.
- Qdrant documents multi-stage retrieval where cheap candidates are prefetched and then refined by a more expensive representation such as ColBERT.
- Vespa explicitly separates cheap first-phase ranking from bounded expensive reranking.
- PostgreSQL `pg_trgm` supports indexed trigram similarity; pgvector 0.8+ supports iterative HNSW scanning and a `relaxed_order` option that can improve recall.
- Modern LLM APIs support schema-constrained structured outputs, reducing the risk of treating syntactically valid but semantically malformed JSON as a controller decision.

These patterns reinforce the same design principle: **cheap broad recall, bounded expensive reasoning, structured applicability, and measurable evidence lineage**.

---

## 27. Final recommendation

Do **not** replace IMS with GraphRAG, a different vector database, a larger embedding model, or more serial agents as the next move.

First make the retrieval contract correct:

```text
ordinary user language
 -> preserve Q0
 -> corpus-aware expansion
 -> global ACL retrieval
 + routed boost
 + governing lane
 + structured/table evidence
 -> bounded fusion/rerank
 -> applicability/revision/condition resolution
 -> evidence-grounded answer
```

Then benchmark the remaining bottlenecks and only introduce additional infrastructure when a reproducible gold set proves that it improves the product.

The foundation patch is deliberately generalized: it does not hard-code the speed example, RS-10, an HR rule, a station, or an expected answer. The same invariants apply to operations, HR, maintenance, electrical, signaling, rolling stock, finance, safety, contracts, and future document families.

## Public references reviewed

- Microsoft Learn - Agentic Retrieval Overview, Azure AI Search (retrieval effort, query decomposition, parallel subqueries).
- Microsoft Learn - Azure AI Search Features / Hybrid Search (keyword + vector parallel retrieval, RRF, synonym/spelling-aware query planning).
- Microsoft Learn - Knowledge Sources (always-query source behavior).
- PostgreSQL 17 documentation - `pg_trgm`.
- pgvector documentation - iterative HNSW scans and relaxed ordering.
- Qdrant documentation - Hybrid / Multi-Stage Queries.
- Vespa documentation - Phased Ranking.
- OpenAI API documentation - Structured Outputs / schema-constrained controller outputs.
- Docling documentation - structured table export and table serialization.

Review date: 2026-09-13.
