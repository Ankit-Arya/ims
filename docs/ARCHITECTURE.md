# IMS Architecture — Planned AI Research v5

This document describes the current Q&A runtime and where each responsibility lives.

## Source tree

```text
src/ike/
├── main.py
├── frontend/                 Browser UI only
├── api/                      HTTP/SSE boundary
├── services/
│   ├── query_execution.py    Single Q&A application entrypoint
│   ├── query_debug.py        Per-query forensic report
│   ├── llm.py                OpenAI structured-output wrapper
│   └── inference_client.py   Query embedding service client
├── agent/
│   ├── models.py             Research-plan / answer schemas
│   ├── prompts.py            Query-intelligence + answer contracts
│   ├── query_intelligence.py AI query understanding / research planning
│   ├── research.py           Bounded execution of the AI-created research graph
│   ├── tools.py              Fast access-safe retrieval primitives
│   ├── answer.py             Evidence reasoning + final answer agent
│   └── service.py            v5 orchestration
├── retrieval/
│   ├── search_engine.py      Dense / lexical / exact primitives
│   ├── fusion.py             Reciprocal-rank fusion
│   ├── evidence_selection.py Mechanical duplicate removal
│   ├── search_plan.py        Low-level token helpers
│   ├── table_context.py      Retrieval text representation
│   ├── corpus_intelligence.py
│   ├── access.py             ACL SQL predicates
│   └── types.py              Candidate / Evidence types
├── ingestion/                PDF parsing, OCR, chunking, indexing
├── reports/                  Deep Analysis workflow
├── tasks/
│   ├── celery_app.py
│   ├── ingestion.py
│   ├── reports.py
│   └── worker_bootstrap.py
├── db/
├── schemas/
└── core/
```

There is one normal Q&A runtime. The old deterministic QA graph, MCP controller loop,
coverage-saturation policies and mandatory cross-encoder reranker are not part of it.

## End-to-end flow

```text
User question
    │
    ▼
Query Intelligence Agent
    │
    ├─ interprets natural language / shorthand / acronyms
    ├─ identifies every answer requirement
    ├─ understands independent vs correlated vs dependent needs
    ├─ rewrites retrieval queries
    ├─ preserves exact identifiers
    └─ creates a minimal research graph
    │
    ▼
Research Executor
    │
    ├─ executes AI-created tasks within ACL/time bounds
    ├─ source_lookup
    ├─ search (lexical / semantic / hybrid chosen by AI)
    ├─ structure inspection
    └─ context inspection
    │
    ▼
Evidence + Answer Agent
    │
    ├─ decides which evidence is applicable
    ├─ reasons across multiple requirements and conditions
    ├─ detects conflicts / missing operational steps
    ├─ selects documentary evidence
    └─ either answers OR requests one precise gap round
    │
    ├──────────── sufficient ────────────► Final answer
    │
    └──── material gap
              │
              ▼
       Targeted gap research
              │
              ▼
       Final Answer Agent
              │
              ▼
         QueryLog / UI
```

Ordinary questions therefore require two AI decisions:

1. query intelligence / research plan;
2. evidence reasoning + final answer.

A third AI decision occurs only when the first evidence pass identifies a concrete
material gap.

## Semantic ownership

### AI owns

- what the user means;
- inferred and explicit conditions;
- query expansion / rephrasing;
- decomposition of multi-part questions;
- relationships between research requirements;
- retrieval mode (lexical / semantic / hybrid);
- literal terms that must be preserved;
- source-routing intent;
- whether evidence applies to the stated situation;
- whether a threshold alone is enough or surrounding procedure is needed;
- whether one targeted follow-up search is necessary;
- final evidence selection;
- final answer organization and wording.

### Python owns

- authentication and ACLs;
- database access;
- exact / FTS / vector retrieval execution;
- reciprocal-rank fusion;
- mechanical duplicate removal;
- context-window retrieval;
- execution of task dependencies created by the AI;
- hard research-time / evidence / token bounds;
- LLM request timeout / retries;
- citation bookkeeping;
- persistence, metrics and cancellation.

Python does not decide semantic completeness, source applicability or what the answer
should mean.

## Fast retrieval

A normal search task does only the retrieval modes requested by the planner.

### lexical

```text
PostgreSQL FTS
+ exact query
+ AI-selected exact phrases
        │
        ▼
       RRF
```

### semantic

```text
one query embedding
+ pgvector search
```

### hybrid

```text
FTS ────────┐
exact ──────┼─► RRF ► evidence candidates
dense ──────┘
```

There is no mandatory cross-encoder reranking and no hidden per-anchor lexical fan-out.

The final AI model performs the semantic relevance judgment over the returned evidence.

## Complex queries

The planner can produce multiple research tasks.

Example:

```text
"If A and B happened, while C was unavailable, what should X and Y do?"

T1: establish consequence of A
T2: establish consequence of B
T3: determine rule/procedure for unavailable C
T4: establish X responsibilities under A+B+C
T5: establish Y responsibilities under A+B+C
```

Tasks can carry `depends_on` relationships. Python executes that graph but does not
invent the relationships.

If a truly dependent search cannot be formulated until retrieved evidence reveals a
specific fact, the first evidence/answer pass creates the targeted gap task.

## Operational context expansion

`inspect_context` is intentionally cheap and available for procedure questions.

If search returns:

```text
3-6 BIC -> 10 km/h
```

but the user asks what must actually be done, the answer agent can request context around
that chunk rather than launch another broad corpus search.

This exposes nearby stop / bypass / mode / detrain / depot instructions in source order.

## Answer behavior

Answers should be:

- direct;
- complete for the requested conditions;
- grounded in selected documentary evidence;
- operationally useful when the question is operational;
- concise for simple facts and appropriately structured for complex cases.

The system does not append generic uncertainty boilerplate.

A missing detail is mentioned only when it materially changes the answer or action, and
the wording must identify the specific missing fact rather than advertise retrieval
weakness.

## Reliability / latency bounds

Current local defaults:

```text
primary research budget : 20 seconds
targeted gap budget     : 10 seconds
gap rounds              : 1
gap tasks                : max 3
search evidence          : max 14 default / 24 hard per task
final selected evidence : max 40
planner reasoning        : medium
answer reasoning         : medium
LLM request timeout      : 120 seconds
LLM retries              : 1
```

These are engineering ceilings, not semantic rules.

## Debugging a query

The downloadable query debug report now shows:

1. original question;
2. Query Intelligence interpretation;
3. answer requirements;
4. research tasks and dependencies;
5. each retrieval operation and elapsed time;
6. first evidence/answer decision;
7. targeted gap request, if any;
8. final evidence selection;
9. final answer and timings.

The key distinction is visible in the trace:

```text
AI decided what to research
        ↓
Python executed it
        ↓
AI decided what the evidence means
```
