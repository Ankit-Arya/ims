# Evaluation strategy — 0.5.0

0.5.0 adds two explicit benchmark layers: `scripts/benchmark_inference.py` for same-reranker runtime comparison, and `scripts/load_test_qa.py` for paced heterogeneous Q&A load. Capacity must be measured on the current VM with one representative PDF processing, ingestion concurrency 1, and 100 varied Q&A requests over five minutes. Report p50/p95 by resolved mode plus inference queue/execution, CPU, RAM, DB utilization, ingestion progress, failures and restarts.

Golden evaluation must include vague wording, single/multiple acronym meanings, exact numbers, tables, procedures, broad duties, multi-document scenarios, exceptions, OCR-heavy pages, figures/diagrams and no-answer cases. Faster settings do not become defaults if retrieval/citation/completeness quality materially regresses.

---

## Historical / earlier-release notes


## Why evaluation is part of the architecture

A RAG system should not be improved by fixing one visible bad answer and assuming quality rose globally. Each failure can occur at a different layer:

```text
PDF extraction
  -> chunk construction
  -> dense/lexical/exact candidate recall
  -> fusion
  -> reranking
  -> evidence assembly
  -> generation
  -> citation compliance
  -> completeness
```

The evaluator and retrieval-debug endpoint are intended to localise the failure before code is changed.

## Build a golden set

Start with 100-300 questions from real documents, then grow it as users find failures. A mature corpus can justify 500-1000+ cases.

Cover at least:

- definition
- acronym/code
- exact number/unit
- table lookup
- procedure
- exception/condition
- multi-part question
- cross-document comparison
- authority/revision conflict
- OCR-heavy page
- ambiguous wording
- no-answer / abstention
- follow-up phrasing variants
- report-analysis objectives

For each factual Q&A case, record what you can objectively judge:

```json
{
  "id": "procedure-014",
  "question": "...",
  "expected_documents": ["Exact document title"],
  "expected_pages": [64, 65],
  "required_terms": ["critical phrase", "120 days"],
  "mode": "research"
}
```

Do **not** encode implementation-specific chunk IDs into the golden set; chunks should be free to improve over time.

## Run

```powershell
python scripts/evaluate.py eval/my_golden_set.jsonl --password "<admin-password>"
```

The script calls the analyst retrieval-debug endpoint and the normal answer endpoint separately.

## Metrics in the supplied runner

### Retrieval document hit

Did all expected document titles appear in the final evidence set?

### Retrieval page hit

Did at least one expected page appear in evidence?

For procedures spanning multiple pages, extend the evaluator to require a page set rather than a single-page intersection.

### Citation document hit

Did the answer actually cite the expected document(s), rather than merely retrieve them?

### Required terms

A coarse answer-content check for facts that must not disappear.

## Metrics to add for a serious benchmark

The supplied runner is intentionally simple and transparent. Extend it with:

- Recall@K at pre-rerank and post-rerank stages
- MRR / nDCG for ranked evidence
- required-fact coverage scoring
- citation entailment / claim-source validation
- answer abstention precision/recall
- p50/p95 latency
- input/output tokens per query class
- ingestion throughput/pages per minute on your hardware
- table extraction fidelity checks

## Regression workflow

For every meaningful change:

1. freeze the code/model/config revision
2. run the same golden set
3. compare retrieval metrics first
4. compare generation/citation metrics second
5. inspect regressions, not only average gains
6. reject improvements that fix a narrow example while materially reducing broader recall

## Failure triage

### Expected evidence absent from `/debug/retrieval`

Investigate extraction, chunking, exact/FTS/dense recall or ACL/effective-date metadata.

### Evidence appears but reranker omits it

Reranker/model/candidate-depth problem.

### Evidence appears in final debug evidence but not citations

Generation/completeness/citation problem.

### Correct citations but wrong wording/number

Generation or verification failure.

### Document unexpectedly inaccessible

ACL/lifecycle/effective-date metadata problem; do not weaken retrieval filters to compensate.

## User feedback

The UI records a coarse positive/negative rating. Use negative feedback to create **new golden cases** with expected evidence before changing prompts. This converts anecdotal failures into durable regression coverage.

## 0.2 mode-specific acceptance tests

Do not aggregate Direct and Research into one latency/quality number.

### Direct Q&A

Measure:

- p50/p95 end-to-end latency
- candidate search time
- reranker time
- answer-provider time
- citation accuracy
- focused fact accuracy

### Research

Measure:

- expansion quality
- expected-document/page recall after broadened retrieval
- completeness across sub-parts
- verifier correction rate
- token/runtime cost

### Acronym / definition completeness

For each acronym with multiple meanings, the golden item should contain every expected source-specific expansion. Example schema extension:

```json
{
  "question": "ABC",
  "mode": "direct",
  "expected_documents": ["manual-a.pdf", "manual-b.pdf"],
  "expected_pages": [12, 91],
  "must_include": ["Expansion One", "Expansion Two"]
}
```

A response that contains one correct meaning but silently omits another should count as **incomplete**, not correct.

### Independent-question contract

Add a regression test that asks an unrelated second question after a first question and confirms the API payload/workflow contains no previous text. UI history must not change retrieval inputs.

## Multi-user acceptance tests

Before organisation testing, add at least two non-admin accounts and verify:

1. both can view/download the same organisation-scope PDF;
2. User A's My Questions never appears for User B;
3. User A's Deep Analysis history never appears for User B;
4. a normal user cannot retry/reprocess/delete corpus documents;
5. an owning analyst can manage their own uploaded document;
6. a failed identical re-upload retries the existing ID rather than creating a second row.

## 0.4 workspace and UX acceptance

Add product-level acceptance cases in addition to retrieval quality:

- phone-width Ask layout does not overflow horizontally;
- the question composer is visible before previous history;
- starting a new question replaces the current request rather than creating a transcript on Ask;
- light/dark themes remain legible for cards, forms, tables, citations and modal dialogs;
- Deep Analysis is started from Ask and appears in My Questions;
- personal PDF owner can retrieve the document;
- explicitly shared user can retrieve it;
- unshared user and unshared admin cannot retrieve/view/download it through normal product routes;
- revoking a share removes future access;
- Q&A/report history remains isolated between accounts.

## 0.4 ingestion throughput benchmark

Do not benchmark only one easy PDF. Use a representative batch containing native-text PDFs, scans, tables, rotated content and large manuals. Record:

- documents/hour;
- pages/minute;
- peak worker RAM;
- CPU utilization;
- extraction failure rate;
- page/chunk coverage;
- table fidelity;
- OCR fidelity on known difficult pages.

Compare concurrency configurations only when extraction quality remains acceptable.

---

## 0.9.0 stage-specific evaluation

A final-answer score is insufficient for diagnosing enterprise retrieval. For every golden question where the expected evidence is known, record the following lineage:

```text
expected evidence exists
  -> found by corpus routing?              CorpusRouteRecall@K
  -> found by first-stage chunk retrieval? CandidateRecall@K
  -> survives reranking?                   RerankRecall@K
  -> retained in monotonic ledger?         EvidenceSurvival
  -> sent to drafting model?               DraftContextRecall
  -> cited/used correctly?                 AnswerSupport / CitationPrecision
```

### Required production metrics

- source/document/page/section Recall@K;
- required-goal recall;
- distinct-variant/list/procedure completeness;
- relationship and conditional correctness;
- source-routing precision/recall;
- recovery semantic-drift rate;
- evidence-survival rate across stages;
- draft-context recall of known gold evidence;
- citation support/precision;
- irrelevant-evidence admission rate;
- p50/p95 total latency and stage latency;
- input/output tokens;
- abstention precision (especially corpus-negative claims).

### Acceptance set

Use unseen, domain-diverse questions. Known historical failures remain regressions only; they must never be used as production-code rules. Include simple facts, terminology paraphrases, procedures, multi-document joins, complete lists/duties, multiple acronym meanings, conditionals, false premises, comparisons, revision/scope conflicts, calculations and true/false corpus-negative cases.

### Performance targets

Tune from measured production distributions, not a single query. Fast questions should demonstrate materially lower p50 than Focused/Research. Research may spend more time for coverage, but independent retrieval/rerank branches should not create avoidable serial latency.
