# IMS 0.7.0 — Compositional Evidence Planning and Multi-Hop Retrieval

## Purpose

IMS 0.7.0 changes Q&A from a primarily **single-query / single-candidate-pool** model to a bounded **evidence-plan** model. A user request can now require several independently retrievable facts, attributes, definitions, conditions, comparisons, relationships, exceptions, outcomes, or claim checks before an answer is drafted.

This release is deliberately **domain-generic**. Production code contains no expected answer for any regression question, no line/station mapping, and no hard-coded acronym expansion. Domain facts still come exclusively from the indexed corpus.

No database migration, OCR rerun, re-chunking, re-embedding, or model-cache reset is required.

## What changes

The Q&A workflow is now:

```text
user question
    -> deterministic safe seed plan
    -> optional fast semantic evidence planner for structurally rich/indirect wording
    -> atomic evidence goals
    -> per-goal hybrid retrieval + independent definition lookup
    -> bounded candidate reservation per required goal
    -> one global rerank pass
    -> goal-satisfaction audit
    -> one targeted recovery pass for only missing/partial goals
    -> answer-structure plan (Research)
    -> evidence-bound synthesis
    -> risk/goal-aware verification
    -> targeted repair when required
```

### Key invariants

1. **One strong branch cannot hide another missing branch.** Confidence is reduced when a required goal has no evidence.
2. **Several identifiers are searched independently.** A multi-term request does not require all terms to occur in one chunk merely to establish each term.
3. **Relationships require relationship evidence.** Separate definitions or co-occurrence do not prove causation, override, dependency, eligibility, entitlement, sequence, or applicability.
4. **User premises are not facts.** Yes/no, “right?”, “correct?”, conditional and claim-like questions are eligible for semantic evidence planning and verification.
5. **Conditions and exceptions survive planning.** The planner is instructed to preserve numbers, negation, temporal qualifiers, exceptions, and intermediate events.
6. **Planner output is bounded and sanitized.** New short identifiers and new numeric values invented by the planner are rejected from retrieval queries; user-grounded seed goals are preserved if the semantic planner drops them.
7. **One bounded recovery pass only.** Missing/partial goals are retried independently instead of repeatedly broadening the whole question.
8. **Corpus absence is not inferred from top-k absence.** Existing corpus-negative safeguards remain in force.
9. **Access control is unchanged.** Every atomic retrieval uses the same document ACL/scope filters as the original query.
10. **Prompt injection in source text remains untrusted data.** Source content cannot change the system role or evidence rules.
11. **Evidence is goal-attributed.** The semantic audit may cite only evidence retrieval actually attributed to that goal; evidence for branch A cannot be borrowed to mark missing branch B complete.
12. **Overview isolation cannot hide later goals.** Compositional retrieval searches the configured overview source once per atomic goal rather than only for the first query.
13. **Recovery evidence cannot be crowded out by old context.** Targeted recovery evidence receives first admission to the bounded merged context.
14. **Semantic-planner failure fails safer.** If semantic planning was required but fails/returns invalid output, Auto is forced toward Research and verification instead of silently falling back to a confident Direct answer.
15. **Unicode wording is preserved.** Structural tokenization is Unicode-aware; non-English or mixed-language multiword requests can reach semantic evidence planning.
16. **Negative corpus inventory is explicit.** Requests for documents that do not mention a target are not answered by inverting ordinary top-k retrieval; until exhaustive inventory is implemented, IMS scopes/abstains instead of fabricating absence.

## Performance controls

Atomic retrieval increases search breadth, so 0.7.0 deliberately caps global cross-encoder work after per-goal candidate reservation:

```text
COMPOSITIONAL_MAX_GOALS=12
COMPOSITIONAL_MAX_QUERIES=14
COMPOSITIONAL_GOAL_CANDIDATE_RESERVE=2
COMPOSITIONAL_MAX_EVIDENCE_K=32
COMPOSITIONAL_RERANK_POOL=28
COMPOSITIONAL_RERANK_TOP_K=16
COMPOSITIONAL_RECOVERY_MAX_GOALS=4
```

This avoids multiplying the expensive CPU reranker by the number of subquestions. The retrieval fan-out happens in cheaper dense/lexical/exact/table lanes; one bounded rerank pool is used for the initial pass, with a single targeted recovery pass only when required.

## Upgrade an existing 0.6.2 VM

Follow the root [`APPLY.md`](../APPLY.md). Only `api-1` and `api-2` need rebuilding. Preserve all Docker volumes.

## Fresh laptop

Use the full 0.7.0 ZIP. From its root:

```powershell
python scripts\init_env.py
notepad .env
# Set OPENAI_API_KEY
python scripts\make_local_env.py
$env:PYTHONPATH = "src;."
python scripts\check_070_compositional_plan.py
```

Then start the local stack:

```powershell
docker compose --env-file .env.local `
  -f docker-compose.yml `
  -f docker-compose.local.yml `
  up -d --build `
  postgres valkey inference-query inference-ingest model-bootstrap migrate api-1 worker report-worker
```

Open `http://127.0.0.1:8081`.

## Acceptance

Do not test only one known question. Use the 42-class structural matrix in `eval/compositional_070_cases.json`, plus your normal golden corpus. Measure at least:

- required-goal retrieval recall;
- final goal completeness;
- unsupported relationship rate;
- false corpus-absence claims;
- citation correctness;
- contradiction handling;
- retrieval/rerank/LLM latency;
- targeted recovery frequency and success rate.

The retrieval trace now persists `evidence_plan`, `goal_stats`, `goal_satisfaction`, `goal_complete`, `goal_missing`, `goal_partial`, `goal_contradicted`, query-to-goal attribution, and recovery summary for production diagnosis.
