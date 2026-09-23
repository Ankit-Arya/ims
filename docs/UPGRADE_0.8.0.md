# Upgrade IMS 0.7.0 -> 0.8.0

IMS 0.8.0 is a retrieval-control-plane upgrade. It keeps the 0.7.0 compositional planner but changes how evidence is located, prioritized, reranked, audited and recovered.

## Why this release exists

0.7.0 could correctly decompose a question yet still let weak surface-word queries generate hundreds of candidates that drowned out a direct provision in a configured governing source. 0.8.0 addresses the generic failure chain rather than encoding regression-specific answers.

Key changes:

- fast semantic evidence planning for ordinary multi-word natural-language questions;
- coverage contracts for variants, enumerations, procedures, relationships and conditions;
- staged priority-source retrieval with stopping criteria;
- separate precise and relaxed lexical lanes;
- hierarchy/section navigation over existing chunk metadata;
- table-lane gating to reduce unrelated table pollution;
- bounded goal-local cross-encoder reranking;
- evidence-quality audit before broadening;
- LLM retrieval repair using search-only hypotheses plus corpus snippets;
- repair queries admitted before failed seed queries during targeted recovery;
- broad fallback limited to unresolved goals;
- richer retrieval trace.

## Data compatibility

There is no schema migration and no corpus rewrite in this release. Existing documents, chunks, BGE-M3 embeddings, pgvector index and source files remain compatible.

Do not run a bulk reprocess merely to install 0.8.0.

## Configuration compatibility

A fresh environment can explicitly set `PRIORITY_DOCUMENT_PATTERNS`. On an existing environment, if that value is absent/blank, IMS uses the existing `OVERVIEW_DOCUMENT_PATTERN` as the priority-source pattern. This preserves the deployment's existing general-rule/source-family selection while changing it from a weighted background lane into a real first-stage retrieval policy.

If no source family should be privileged globally, set:

```text
PRIORITY_DOCUMENT_PATTERNS=
OVERVIEW_DOCUMENT_PATTERN=
```

or disable the stage with:

```text
PRIORITY_RETRIEVAL_ENABLED=false
```

Explicit user-selected `document_ids` remain a hard scope regardless of these settings.

## Recommended validation sequence

Before production rollout:

```bash
PYTHONPATH=src:. python3 scripts/check_080_retrieval_control_plane.py
python3 -m compileall -q src scripts
```

Run the full suite in the Python 3.12 API image rather than installing the application into a Python 3.13 VM host:

```bash
docker compose run --rm \
  --no-deps \
  --user root \
  -v "$PWD:/work" \
  -w /work \
  -e PYTHONPATH=/work/src:/work \
  api-1 \
  sh -lc 'python -m pip install --no-cache-dir "pytest>=8.3,<9" && python -m pytest -q'
```

Then use a representative golden set that includes easy direct facts as well as paraphrase, homonym, variants, procedures, lists, relationships, false premises, conditions, scope/revision conflicts and source-priority questions.

## Runtime rollout

Only the API replicas need rebuilding for this patch. Inference-query, inference-ingest, ingestion worker, report worker, PostgreSQL, Valkey and Nginx configurations are unchanged.

Roll API replicas one at a time and confirm `/health/live` between replicas. Full commands are in the root `APPLY.md`.

## Rollback

Because no migration or corpus rewrite occurs, rollback is source/image-only: restore the 0.7.0 source backup, rebuild the two API images and recreate them one at a time. Preserve all volumes.
