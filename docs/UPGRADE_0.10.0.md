# Upgrade to IMS 0.10.0 retrieval-control foundation

This release is an additive retrieval/control-plane upgrade. It does not require re-OCR or re-embedding existing ready PDFs for the features in this foundation release.

## 1. Back up

Before changing the deployment, back up PostgreSQL and preserve the current `.env` and 0.9.1 application images/source.

## 2. Review environment changes

Start from the updated `.env.example`. Important new controls include:

- `GOVERNING_REFERENCE_*`
- `ROUTED_DOCUMENT_BOOST_*`
- `CORPUS_TERM_*`
- `QUERY_FRAME_ENABLED`
- `LANE_RESERVATION_*`
- `RECOVERY_MAX_QUERIES_PER_GOAL`
- `HNSW_ITERATIVE_SCAN_MODE`
- `QUERY_CPUSET`

`SPECULATIVE_RETRIEVAL_ENABLED` remains `false`; true parallel fan-out is not part of this foundation patch yet.

Do not copy `deploy/env/0.10-query-benchmark.env` blindly into production. It is a challenger profile for measured reranker/CPU tests.

## 3. Build the new images

If testing OpenVINO, the inference image must be rebuilt with:

```text
INSTALL_OPTIMIZED_RERANK=1
```

For a normal compatibility deployment, you may keep the existing Torch backend first, validate correctness, then benchmark OpenVINO separately.

## 4. Apply migration

Run the existing migrate service / Alembic upgrade path. New revision:

```text
0005_retrieval_control_foundation
```

It adds document authority metadata and the `corpus_terms` table/index.

## 5. Start services

Start the normal Compose stack and verify health endpoints before backfill.

## 6. Backfill retrieval intelligence / corpus terms

Run:

```bash
python scripts/build_retrieval_intelligence.py --limit 100
```

Repeat until the script reports:

```text
pending_documents=0
```

The 0.10 intelligence index derives document/section vectors from existing chunk embeddings and builds corpus vocabulary from existing metadata/text. It should not invoke PDF OCR.

## 7. Mark governing sources

MRGR continues to work through `GOVERNING_REFERENCE_PATTERNS=MRGR` as a compatibility fallback.

For durable future administration, set `source_role=governing_reference` on governing sources when they are uploaded/managed. Do not hard-code governing document UUIDs in application logic.

## 8. Run regression evaluation

At minimum run the existing retrieval-control/industrial/compositional tests and the organization golden set.

Compare 0.9.1 and 0.10 on:

- gold evidence recall before reranking;
- rerank-pool recall;
- post-rerank recall;
- governing-source recall and false inclusion;
- table/scalar questions;
- condition-sensitive questions;
- typo/informal questions;
- wrong-premise questions;
- ACL cases;
- p50/p95 per stage.

## 9. Reranker benchmark

Evaluate current Torch settings against OpenVINO candidate profiles. Do not accept a faster profile if it materially reduces gold-evidence recall or citation support.

On an 8-vCPU host, also benchmark separated CPU sets. The supplied challenger profile uses:

```text
ONLINE_CPUSET=0-1
QUERY_CPUSET=2-5
BACKGROUND_CPUSET=6-7
QUERY_ML_NUM_THREADS=4
```

Treat this as an experiment, not a universal default.

## 10. Rollback

The migration is additive. If application behavior regresses, the preferred emergency rollback is:

1. stop 0.10 application containers;
2. restore 0.9.1 application images/config;
3. leave the additive 0.10 database columns/table in place unless database downgrade is specifically required.

0.9.1 code does not need to use the new fields.

## Important behavior changes

- MRGR/governing references no longer short-circuit broad retrieval.
- corpus/document routing no longer silently becomes a hard search scope.
- original Q0 is explicitly retained before expansions.
- corpus typo/terminology candidates are search hints only.
- generic scalar questions can activate table retrieval without domain-specific keyword patches.
- new controller payloads use strict Pydantic structured-output paths where migrated.

See `docs/RETRIEVAL_CONTROL_PLANE_0.10.0.md` for the complete design and remaining phases.
