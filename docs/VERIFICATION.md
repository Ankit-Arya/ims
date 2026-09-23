# Verification status — release 0.4.0

## Verified in the source workspace

The source release was statically/regression checked for:

- Python compilation;
- browser JavaScript syntax (`node --check`);
- TOML/YAML parsing;
- deterministic pytest suite;
- API/worker dependency boundary;
- Q&A mode/current-request/history UI contracts;
- responsive navigation/theme/personal-library UI contract;
- personal document owner/share authorization predicate;
- personal upload/share endpoints;
- owner-scoped personal deduplication/retry behavior;
- CPU/memory-aware worker bootstrap contract;
- Tesseract one-thread wrapper contract;
- Docling threaded-pipeline configuration contract;
- existing document lifecycle/retry/source-view/download contracts.

The deterministic regression suite currently passes **26 tests** in this source workspace.

## What still requires target-machine acceptance

This build environment cannot certify the user's Docker Desktop runtime, CPU/RAM scheduling, real PDFs, live model volume or LLM credentials. Test on the target laptop/server:

1. apply migration 0002;
2. API/worker/report-worker healthy startup;
3. worker log reports sensible auto concurrency;
4. upload at least 3 representative PDFs simultaneously;
5. observe CPU/RAM and confirm concurrent processing without OOM/restarts;
6. compare extraction quality to the previous accurate-table baseline;
7. test phone-width UI in a real mobile browser;
8. light/dark theme persistence;
9. personal PDF private upload;
10. explicit share/revoke across two users;
11. direct UUID/view/download/Q&A denial for an unshared user;
12. Auto/Direct/Research/Deep Analysis from Ask;
13. private My Questions across two users.

## Performance caveat

The 75% ingestion CPU target does not guarantee 75% observed host CPU. Docker CPU allocation, memory cap, PDF characteristics, OCR density, model/inference bottlenecks and I/O can all lower or vary utilization. Tune from measured throughput and quality, not CPU percentage alone.

## 0.4.2 verification

The 0.4.2 procedure-coverage changes were statically verified with Python compilation and the full regression suite. Result: **29 tests passed**. New tests cover deterministic coverage-query extraction and procedure/steps classification. Runtime corpus-completeness still requires validation against the real GCP document index.

## 0.4.3 verification

The strict ingestion-integrity guard was added after GCP/local index comparison showed identical source checksums but partial GCP chunk coverage. Static regression checks verify that non-`SUCCESS` Docling conversions and timeout errors cannot pass through the indexing path as successful documents.
