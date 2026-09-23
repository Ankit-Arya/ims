# Upgrade IMS 0.9.0 -> 0.9.1

0.9.1 is a backend efficiency/safety patch for the corpus-intelligence layer. It requires **no new Alembic migration, no PDF reprocessing, no OCR, no re-chunking, no re-embedding of source chunks, and no inference-query model rebuild**.

Existing `retrieval_nodes` created by 0.9.0 are rebuildable secondary data. The 0.9.1 backfill automatically replaces older-version nodes document-by-document.

Follow `APPLY.md` for the production command sequence.
