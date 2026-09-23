# IMS 0.12.0 Upgrade

IMS 0.12.0 splits the product into two experiences on the same authorised corpus.

## Q&A
- Open-ended knowledge questions.
- Comprehensive is the default.
- Existing research, coverage, recovery and verification remain available.

## Real-Time IMS
- Session context: Line, rolling stock, operational role, department, optional mode/location.
- Reported conditions are treated as scenario premises, not claims to fact-check.
- Direct-only bounded retrieval with a small candidate/reranker pool.
- Broad research escalation and recovery are disabled for the real-time path.
- Answers are instructed to lead with ACTION NOW and remain concise.

## Deployment properties
- No Alembic migration.
- No corpus reprocessing.
- No vector re-embedding.
- Only API containers need rebuilding/recreation.
- Rollback to 0.11.0 requires restoring the previous code tree/image and recreating API containers.
