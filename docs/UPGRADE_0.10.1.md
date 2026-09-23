# Upgrade to IMS 0.10.1 - stateless Q&A requests

IMS 0.10.1 removes cross-query retrieval context. A new Q&A request no longer inherits `line` or `rolling_stock` from earlier requests by the same user.

## Behavior

Every question is independently interpreted from the current request. The only persistent state that can affect retrieval is institutional/index state, ACL/security state, administrator metadata, and document IDs explicitly supplied on the current request. Query history remains available for display, audit, feedback, and evaluation but is not used as retrieval context.

Same-query intelligence is unchanged: IMS can still normalize terminology, expand acronyms/typos, generate retrieval hypotheses, search governing references, and search the global ACL-accessible corpus for the current question.

## Database and indexing

No database migration is required. No PDF reprocessing, OCR, embedding rebuild, or corpus-intelligence backfill is required. `INDEX_VERSION` intentionally remains `0.10.0`.

## Deployment

Replace the modified source files, rebuild the API-derived images, and recreate the application containers. The PostgreSQL/Valkey/model/shared-data volumes are not changed. Never use `docker compose down -v`.
