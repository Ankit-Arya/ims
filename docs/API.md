# API Guide — 0.5.0

The external API is served through Nginx/HTTPS and load-balanced across two equivalent API containers. `POST /api/v1/query/stream` remains NDJSON and now includes a truthful Q&A admission/queue stage when applicable. Query results may contain `visuals[]` with short-lived signed tokens. `GET /api/v1/visuals/render/{token}` renders the authorised source page/crop after rechecking document ACLs. Clients never supply filesystem paths.

Local ML is private: query API uses `QUERY_INFERENCE_URL`; ingestion uses `INGESTION_INFERENCE_URL`.

---

## Historical / earlier-release notes


Interactive OpenAPI is available at `/docs` on a running API.

## Authentication

### `POST /api/v1/auth/login`

```json
{"username":"admin","password":"..."}
```

Returns a bearer token and sets the built-in UI session cookie.

### `GET /api/v1/auth/me`

Current identity, role and department.

### `GET /api/v1/auth/directory`

Returns active colleagues (excluding the current user) as a minimal directory used by the personal-PDF sharing picker.

Admin-only user management remains under `/api/v1/auth/users...`.

## Organisation Knowledge documents

### `GET /api/v1/documents`

Returns accessible **organisation** documents, including non-ready lifecycle states where authorized.

### `POST /api/v1/documents`

Analyst/admin multipart PDF upload to the governed organisation corpus.

### Generic document source/lifecycle routes

These routes work for either an accessible organisation document or an accessible personal document:

- `GET /api/v1/documents/{id}`
- `GET /api/v1/documents/{id}/view`
- `GET /api/v1/documents/{id}/download`
- `POST /api/v1/documents/{id}/retry`
- `POST /api/v1/documents/{id}/reindex`
- `DELETE /api/v1/documents/{id}`

Management authorization differs by workspace: organisation rules apply to Knowledge; only the personal-PDF owner may manage a personal PDF.

## Personal PDF library

### `GET /api/v1/library`

Returns personal documents the current user owns **or** has been explicitly shared.

### `POST /api/v1/library`

Any authenticated user may upload a personal PDF.

Multipart fields:

- `file` — required PDF;
- `title` — optional;
- `shared_user_ids` — optional comma-separated user UUIDs.

Personal duplicate detection is SHA-256-based and scoped to the same owner. A duplicate failed document is retried rather than duplicated.

### `PUT /api/v1/library/{document_id}/shares`

Owner-only. Replaces the complete share list.

```json
{"user_ids":["uuid-1","uuid-2"]}
```

## Q&A

Accepted Q&A modes:

- `auto`
- `direct`
- `research`
- `qa` (legacy alias for direct)

**Deep Analysis is a UI answer mode backed by the asynchronous `/reports` API**, not by the synchronous Q&A endpoint.

### `POST /api/v1/query`

```json
{
  "question":"What is BIC?",
  "mode":"auto",
  "document_ids":null
}
```

If `document_ids` is supplied, every ID must refer to a ready document accessible to the current user. Personal-document row-level access is checked in the same retrieval permission predicate as organisation ACLs.

### `POST /api/v1/query/stream`

Preferred browser endpoint. Returns `application/x-ndjson` progress/result events.

Example progress event:

```json
{"type":"progress","stage":"rerank","label":"Reranking the strongest passages","percent":42}
```

The UI deliberately renders shorter end-user labels. The API does not stream private chain-of-thought.

### `GET /api/v1/query/history?limit=100`

Current user's Q&A only.

## Deep Analysis

### `POST /api/v1/reports`

Creates a private asynchronous analysis job.

```json
{
  "title":"Compare door isolation procedures",
  "objective":"Compare door isolation procedures, differences and exceptions across the selected manuals.",
  "document_ids":["uuid-1","uuid-2"]
}
```

The caller must be authorized for every selected document. The count is capped by `REPORT_MAX_DOCUMENTS`.

### `GET /api/v1/reports`

Current user's report/Deep Analysis jobs only.

### `GET /api/v1/reports/{id}`

Current user's job only. Response includes progress fields:

- `progress_stage`
- `progress_percent`
- `progress_message`

Completed results contain Markdown and validated citations.

## Access-control rule

Document IDs supplied by a client are never trusted as authorization. The database query applies the same document access clause used by normal retrieval/view/download.

## Dashboard summary

### `GET /api/v1/dashboard/summary`

Authenticated, user-scoped summary used by the workspace activity rail.

Response fields:

- `question_count`
- `deep_analysis_count`
- `available_pdf_count`
- `knowledge_pdf_count`
- `personal_pdf_count`
- `ready_pdf_count`
- `indexing_pdf_count`
- `failed_pdf_count`
- `recent_questions` (latest five Q&A items)

Document counts apply the same row-level access predicate as normal document listing/retrieval. The endpoint does not grant additional visibility into personal PDFs.