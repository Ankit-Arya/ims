# Upgrade to 0.4.6 — UI cache-integrity hotfix

## Why this release exists

0.4.5 introduced a new three-column desktop shell and new dashboard JavaScript while keeping the same static asset URLs (`/static/app.css` and `/static/app.js`). A browser or intermediary cache could therefore combine the new 0.4.5 HTML with an older 0.4.4 stylesheet/script. The visual signature is an old horizontal Workspace nav, the new activity rail stacked below the page, oversized/incorrect Ask layout, and dashboard placeholders that never populate.

0.4.6 removes that mixed-version state.

## Changes

- HTML references CSS and JavaScript with the application release as a cache key, for example `app.css?v=0.4.6` and `app.js?v=0.4.6`.
- The root HTML shell is returned with `Cache-Control: no-store` so it is revalidated on every app load.
- Versioned static assets are cacheable as immutable resources.
- The application version is sourced from `ike.__version__` in FastAPI and supplied to the template.
- No ingestion, worker, database schema, OCR, embeddings, queue or retrieval changes.

## Deployment

Rebuild and recreate only `api`:

```bash
docker compose build api
docker compose run --rm --no-deps api python -c "import ike; print(ike.__version__)"
docker compose up -d --no-deps --force-recreate api
```

Expected version: `0.4.6`.

A normal browser refresh is sufficient after this release because the HTML points to a new static asset URL. A one-time hard refresh is harmless but should no longer be required for subsequent releases using the same versioning scheme.
