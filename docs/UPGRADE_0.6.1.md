# IMS 0.6.1 — Compound-Entity & Facet-Aware Retrieval

## What changed

0.6.1 addresses a general retrieval failure exposed by questions that combine a multi-word entity with one or more requested attributes. In 0.6.0, a definition cue could truncate `Crew Control` to `CREW`, while the raw conversational sentence was sent to PostgreSQL `simple` FTS. The result could be strong dense/lookup activity but zero useful lexical/table/exact hits, followed by candidate crowd-out and an incomplete answer.

The 0.6.1 path now:

- preserves bounded multi-word entities in definition and attribute formulations;
- records explicit query facets (`definition`, `location`, `contact`, `constraint`, `enumeration`);
- generates bounded entity/topic lexical and exact forms and consumes them before optional LLM query expansions;
- supplies conservative singular/plural forms for PostgreSQL `simple` FTS without general stemming;
- uses a document-diverse entity-attribute coverage lane in Research mode, with attribute words used as ranking hints rather than mandatory filters so terse tables remain searchable;
- protects coverage evidence from global reranker crowd-out;
- routes Auto multi-facet/entity-attribute questions directly to Research;
- adds verification risk for entity coverage/multi-facet answers;
- prevents unsupported corpus-wide negative claims when deterministic coverage was not completed.

Production retrieval remains corpus-generic. The regression tests include the query that exposed the problem, but production code contains no known Crew Control locations, DMRC line mappings, station mappings or document-specific answer rules.

## Data/model impact

There is no new migration and no ingestion-format change. Existing chunks, table context, embeddings and source files remain compatible. Therefore **do not reprocess the PDF corpus for this upgrade**.

## Fresh laptop / personal-PC test

### Prerequisites

Use Docker Desktop (Windows/macOS) or Docker Engine + Compose v2 (Linux), Python 3.12 for the helper/check scripts, at least 4 logical CPU threads, and enough free RAM/disk for the local BGE embedding/reranker and Docling model cache. The first model bootstrap is slower because model artifacts may need to download; subsequent builds/runs reuse the Docker/model volumes.

### Windows PowerShell

From the extracted `institutional-knowledge-engine-0.6.1` directory:

```powershell
python scripts/init_env.py
notepad .env
```

Set `OPENAI_API_KEY` in `.env`, save it, then create the laptop profile:

```powershell
python scripts/make_local_env.py
$env:PYTHONPATH = "src"
python scripts/check_061_query_plan.py
```

Start the standalone local stack without Nginx/TLS:

```powershell
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml up -d --build postgres valkey inference-query inference-ingest model-bootstrap migrate api-1 worker report-worker
```

Open `http://127.0.0.1:8081`.

Follow startup progress with:

```powershell
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml ps
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml logs -f --tail=100 api-1 inference-query worker
```

### Linux/macOS shell

```bash
python3 scripts/init_env.py
# Edit .env and set OPENAI_API_KEY.
python3 scripts/make_local_env.py
PYTHONPATH=src python3 scripts/check_061_query_plan.py

docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml \
  up -d --build postgres valkey inference-query inference-ingest model-bootstrap migrate api-1 worker report-worker
```

Open `http://127.0.0.1:8081`.

### Upload a small test corpus

Use the normal IMS upload UI. For retrieval regression testing, upload only a few PDFs first. Wait for their ingestion status to complete, then test compound-entity/facet questions and ordinary procedure/acronym questions. No preloaded corporate corpus is bundled in the source ZIP.

### Stop the laptop stack without deleting data

```bash
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml down
```

Do **not** add `-v` unless you intentionally want to delete the local PostgreSQL corpus and model/shared-data volumes.

## Existing 0.6.0 VM

Use the root [`APPLY.md`](../APPLY.md). Because only API-side retrieval/workflow Python changed, the fast deployment path is:

```bash
docker compose build api-1 api-2
docker compose up -d --no-deps --force-recreate api-1 api-2
docker compose restart nginx
```

`inference-query` does not need a rebuild for 0.6.1. No PDF reprocessing is required.
