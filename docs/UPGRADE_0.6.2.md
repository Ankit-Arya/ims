# IMS 0.6.2 — Multi-Entity Enumeration Hotfix + Citation Preview

## Why this hotfix exists

IMS 0.6.1 fixed compound definition/entity parsing but introduced a regression for coordinated list requests. A question such as `List of depots and crew controls` was represented as one entity phrase (`depots and crew controls`). PostgreSQL lexical/exact retrieval and the deterministic entity-coverage lane therefore favored passages where all terms happened to co-occur, which can surface incidental operating/control material instead of authoritative lists for each requested category.

0.6.2 represents coordinated enumeration categories independently. It also tightens protected coverage so low-signal incidental mentions are not automatically preserved merely because they occur in a different document.

## Retrieval changes

- `List of depots and crew controls` becomes two targets: `depots` and `crew controls`.
- Compound names such as `safety and security equipment` are not blindly split on `and`.
- Lexical/exact/singular-plural forms are generated independently for every extracted category.
- Entity coverage performs one indexed discovery query per category and round-robins protected evidence across categories, preventing the first target from consuming the complete evidence budget.
- Structured-list candidates receive table/structural ranking signals; candidates far below the strongest structural score are not automatically protected.
- When only weak entity evidence is available, the coverage trace is marked incomplete instead of turning many incidental mentions into an apparent exhaustive result.
- Table pre-ranking ignores conversational function words such as `what`, `the`, `and`, `list`, and `provide`.
- Enumeration answer instructions prioritize requested identity/name/location/count facts and suppress unrelated operating context unless it directly qualifies an item.
- Verification explicitly checks that every requested category is represented.

Production code contains no depot names, crew-control locations, line mappings, or document-specific answers.

## UI changes

- The Copy action is explicitly retained for completed Q&A and Deep Analysis cards, with both modern Clipboard API and legacy browser fallback.
- Inline citations such as `[E3]` are interactive. Hover or keyboard focus shows a compact source preview containing document name, page, section, filename, and the stored cited excerpt.
- The existing expandable Sources panel is unchanged.
- Version `0.6.2` changes the static asset URL, preventing 0.6.1 JS/CSS from being mixed with the new API after deployment.

## Data and model impact

No schema or ingestion-format changes are present. Existing PDFs, Docling output, chunks, search vectors, embeddings, and model volumes remain compatible. **Do not reprocess the corpus.**

## Fresh laptop test

From the extracted `institutional-knowledge-engine-0.6.2` directory:

```powershell
python scripts\init_env.py
notepad .env
```

Set `OPENAI_API_KEY`, then:

```powershell
python scripts\make_local_env.py
$env:PYTHONPATH = "src"
python scripts\check_062_query_plan.py

docker compose --env-file .env.local `
  -f docker-compose.yml `
  -f docker-compose.local.yml `
  up -d --build `
  postgres valkey inference-query inference-ingest model-bootstrap migrate api-1 worker report-worker
```

Open `http://127.0.0.1:8081`, upload a small test corpus, wait for ingestion to complete, and run the acceptance questions in the root `APPLY.md`.

Stop without deleting local data/model volumes:

```powershell
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml down
```

Never add `-v` unless you intentionally want to erase the local test database and model/shared-data volumes.
