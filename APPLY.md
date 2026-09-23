# Apply IMS 0.10.0 retrieval-control foundation

For 0.10.0, use [`docs/UPGRADE_0.10.0.md`](docs/UPGRADE_0.10.0.md). The steps below are retained as historical 0.9.1 upgrade guidance only.

Key 0.10 requirements before production promotion:

- apply Alembic migration `0005_retrieval_control_foundation`;
- rebuild/backfill retrieval intelligence to index version 0.10.0;
- run the gold retrieval set and ACL tests;
- do not enable speculative retrieval yet;
- treat `deploy/env/0.10-query-benchmark.env` as a benchmark profile, not production defaults;
- compare Torch/OpenVINO and CPU-isolation profiles on evidence recall and p95 latency.

---

# Apply IMS 0.9.1 over IMS 0.9.0

This procedure assumes the 0.9.0 corpus-intelligence backfill has been stopped. It is written for the production state where one or more 0.9.0 secondary-index documents may already be committed.

## 1. Confirm the old backfill is stopped

```bash
cd ~/institutional-knowledge-engine

docker ps --format 'table {{.Names}}\t{{.Status}}' \
  | grep 'api-1-run' \
  || echo "No retrieval-intelligence backfill container is running"
```

Do not continue if a `build_retrieval_intelligence.py` container is still running.

## 2. Keep corpus-intelligence routing disabled during the rebuild

```bash
if grep -q '^RETRIEVAL_INTELLIGENCE_ENABLED=' .env; then
  sed -i \
    's/^RETRIEVAL_INTELLIGENCE_ENABLED=.*/RETRIEVAL_INTELLIGENCE_ENABLED=false/' \
    .env
else
  printf '\nRETRIEVAL_INTELLIGENCE_ENABLED=false\n' >> .env
fi

grep '^RETRIEVAL_INTELLIGENCE_ENABLED=' .env
```

Expected:

```text
RETRIEVAL_INTELLIGENCE_ENABLED=false
```

Normal chunk retrieval remains available while this secondary index is rebuilt.

## 3. Back up current source and environment

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
cp .env ~/.env-before-0.9.1-$STAMP

tar \
  --exclude='./data' \
  --exclude='./.git' \
  --exclude='*.zip' \
  -czf ~/ims-source-before-0.9.1-$STAMP.tar.gz .
```

A new database migration is not required. The existing 0.9.0 DB backup remains valid, but take a fresh DB dump if your change-control policy requires one.

## 4. Overlay the 0.9.1 patch

Assuming the patch is at `~/ims-0.9.1-corpus-intelligence-optimization-patch.zip`:

```bash
rm -rf /tmp/ims091
mkdir -p /tmp/ims091

python3 -m zipfile -e \
  ~/ims-0.9.1-corpus-intelligence-optimization-patch.zip \
  /tmp/ims091

cp -a \
  /tmp/ims091/ims-0.9.1-corpus-intelligence-optimization-patch/. \
  ~/institutional-knowledge-engine/

cd ~/institutional-knowledge-engine

grep '__version__' src/ike/__init__.py
grep '^version' pyproject.toml
```

Both must report `0.9.1`.

## 5. Validate before rebuilding services

```bash
PYTHONPATH=src:. python3 scripts/check_091_corpus_intelligence.py
python3 -m compileall -q src scripts inference_service
```

Expected:

```text
PASS: IMS 0.9.1 corpus-intelligence efficiency and coverage-safety checks
```

Run the full test suite in the existing Python 3.12 API image:

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

Do not proceed if tests fail.

## 6. Build only changed runtime images

```bash
docker compose build api-1 api-2 worker
```

No rebuild is required for PostgreSQL, Valkey, inference-query, inference-ingest, migrate or Nginx.

## 7. Rolling API recreation while corpus intelligence stays disabled

```bash
docker compose up -d --no-deps --force-recreate api-1
until curl -fsS http://127.0.0.1:8081/health/live >/dev/null; do sleep 3; done
docker compose exec api-1 python -c "import ike; print(ike.__version__)"
```

Expected `0.9.1`.

Then:

```bash
docker compose up -d --no-deps --force-recreate api-2
until curl -fsS http://127.0.0.1:8082/health/live >/dev/null; do sleep 3; done
docker compose exec api-2 python -c "import ike; print(ike.__version__)"
```

Expected `0.9.1`.

Recreate the worker so future uploads use the optimized node builder:

```bash
docker compose up -d --no-deps --force-recreate worker
docker compose ps worker
```

Do not restart `inference-ingest`; 0.9.1 backfill does not use it.

## 8. Run a small optimized backfill batch first

Start with 10 documents to verify production speed and node counts:

```bash
docker compose run --rm --no-deps api-1 \
  python scripts/build_retrieval_intelligence.py --limit 10
```

Important expected behavior:

- there should be **no** `POST http://inference-ingest:8090/embed/documents` lines;
- each completed document should log `retrieval_intelligence_indexed ... elapsed_s=...`;
- the already indexed 0.9.0 document is automatically rebuilt because its index version is old;
- node counts should be approximately `1 + number of section groups`, not hundreds of concept vectors.

Inspect:

```bash
docker compose exec -T postgres sh -lc \
  'psql -X -U "$POSTGRES_USER" -d "$POSTGRES_DB" -P pager=off' <<'SQL'

SELECT
    node_type,
    COUNT(*) AS nodes
FROM retrieval_nodes
GROUP BY node_type
ORDER BY node_type;

SELECT
    COUNT(*) FILTER (
        WHERE extra_metadata->'retrieval_intelligence'->>'version' = '0.9.1'
          AND extra_metadata->'retrieval_intelligence'->>'status' = 'ready'
    ) AS current_index_documents,
    COUNT(*) FILTER (WHERE ingestion_status = 'ready') AS ready_documents
FROM documents;

SQL
```

For a fully 0.9.1-built index there should be no newly created `concept` rows. A legacy concept count can exist only until its owning 0.9.0 document is rebuilt.

## 9. Continue in resumable batches

If the first 10-document batch is healthy:

```bash
docker compose run --rm --no-deps api-1 \
  python scripts/build_retrieval_intelligence.py --limit 100
```

Repeat the same command. Each run skips current 0.9.1 documents and advances through missing/outdated documents.

The final line now includes:

```text
indexed_documents=... failed_documents=... retrieval_nodes=... pending_documents=... elapsed_s=...
```

Continue until:

```text
pending_documents=0
```

Do **not** use `--rebuild` for this upgrade. Version-aware selection already rebuilds the old 0.9.0 nodes exactly once.

## 10. Verify full secondary-index coverage

```bash
docker compose exec -T postgres sh -lc \
  'psql -X -U "$POSTGRES_USER" -d "$POSTGRES_DB" -P pager=off' <<'SQL'

SELECT
    COUNT(*) FILTER (WHERE ingestion_status = 'ready') AS ready_documents,
    COUNT(*) FILTER (
        WHERE ingestion_status = 'ready'
          AND extra_metadata->'retrieval_intelligence'->>'version' = '0.9.1'
          AND extra_metadata->'retrieval_intelligence'->>'status' = 'ready'
    ) AS indexed_091_documents
FROM documents;

SELECT
    node_type,
    COUNT(*) AS nodes,
    pg_size_pretty(pg_total_relation_size('retrieval_nodes')) AS total_relation_size
FROM retrieval_nodes
GROUP BY node_type
ORDER BY node_type;

SQL
```

`ready_documents` and `indexed_091_documents` must match before enabling hard corpus routing globally.

## 11. Enable corpus intelligence

```bash
sed -i \
  's/^RETRIEVAL_INTELLIGENCE_ENABLED=.*/RETRIEVAL_INTELLIGENCE_ENABLED=true/' \
  .env

grep '^RETRIEVAL_INTELLIGENCE_ENABLED=' .env
```

Rolling-recreate the APIs again so they load the setting:

```bash
docker compose up -d --no-deps --force-recreate api-1
until curl -fsS http://127.0.0.1:8081/health/live >/dev/null; do sleep 3; done

docker compose up -d --no-deps --force-recreate api-2
until curl -fsS http://127.0.0.1:8082/health/live >/dev/null; do sleep 3; done
```

No worker or inference service restart is needed at this point.

## 12. Verify the coverage gate in a real trace

After an unseen Auto question:

```sql
SELECT
    created_at AT TIME ZONE 'Asia/Kolkata' AS created_at_ist,
    question,
    retrieval_trace->'corpus_discovery' AS corpus_discovery,
    retrieval_trace->'routed_document_ids' AS routed_document_ids,
    retrieval_trace->>'retrieval_effort' AS retrieval_effort,
    latency_ms,
    input_tokens,
    output_tokens
FROM query_logs
ORDER BY created_at DESC
LIMIT 5;
```

`corpus_discovery` should expose `scope_documents`, `indexed_documents`, `coverage_ratio`, `routing_safe` and `index_version`.

## Do not run

```text
docker compose down
docker compose down -v
--remove-orphans
bulk PDF reprocessing
OCR/re-chunk/re-embed jobs
restart PostgreSQL
restart Nginx
restart inference-ingest for this patch
```
