# Deployment — 0.5.0

Production on the current VM is Nginx/HTTPS -> two internal API replicas -> isolated query inference, with a separate background ingestion inference plane. Do not expose API, PostgreSQL, Valkey or inference ports publicly. The supplied Compose maps API/inference diagnostics to loopback only.

Use `/etc/ims/tls/fullchain.pem` and `/etc/ims/tls/dmrc.key` via `TLS_CERT_DIR`; never place TLS private keys inside the repository or release archive. See `docs/UPGRADE_0.5.0.md` for the in-place upgrade sequence.

---

## Historical / earlier-release notes


## Laptop pilot

Recommended starting point:

```text
ML_DEVICE=cpu
INGEST_WORKER_CONCURRENCY=auto
INGEST_CPU_TARGET_PERCENT=75
INGEST_MAX_CONCURRENCY=8
DOCLING_NUM_THREADS_PER_DOCUMENT=2
TESSERACT_OMP_THREAD_LIMIT=1
```

Start:

```powershell
python scripts/init_env.py
docker compose up --build -d
```

Check:

```powershell
docker compose ps
curl.exe http://localhost:8080/health/ready
docker compose logs --tail=100 worker
```

The worker startup log reports chosen auto concurrency.

## Upgrade deployment

0.4 adds Alembic migration `0002_personal_workspaces`.

```powershell
docker compose build api worker report-worker migrate model-bootstrap
docker compose run --rm migrate
docker compose up -d --force-recreate api worker report-worker
```

Do not delete volumes.

## Ingestion capacity

One worker container can now process multiple PDFs concurrently through prefork processes. For a server, first tune the automatic concurrency limits against the server's Docker CPU/RAM allocation.

Only after measuring one worker container should you consider multiple worker replicas. If you scale replicas, remember that **each replica independently computes its own concurrency from the resources visible inside that container**. Set Docker resource limits or explicit `INGEST_WORKER_CONCURRENCY` so total host concurrency is intentional.

Example scale-out after capacity testing:

```powershell
docker compose up -d --scale worker=2
```

Do not combine a high per-container prefork count with many replicas without measuring memory.

## Reverse proxy / mobile access

To expose IKE to phones on an internal network, bind/rout the API through your approved reverse proxy/DNS/TLS configuration. The UI itself is responsive; network reachability, TLS certificates and firewall rules are deployment concerns.

For organisational access, do not rely on an unauthenticated public port. Put the application behind the organisation's standard ingress and identity/network controls.

For streaming Q&A, configure the reverse proxy not to buffer `/api/v1/query/stream` responses.

## Production recommendations

The supplied Compose stack is appropriate for a pilot/controlled internal deployment. Before broad production rollout consider:

- SSO/enterprise identity integration;
- managed secret storage;
- TLS/reverse proxy;
- PostgreSQL backup/restore drills;
- persistent source-file backup;
- centralized logs/metrics/tracing;
- rate limits/quotas;
- container vulnerability scanning;
- explicit CPU/RAM limits;
- tested worker capacity;
- retention policies for personal PDFs and question history.