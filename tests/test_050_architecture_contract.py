from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_online_and_ingestion_inference_are_physically_separate():
    compose = read("docker-compose.yml")
    main = read("inference_service/main.py")
    client = read("src/ike/services/inference_client.py")
    assert "inference-query:" in compose
    assert "inference-ingest:" in compose
    assert "INFERENCE_ROLE: query" in compose
    assert "INFERENCE_ROLE: ingest" in compose
    assert "RERANK_MODEL" in compose
    ingest_block = compose.split("inference-ingest:", 1)[1].split("model-bootstrap:", 1)[0]
    assert "RERANK_MODEL" not in ingest_block
    assert 'require_role("query")' in main
    assert 'require_role("ingest")' in main
    assert "for_query" in client
    assert "for_ingestion" in client


def test_current_vm_defaults_prioritize_online_qa_and_single_ingestion():
    env = read(".env.example")
    config = read("src/ike/core/config.py")
    compose = read("docker-compose.yml")
    assert "INGEST_WORKER_CONCURRENCY=1" in env
    assert "INGEST_MAX_CONCURRENCY=1" in env
    assert 'ingest_worker_concurrency: str = "1"' in config
    assert "ONLINE_CPUSET=0-5" in env
    assert "BACKGROUND_CPUSET=6-7" in env
    assert "cpuset: ${BACKGROUND_CPUSET:-6-7}" in compose
    assert "mem_limit: ${WORKER_MEMORY_LIMIT:-8g}" in compose
    assert "mem_limit: ${QUERY_INFERENCE_MEMORY_LIMIT:-5g}" in compose


def test_two_lightweight_api_replicas_are_behind_nginx_least_conn():
    compose = read("docker-compose.yml")
    nginx = read("deploy/nginx/ims.conf")
    assert "api-1:" in compose and "api-2:" in compose
    assert '"127.0.0.1:8081:8080"' in compose
    assert '"127.0.0.1:8082:8080"' in compose
    assert "least_conn;" in nginx
    assert "server api-1:8080" in nginx
    assert "server api-2:8080" in nginx
    assert "proxy_buffering off;" in nginx


def test_additive_query_indexes_do_not_rewrite_embeddings():
    migration = read("migrations/versions/0003_query_performance_indexes.py")
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in migration
    assert "gin_trgm_ops" in migration
    assert "CREATE INDEX CONCURRENTLY" in migration
    assert "UPDATE chunks" not in migration
    assert "embedding =" not in migration


def test_role_coverage_uses_single_structural_sql_discovery_query():
    source = Path('src/ike/retrieval/engine.py').read_text(encoding='utf-8')
    assert 'role_structural' in source
    assert 'role_coverage_sql_queries' in source
    assert 'tsquery_text' in source


def test_api_image_contains_release_benchmark_tools():
    dockerfile = Path('docker/api.Dockerfile').read_text(encoding='utf-8')
    assert 'COPY scripts ./scripts' in dockerfile
    assert 'COPY eval ./eval' in dockerfile
