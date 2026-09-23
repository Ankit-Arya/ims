from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ingestion_worker_uses_cpu_and_memory_aware_prefork_concurrency():
    source = (ROOT / "src" / "ike" / "tasks" / "worker_bootstrap.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "ingest_cpu_target_percent" in source
    assert "memory_limit" in source
    assert '"--pool=prefork"' in source
    assert 'command: ["python", "-m", "ike.tasks.worker_bootstrap"]' in compose


def test_tesseract_threads_are_bounded_without_globally_limiting_docling():
    dockerfile = (ROOT / "docker" / "worker.Dockerfile").read_text(encoding="utf-8")
    pipeline = (ROOT / "src" / "ike" / "ingestion" / "docling_pipeline.py").read_text(encoding="utf-8")
    assert "TESSERACT_OMP_THREAD_LIMIT" in dockerfile
    assert "ThreadedPdfPipelineOptions" in pipeline
    assert "AcceleratorOptions" in pipeline
    assert "docling_num_threads_per_document" in pipeline
