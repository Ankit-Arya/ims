from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_ROUTES = ROOT / "src" / "ike" / "api" / "routes"


def test_api_routes_do_not_import_worker_implementations() -> None:
    forbidden = ("ike.tasks.jobs", "ike.ingestion.", "docling", "docling_core")
    offenders: list[str] = []
    for source in API_ROUTES.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(source.name)
    assert not offenders, f"API/worker dependency boundary violated by: {offenders}"


def test_task_dispatch_uses_stable_named_tasks() -> None:
    dispatch = (ROOT / "src" / "ike" / "services" / "task_dispatch.py").read_text(encoding="utf-8")
    worker = (ROOT / "src" / "ike" / "tasks" / "jobs.py").read_text(encoding="utf-8")
    for task_name in ("ike.ingest_document", "ike.build_report"):
        assert task_name in dispatch
        assert f'name="{task_name}"' in worker
