from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_ROUTES = ROOT / "src" / "ike" / "api" / "routes"


def test_api_routes_do_not_import_worker_implementations() -> None:
    forbidden = (
        "ike.tasks.ingestion",
        "ike.tasks.reports",
        "ike.ingestion.",
        "docling",
        "docling_core",
    )
    offenders: list[str] = []
    for source in API_ROUTES.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(source.name)
    assert not offenders, f"API/worker dependency boundary violated by: {offenders}"


def test_task_dispatch_uses_stable_named_tasks() -> None:
    dispatch = (ROOT / "src" / "ike" / "services" / "task_dispatch.py").read_text(encoding="utf-8")
    ingestion = (ROOT / "src" / "ike" / "tasks" / "ingestion.py").read_text(encoding="utf-8")
    reports = (ROOT / "src" / "ike" / "tasks" / "reports.py").read_text(encoding="utf-8")

    assert "ike.ingest_document" in dispatch
    assert 'name="ike.ingest_document"' in ingestion
    assert "ike.build_report" in dispatch
    assert 'name="ike.build_report"' in reports


def test_each_worker_imports_only_its_own_task_module() -> None:
    bootstrap = (ROOT / "src" / "ike" / "tasks" / "worker_bootstrap.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '"ike.tasks.ingestion"' in bootstrap
    report_block = compose.split("report-worker:", 1)[1]
    assert '"ike.tasks.reports"' in report_block
    assert '"ike.tasks.ingestion"' not in report_block
