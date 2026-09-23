from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_summary_is_api_only_and_user_scoped():
    route = (ROOT / "src" / "ike" / "api" / "routes" / "dashboard.py").read_text(encoding="utf-8")
    main = (ROOT / "src" / "ike" / "main.py").read_text(encoding="utf-8")
    schema = (ROOT / "src" / "ike" / "schemas" / "dashboard.py").read_text(encoding="utf-8")

    assert 'router = APIRouter(prefix="/dashboard"' in route
    assert '@router.get("/summary"' in route
    assert "QueryLog.user_id == user.id" in route
    assert "ReportJob.created_by == user.id" in route
    assert "document_access_clause(user" in route
    assert 'Document.workspace_scope == "organization"' in route
    assert 'Document.workspace_scope == "personal"' in route
    assert ".limit(5)" in route
    assert "DashboardSummaryOut" in schema
    assert "recent_questions" in schema
    assert "app.include_router(dashboard.router" in main
