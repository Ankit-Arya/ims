from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_query_history_is_user_scoped():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "query.py").read_text(encoding="utf-8")
    assert ".where(QueryLog.user_id == user.id)" in source
    assert "query_log.user_id != user.id" in source


def test_reports_are_user_scoped_even_for_admin_normal_product_routes():
    source = (ROOT / "src" / "ike" / "api" / "routes" / "reports.py").read_text(encoding="utf-8")
    assert ".where(ReportJob.created_by == user.id)" in source
    assert "job.created_by != user.id" in source
    assert 'user.role == "admin"' not in source
