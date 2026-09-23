from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_primary_ask_ui_is_chat_composer_driven_and_reports_are_folded_into_ask():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="ask"' in html
    assert 'data-tab="history"' in html
    assert "My Questions" in html
    assert 'id="queryMode"' in html
    assert '<option value="auto">Auto</option>' in html
    assert '<option value="direct">Direct</option>' in html
    assert '<option value="research">Research</option>' in html
    assert '<option value="deep">Deep Analysis</option>' in html
    assert "Deep Analysis" in html
    assert 'data-tab="reports"' not in html
    assert 'id="modeGuide"' in html
    assert 'id="question" class="question-input"' in html
    assert 'class="ask-composer card"' in html
    assert 'id="queryScope"' in html
    assert 'id="choosePdfsBtn"' in html


def test_browser_streams_qa_auto_scrolls_progress_and_polls_deep_analysis_without_carryover():
    js = (ROOT / "src" / "ike" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "/api/v1/query/stream" in js
    assert "/api/v1/query/history" in js
    assert "/api/v1/reports" in js
    assert "startDeepAnalysis" in js
    assert "renderMarkdown" in js
    assert "container.appendChild(article)" in js
    assert "scrollConversationToBottom" in js
    assert "scrollConversationToNode" in js
    assert "Each question is answered independently" not in js  # the product note is template-only
    assert "historyItems" in js


def test_answer_first_sidebar_activity_drawer_and_pdf_search_are_present():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "ike" / "web" / "static" / "app.css").read_text(encoding="utf-8")
    js = (ROOT / "src" / "ike" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'class="workspace-sidebar"' in html
    assert 'id="activityDrawer" class="activity-drawer"' in html
    assert 'id="activityToggle"' in html
    assert 'class="insight-rail"' not in html
    assert 'id="dashboardStats"' in html
    assert 'id="recentQuestionList"' in html
    assert 'id="documentSearch"' in html
    assert 'id="personalDocumentSearch"' in html
    assert 'id="queryDocumentSearch"' in html
    assert 'class="document-scroll"' in html
    assert ".viewport-library" in css
    assert ".document-scroll" in css
    assert "renderFilteredDocuments" in js
    assert "/api/v1/dashboard/summary" in js
    assert "setActivityDrawer" in js
    assert "visualEvidenceHtml" in js
    assert "/api/v1/visuals/render/" in js


def test_mobile_theme_and_personal_library_controls_are_present():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "ike" / "web" / "static" / "app.css").read_text(encoding="utf-8")
    js = (ROOT / "src" / "ike" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'data-tab="library"' in html
    assert "My PDFs" in html
    assert 'id="themeToggle"' in html
    assert 'body[data-theme="dark"]' in css
    assert "@media(max-width:680px)" in css
    assert "localStorage.setItem('ike-theme'" in js
    assert "/api/v1/library" in js
    assert "/api/v1/auth/directory" in js


def test_admin_user_management_remains_separate_from_user_history():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "src" / "ike" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'data-tab="users"' in html
    assert "/api/v1/auth/users" in js
    assert "/api/v1/query/history" in js


def test_dmrc_ims_branding_is_visible_and_internal_namespace_is_not_required_in_ui():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    config = (ROOT / "src" / "ike" / "core" / "config.py").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "Incident Management System" in html
    assert "IMS" in html
    assert "organization_short_name" in html
    assert 'app_name: str = "IMS - Incident Management System"' in config
    assert 'organization_name: str = "Delhi Metro Rail Corporation (DMRC)"' in config
    assert "APP_NAME=IMS - Incident Management System" in env
    assert "ORGANIZATION_SHORT_NAME=DMRC" in env


def test_frontend_assets_are_release_versioned_to_prevent_mixed_ui_cache():
    html = (ROOT / "src" / "ike" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    main = (ROOT / "src" / "ike" / "main.py").read_text(encoding="utf-8")
    assert '/static/app.css?v={{ app_version }}' in html
    assert '/static/app.js?v={{ app_version }}' in html
    assert 'data-app-version="{{ app_version }}"' in html
    assert '"app_version": __version__' in main
    assert 'request.url.path == "/"' in main
    assert '"Cache-Control"] = "no-store, max-age=0"' in main
    assert 'request.url.path.startswith("/static/")' in main
    assert 'max-age=31536000, immutable' in main
