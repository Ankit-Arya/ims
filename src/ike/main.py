import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ike import __version__
from ike.api.routes import auth, dashboard, debug, documents, health, library, query, reports, visuals
from ike.core.config import get_settings
from ike.core.logging import configure_logging
from ike.services.bootstrap import ensure_bootstrap_admin

settings = get_settings()
configure_logging(settings.log_level)

REQUESTS = Counter("ike_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("ike_http_request_duration_seconds", "HTTP request latency", ["method", "path"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_root.mkdir(parents=True, exist_ok=True)
    ensure_bootstrap_admin()
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    route_path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
    LATENCY.labels(request.method, route_path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # The HTML shell must never be reused across UI releases. Static assets are
    # versioned in index.html, so each release gets a unique cache key while
    # still allowing the browser/proxy to cache that immutable asset URL.
    if request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    elif request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

    return response


app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(library.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(visuals.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(debug.router, prefix="/api/v1")

web_root = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=web_root / "static"), name="static")
templates = Jinja2Templates(directory=web_root / "templates")


@app.get("/", include_in_schema=False)
def web_app(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "app_version": __version__,
            "organization_name": settings.organization_name,
            "organization_short_name": settings.organization_short_name,
            "report_max_documents": settings.report_max_documents,
        },
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
