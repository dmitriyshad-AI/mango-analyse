from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from mango_mvp.productization.product_api import ProductApiFacade, build_product_api_readiness_report, read_only_actions
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


PRODUCT_API_HTTP_SCHEMA_VERSION = "product_api_http_readonly_v1"


@dataclass(frozen=True)
class ProductApiHttpSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    routes: int
    read_only: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


RouteResult = tuple[int, Mapping[str, Any]]


def route_product_api_request(api: ProductApiFacade, method: str, raw_path: str) -> RouteResult:
    method = clean(method).upper()
    parsed = urlparse(raw_path)
    route = parsed.path.rstrip("/") or "/"
    query = parse_qs(parsed.query)
    if method != "GET":
        return blocked_mutation_response(method=method, route=route)
    if route == "/health":
        return 200, {
            "schema_version": PRODUCT_API_HTTP_SCHEMA_VERSION,
            "route": route,
            "ok": True,
            "read_only": True,
            "actions": read_only_actions(),
        }
    route_map: Mapping[str, Callable[[], Mapping[str, Any]]] = {
        "/dashboard/summary": api.dashboard_summary,
        "/capture/recent": lambda: api.capture_recent(limit=query_limit(query, default=50)),
        "/queues/processing": lambda: api.processing_queue(limit=query_limit(query, default=50)),
        "/scheduler/runs": lambda: api.scheduler_runs(limit=query_limit(query, default=50)),
        "/asr/gates": api.asr_gate_status,
        "/writeback/previews": api.writeback_previews,
        "/knowledge/playbook": api.knowledge_playbook,
        "/settings/adapters": api.settings_adapters,
    }
    handler = route_map.get(route)
    if handler is None:
        return 404, {
            "schema_version": PRODUCT_API_HTTP_SCHEMA_VERSION,
            "route": route,
            "error": "route_not_found",
            "implemented_routes": sorted(route_map),
            "actions": read_only_actions(),
        }
    try:
        payload = handler()
    except ValueError as exc:
        return 400, {
            "schema_version": PRODUCT_API_HTTP_SCHEMA_VERSION,
            "route": route,
            "method": "GET",
            "error": "bad_request",
            "detail": str(exc),
            "actions": read_only_actions(),
        }
    return 200, {
        "schema_version": PRODUCT_API_HTTP_SCHEMA_VERSION,
        "route": route,
        "method": "GET",
        "payload": payload,
        "actions": read_only_actions(),
    }


def build_product_api_http_readiness_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_api_http_paths(product_root=product_root, product_db_path=product_db_path, out_path=out_path)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)
    api_readiness = build_product_api_readiness_report(
        product_root=product_root,
        product_db_path=product_db_path,
        workspace_root=workspace_root,
    )
    route_checks = {
        route: route_product_api_request(api, "GET", route)[0]
        for route in PRODUCT_API_HTTP_ROUTES
    }
    mutation_check = route_product_api_request(api, "POST", "/capture/recent")
    blocked = 0 if all(status == 200 for status in route_checks.values()) and mutation_check[0] == 405 else 1
    validation_ok = bool(api_readiness["summary"].get("validation_ok")) and blocked == 0
    report = {
        "summary": ProductApiHttpSummary(
            schema_version=PRODUCT_API_HTTP_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            routes=len(PRODUCT_API_HTTP_ROUTES),
            read_only=True,
            validation_ok=validation_ok,
            blocked=blocked,
            warnings=int(api_readiness["summary"].get("warnings") or 0),
        ).to_json_dict()
        | {
            "report_generated_ok": True,
            "api_readiness_ok": bool(api_readiness["summary"].get("validation_ok")),
        },
        "routes": {
            "implemented": list(PRODUCT_API_HTTP_ROUTES),
            "checks": route_checks,
            "health": "/health",
        },
        "blocked_mutation_check": {
            "status": mutation_check[0],
            "payload": mutation_check[1],
        },
        "api_readiness": api_readiness,
        "safety": read_only_actions(),
    }
    if out_path:
        write_json(out_path, report)
    return report


PRODUCT_API_HTTP_ROUTES = (
    "/dashboard/summary",
    "/capture/recent",
    "/queues/processing",
    "/scheduler/runs",
    "/asr/gates",
    "/writeback/previews",
    "/knowledge/playbook",
    "/settings/adapters",
)


def make_product_api_handler(product_root: Path, product_db_path: Path, workspace_root: Optional[Path] = None) -> type[BaseHTTPRequestHandler]:
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)

    class ProductApiHandler(BaseHTTPRequestHandler):
        server_version = "MangoProductApiReadonly/0.1"

        def do_GET(self) -> None:  # noqa: N802
            status, payload = route_product_api_request(api, "GET", self.path)
            self._write_json(status, payload)

        def do_POST(self) -> None:  # noqa: N802
            status, payload = route_product_api_request(api, "POST", self.path)
            self._write_json(status, payload)

        def do_PUT(self) -> None:  # noqa: N802
            status, payload = route_product_api_request(api, "PUT", self.path)
            self._write_json(status, payload)

        def do_PATCH(self) -> None:  # noqa: N802
            status, payload = route_product_api_request(api, "PATCH", self.path)
            self._write_json(status, payload)

        def do_DELETE(self) -> None:  # noqa: N802
            status, payload = route_product_api_request(api, "DELETE", self.path)
            self._write_json(status, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ProductApiHandler


def run_product_api_http_server(
    product_root: Path,
    product_db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace_root: Optional[Path] = None,
) -> None:
    handler = make_product_api_handler(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.serve_forever()


def blocked_mutation_response(method: str, route: str) -> RouteResult:
    return 405, {
        "schema_version": PRODUCT_API_HTTP_SCHEMA_VERSION,
        "route": route,
        "method": method,
        "error": "mutations_blocked_in_read_only_product_api",
        "allowed_methods": ["GET"],
        "blocked_actions": ["download_audio", "run_asr", "run_ra", "write_crm", "write_runtime_db"],
        "actions": read_only_actions(),
    }


def query_limit(query: Mapping[str, Sequence[str]], default: int) -> int:
    raw = (query.get("limit") or [str(default)])[0]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc
    if value < 1:
        raise ValueError("limit must be positive")
    return min(value, 500)


def guard_product_api_http_paths(product_root: Path, product_db_path: Path, out_path: Optional[Path]) -> None:
    for label, path in (("product root", product_root), ("product DB", product_db_path), ("Product API HTTP audit", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"Product API HTTP audit must stay under product root: {product_root}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
