from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
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
        "/dashboard/appliance": lambda: api.appliance_dashboard(
            capture_limit=query_positive_int(query, "capture_limit", default=query_limit(query, default=25)),
            scheduler_limit=query_positive_int(query, "scheduler_limit", default=query_limit(query, default=25)),
            capture_status=query_scalar(query, "capture_status", default=""),
            manager_ref=query_scalar(query, "manager_ref", default=""),
            q=query_scalar(query, "q", default=""),
            scheduler_status=query_scalar(query, "scheduler_status", default=""),
            scheduler_job_type=query_scalar(query, "scheduler_job_type", default=""),
        ),
        "/dashboard/summary": api.dashboard_summary,
        "/capture/recent": lambda: api.capture_recent(
            limit=query_limit(query, default=50),
            status=query_scalar(query, "status", default=""),
            manager_ref=query_scalar(query, "manager_ref", default=""),
            q=query_scalar(query, "q", default=""),
        ),
        "/queues/processing": lambda: api.processing_queue(limit=query_limit(query, default=50)),
        "/scheduler/runs": lambda: api.scheduler_runs(
            limit=query_limit(query, default=50),
            status=query_scalar(query, "status", default=""),
            job_type=query_scalar(query, "job_type", default=""),
        ),
        "/scheduler/health": api.scheduler_health,
        "/scheduler/control-plane": api.scheduler_control_plane,
        "/processing/lifecycle": api.lifecycle_readiness,
        "/crm/mapping-preview": lambda: api.crm_mapping_preview(limit=query_limit(query, default=50)),
        "/asr/gates": api.asr_gate_status,
        "/writeback/previews": lambda: api.writeback_previews(
            limit=query_limit(query, default=25),
            stage=query_scalar(query, "stage", default="batch_10"),
        ),
        "/operator/status": api.operator_runtime_status,
        "/manual-resolution/status": api.manual_resolution_status,
        "/waiting-work/status": api.waiting_autonomous_work_status,
        "/writeback/dry-run-readiness": api.writeback_dry_run_readiness,
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
    "/dashboard/appliance",
    "/dashboard/summary",
    "/capture/recent",
    "/queues/processing",
    "/scheduler/runs",
    "/scheduler/health",
    "/scheduler/control-plane",
    "/processing/lifecycle",
    "/crm/mapping-preview",
    "/asr/gates",
    "/writeback/previews",
    "/operator/status",
    "/manual-resolution/status",
    "/waiting-work/status",
    "/writeback/dry-run-readiness",
    "/knowledge/playbook",
    "/settings/adapters",
)


def make_product_api_handler(product_root: Path, product_db_path: Path, workspace_root: Optional[Path] = None) -> type[BaseHTTPRequestHandler]:
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)

    class ProductApiHandler(BaseHTTPRequestHandler):
        server_version = "MangoProductApiReadonly/0.1"

        def do_GET(self) -> None:  # noqa: N802
            route = (urlparse(self.path).path.rstrip("/") or "/")
            if route in {"/", "/dashboard"}:
                self._write_html(200, render_appliance_dashboard_html())
                return
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

        def _write_html(self, status: int, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
    return query_positive_int(query, "limit", default=default)


def query_positive_int(query: Mapping[str, Sequence[str]], name: str, default: int) -> int:
    raw = (query.get(name) or [str(default)])[0]
    label = clean(name) or "limit"
    if label not in {"limit", "capture_limit", "scheduler_limit"}:
        raise ValueError("unsupported integer query parameter")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return min(value, 500)


def query_scalar(query: Mapping[str, Sequence[str]], name: str, default: str) -> str:
    values = query.get(name)
    if not values:
        return default
    return clean(values[0]) or default


def render_appliance_dashboard_html() -> str:
    title = "Mango Analyse Appliance Dashboard"
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-soft: #eef3f8;
      --text: #17202a;
      --muted: #637083;
      --border: #d8e0e8;
      --accent: #2563eb;
      --ok: #16855b;
      --warn: #b7791f;
      --danger: #b42318;
      --shadow: 0 8px 24px rgba(29, 41, 57, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(245, 247, 250, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }}
    .bar {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .brand h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .brand p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      min-height: 36px;
      padding: 8px 12px;
      border-radius: 7px;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    button[aria-selected="true"] {{
      background: #17202a;
      border-color: #17202a;
      color: #fff;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 24px 40px;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric, .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{
      min-height: 92px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .metric strong {{
      font-size: 24px;
      line-height: 1;
    }}
    .metric small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 18px 0;
    }}
    .filters {{
      display: grid;
      grid-template-columns: minmax(220px, 1.2fr) repeat(4, minmax(140px, 1fr)) auto;
      gap: 10px;
      align-items: end;
      margin: 18px 0 10px;
      padding: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    input, select {{
      min-height: 36px;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 7px 9px;
      background: #fff;
      color: var(--text);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
      gap: 16px;
    }}
    .panel {{
      overflow: hidden;
      min-width: 0;
    }}
    .panel header {{
      position: static;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    .panel-title {{
      padding: 14px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .panel-title h2 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
    }}
    .panel-title p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: var(--surface-soft);
      color: var(--muted);
      white-space: nowrap;
    }}
    .badge.ok {{ background: #e7f6ef; color: var(--ok); }}
    .badge.warn {{ background: #fff5db; color: var(--warn); }}
    .badge.danger {{ background: #fee4e2; color: var(--danger); }}
    .table-wrap {{ overflow: auto; max-height: 520px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fafbfd;
    }}
    td.wrap {{
      white-space: normal;
      min-width: 220px;
    }}
    .side {{
      display: grid;
      gap: 16px;
    }}
    .list {{
      padding: 8px 16px 16px;
      display: grid;
      gap: 10px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}
    .row strong {{ color: var(--text); text-align: right; }}
    pre {{
      margin: 0;
      max-height: 420px;
      overflow: auto;
      padding: 14px;
      background: #111827;
      color: #e5e7eb;
      font-size: 12px;
      line-height: 1.45;
    }}
    .hidden {{ display: none; }}
    .empty {{
      padding: 26px 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .status-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .filters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .bar {{ align-items: flex-start; flex-direction: column; padding: 14px 16px; }}
      main {{ padding: 16px; }}
      .status-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filters {{ grid-template-columns: 1fr; }}
      th, td {{ padding: 9px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="brand">
        <h1>Mango Analyse Appliance</h1>
        <p>Read-only product dashboard over Product API</p>
      </div>
      <div class="toolbar">
        <span id="mode" class="badge">loading</span>
        <button id="refresh" class="primary" type="button">Refresh</button>
      </div>
    </div>
  </header>
  <main>
    <section id="metrics" class="status-grid" aria-live="polite"></section>
    <form id="filters" class="filters">
      <label>Search
        <input id="filter-q" name="q" type="search" autocomplete="off" placeholder="phone, call id, recording">
      </label>
      <label>Manager
        <input id="filter-manager" name="manager_ref" type="text" autocomplete="off" placeholder="101">
      </label>
      <label>Capture status
        <select id="filter-capture-status" name="capture_status">
          <option value="">All</option>
          <option value="ready_for_capture">Ready</option>
          <option value="blocked_no_recording">No recording</option>
          <option value="blocked_duplicate">Duplicate</option>
        </select>
      </label>
      <label>Job status
        <select id="filter-scheduler-status" name="scheduler_status">
          <option value="">All</option>
          <option value="planned">Planned</option>
          <option value="retry_wait">Retry wait</option>
          <option value="running">Running</option>
          <option value="succeeded">Succeeded</option>
          <option value="failed">Failed</option>
        </select>
      </label>
      <label>Job type
        <input id="filter-job-type" name="scheduler_job_type" type="text" autocomplete="off" placeholder="shadow_poll">
      </label>
      <button class="primary" type="submit">Apply</button>
    </form>
    <nav class="tabs" aria-label="Dashboard tabs">
      <button type="button" data-tab="capture" aria-selected="true">Capture</button>
      <button type="button" data-tab="scheduler">Scheduler</button>
      <button type="button" data-tab="lifecycle">Lifecycle</button>
      <button type="button" data-tab="writeback">Writeback</button>
      <button type="button" data-tab="waiting_work">Waiting work</button>
      <button type="button" data-tab="safety">Safety</button>
      <button type="button" data-tab="demo_readiness">Demo</button>
      <button type="button" data-tab="knowledge">Knowledge</button>
      <button type="button" data-tab="settings">Settings</button>
    </nav>
    <section class="grid">
      <div id="primary" class="panel"></div>
      <aside class="side">
        <section class="panel">
          <header><div class="panel-title"><div><h2>Safety gates</h2><p>Actions blocked by default</p></div><span class="badge ok">read-only</span></div></header>
          <div id="safety" class="list"></div>
        </section>
        <section class="panel">
          <header><div class="panel-title"><div><h2>Selected payload</h2><p>Raw JSON for current tab</p></div></div></header>
          <pre id="payload">{{}}</pre>
        </section>
      </aside>
    </section>
  </main>
  <script>
    const state = {{ data: null, tab: "capture", filters: {{}} }};
    const $ = (selector) => document.querySelector(selector);
    const fmt = (value) => value === null || value === undefined || value === "" ? "—" : value;
    const badgeClass = (value) => value === true || value === "ready" || value === "preview_only" ? "ok" : value ? "warn" : "danger";

    async function loadDashboard() {{
      $("#mode").textContent = "loading";
      $("#mode").className = "badge";
      const params = new URLSearchParams({{ capture_limit: "25", scheduler_limit: "25" }});
      Object.keys(state.filters).forEach((key) => {{
        if (state.filters[key]) params.set(key, state.filters[key]);
      }});
      const response = await fetch("/dashboard/appliance?" + params.toString(), {{ cache: "no-store" }});
      if (!response.ok) throw new Error("Dashboard request failed: " + response.status);
      const envelope = await response.json();
      state.data = envelope.payload || envelope;
      $("#mode").textContent = state.data.status || "ready";
      $("#mode").className = "badge " + badgeClass(state.data.status);
      render();
    }}

    function renderMetrics(summary) {{
      const metrics = [
        ["Validation", summary.validation_ok ? "OK" : "Blocked", "blocked " + summary.blocked],
        ["Product calls", summary.product_calls, "tenant rows " + summary.tenants],
        ["Capture ready", summary.capture_inbox_ready, "items " + summary.capture_inbox_items],
        ["Capture blocked", summary.capture_inbox_blocked, "warnings " + summary.warnings],
        ["Job runs", summary.job_runs, "due " + fmt(summary.scheduler_due_jobs)],
        ["Scheduler failed", summary.scheduler_failed_jobs, "needs review"],
        ["Owner review", summary.pending_owner_mappings, "pending mappings"],
        ["Manager tasks", summary.manager_task_total_rows, "manual CRM tasks"],
        ["Duplicate recheck", summary.operator_duplicate_recheck_passed ? "PASS" : "BLOCK", "blocked " + fmt(summary.operator_duplicate_recheck_blocked_rows)],
        ["Waiting dry-run", summary.operator_waiting_work_dry_run_prepared_rows, "refresh " + fmt(summary.operator_waiting_work_refresh_candidate_rows)],
        ["Readback missing", summary.operator_waiting_work_readback_missing_rows, "waiting work"],
        ["Stage50", summary.stage50_preflight_allowed ? "allowed" : "blocked", "writeback rollout"],
        ["Stage86", summary.stage86_preflight_allowed ? "allowed" : "blocked", "writeback rollout"],
        ["Cleanup", summary.cleanup_candidate_rows, "safe " + fmt(summary.cleanup_safe_to_quarantine_rows)],
        ["Demo data", summary.demo_snapshot_files, "artifacts " + fmt(summary.demo_artifacts)],
      ];
      $("#metrics").innerHTML = metrics.map(([label, value, hint]) => `
        <article class="metric">
          <span>${{label}}</span>
          <strong>${{fmt(value)}}</strong>
          <small>${{hint}}</small>
        </article>
      `).join("");
    }}

    function renderTable(title, subtitle, rows, columns) {{
      const body = rows.length ? `
        <div class="table-wrap"><table>
          <thead><tr>${{columns.map((column) => `<th>${{column.label}}</th>`).join("")}}</tr></thead>
          <tbody>${{rows.map((row) => `<tr>${{columns.map((column) => `<td class="${{column.wrap ? "wrap" : ""}}">${{fmt(row[column.key])}}</td>`).join("")}}</tr>`).join("")}}</tbody>
        </table></div>
      ` : `<div class="empty">No rows yet. The API is healthy, but this product DB has no records for this panel.</div>`;
      $("#primary").innerHTML = `
        <header><div class="panel-title"><div><h2>${{title}}</h2><p>${{subtitle}}</p></div><span class="badge">${{rows.length}} rows</span></div></header>
        ${{body}}
      `;
    }}

    function renderSide(data) {{
      const safety = data.safety || {{}};
      const rows = [
        ["Run ASR", safety.run_asr],
        ["Run R+A", safety.run_ra],
        ["Write runtime DB", safety.write_runtime_db],
        ["Write CRM", safety.write_crm],
        ["Write Tallanto", safety.write_tallanto],
      ];
      $("#safety").innerHTML = rows.map(([label, value]) => `
        <div class="row"><span>${{label}}</span><strong>${{value ? "allowed" : "blocked"}}</strong></div>
      `).join("");
    }}

    function render() {{
      const data = state.data;
      if (!data) return;
      renderMetrics(data.summary || {{}});
      renderSide(data);
      const panels = data.panels || {{}};
      const selectedPayload = panels[state.tab] || panels.dashboard || data;
      $("#payload").textContent = JSON.stringify(selectedPayload, null, 2);
      if (state.tab === "capture") {{
        const filters = data.filters || {{}};
        const subtitle = ["Recent Mango events normalized for product mode"];
        if (filters.q) subtitle.push("search " + filters.q);
        if (filters.manager_ref) subtitle.push("manager " + filters.manager_ref);
        if (filters.capture_status) subtitle.push("status " + filters.capture_status);
        renderTable("Capture inbox", subtitle.join(" | "), (panels.capture || {{}}).items || [], [
          {{ key: "status", label: "Status" }},
          {{ key: "event_key", label: "Event key", wrap: true }},
          {{ key: "started_at", label: "Started" }},
          {{ key: "client_phone", label: "Phone" }},
          {{ key: "manager_ref", label: "Manager" }},
          {{ key: "recording_ref", label: "Recording", wrap: true }},
          {{ key: "enqueue_count", label: "Seen" }},
        ]);
      }} else if (state.tab === "scheduler") {{
        const health = ((panels.scheduler || {{}}).health || {{}}).summary || {{}};
        const control = ((panels.scheduler || {{}}).control_plane || {{}});
        const controlSummary = control.summary || {{}};
        renderTable("Scheduler runs", "Latest product job rows", (panels.scheduler || {{}}).items || [], [
          {{ key: "status", label: "Status" }},
          {{ key: "job_type", label: "Job type" }},
          {{ key: "tenant_id", label: "Tenant" }},
          {{ key: "planned_at", label: "Planned" }},
          {{ key: "next_run_at", label: "Next run" }},
          {{ key: "output_ref", label: "Output", wrap: true }},
        ]);
        $("#primary").insertAdjacentHTML("beforeend", `<div class="list">
          <div class="row"><span>Due jobs</span><strong>${{fmt(health.due_jobs)}}</strong></div>
          <div class="row"><span>Failed jobs</span><strong>${{fmt(health.failed_jobs)}}</strong></div>
          <div class="row"><span>Stale running</span><strong>${{fmt(health.stale_running_jobs)}}</strong></div>
          <div class="row"><span>Locked jobs</span><strong>${{fmt(health.locked_jobs)}}</strong></div>
          <div class="row"><span>Control actions</span><strong>${{fmt(controlSummary.recommended_actions)}}</strong></div>
        </div>`);
      }} else if (state.tab === "lifecycle") {{
        const lifecycle = panels.lifecycle || {{}};
        const summary = (((lifecycle.report || {{}}).summary) || {{}});
        renderTable("Processing lifecycle", "Read-only capture-to-handoff readiness", [
          {{ name: "Mode", value: lifecycle.mode }},
          {{ name: "Report", value: lifecycle.report_path }},
          {{ name: "Candidates", value: summary.candidate_asr_handoff_dry_run }},
          {{ name: "Waiting assets", value: summary.wait_recording_asset }},
          {{ name: "Blocked", value: summary.blocked }},
          {{ name: "Auto trigger", value: "disabled" }},
        ], [
          {{ key: "name", label: "Check" }},
          {{ key: "value", label: "Value", wrap: true }},
        ]);
      }} else if (state.tab === "writeback") {{
        const writeback = panels.writeback || {{}};
        const mapping = panels.crm_mapping || {{}};
        const previewSummary = (((writeback.preview || {{}}).summary) || {{}});
        const resolutionSummary = (((writeback.preview || {{}}).crm_resolution || {{}}).summary || {{}});
        const mappingSummary = mapping.summary || {{}};
        renderTable("Writeback readiness", "Preview-only CRM policy", [
          {{ name: "Mode", value: writeback.current_mode }},
          {{ name: "Write CRM", value: writeback.write_crm }},
          {{ name: "Write Tallanto", value: writeback.write_tallanto }},
          {{ name: "Preview ready", value: previewSummary.preview_ready }},
          {{ name: "Missing CRM entity", value: previewSummary.blocked_missing_crm_entity }},
          {{ name: "Resolved entities", value: resolutionSummary.resolve_crm_entity }},
          {{ name: "AMO mapping", value: fmt(mappingSummary.amo_resolved) + " resolved / " + fmt(mappingSummary.amo_missing) + " missing" }},
          {{ name: "Tallanto mapping", value: fmt(mappingSummary.tallanto_resolved) + " resolved / " + fmt(mappingSummary.tallanto_missing) + " missing" }},
          {{ name: "Blocked reasons", value: (writeback.blocked_reasons || []).join(", ") || "none" }},
          {{ name: "Required sequence", value: (writeback.required_sequence || []).join(" -> ") }},
        ], [
          {{ key: "name", label: "Check" }},
          {{ key: "value", label: "Value", wrap: true }},
        ]);
      }} else if (state.tab === "waiting_work") {{
        const waiting = panels.waiting_autonomous_work || {{}};
        const summary = waiting.summary || {{}};
        renderTable("Waiting work", "Safe dry-run/readback work while staff merges duplicates", [
          {{ name: "Status", value: summary.status }},
          {{ name: "Dry-run when tunnel available", value: summary.dry_run_allowed_when_tunnel_available }},
          {{ name: "Live write now", value: summary.live_write_allowed_now }},
          {{ name: "Non-duplicate candidates", value: summary.non_duplicate_live_candidate_rows }},
          {{ name: "Refresh candidates", value: summary.refresh_candidate_rows }},
          {{ name: "Missing readback rows", value: summary.readback_missing_rows }},
          {{ name: "Contact-id mismatch rows", value: summary.contact_id_mismatch_rows }},
          {{ name: "Required sequence", value: (waiting.required_sequence || []).join(" -> ") }},
        ], [
          {{ key: "name", label: "Check" }},
          {{ key: "value", label: "Value", wrap: true }},
        ]);
      }} else if (state.tab === "safety") {{
        const safety = data.safety || {{}};
        renderTable("Safety gates", "Live actions blocked unless explicit guarded workflow is used", Object.keys(safety).map((key) => ({{
          name: key,
          value: Array.isArray(safety[key]) ? safety[key].join(", ") : safety[key],
        }})), [
          {{ key: "name", label: "Gate" }},
          {{ key: "value", label: "State", wrap: true }},
        ]);
      }} else if (state.tab === "demo_readiness") {{
        const demo = panels.demo_readiness || {{}};
        const summary = demo.summary || {{}};
        renderTable("Demo readiness", "Real-data demo contract without exposing personal data", [
          {{ name: "Product DB present", value: summary.product_db_present }},
          {{ name: "Required panels", value: fmt(summary.panels_present) + " / " + fmt(summary.required_panels) }},
          {{ name: "Snapshot files", value: summary.snapshot_files }},
          {{ name: "Snapshot entities", value: summary.snapshot_entities }},
          {{ name: "Demo artifacts", value: summary.demo_artifacts }},
          {{ name: "Missing panels", value: (demo.missing_panels || []).join(", ") || "none" }},
          {{ name: "Warnings", value: (demo.warning_reasons || []).join(", ") || "none" }},
          {{ name: "Blocked", value: (demo.blocked_reasons || []).join(", ") || "none" }},
        ], [
          {{ key: "name", label: "Check" }},
          {{ key: "value", label: "Value", wrap: true }},
        ]);
      }} else if (state.tab === "knowledge") {{
        const knowledge = panels.knowledge || {{}};
        renderTable("Knowledge readiness", "Current playbook contract state", [
          {{ name: "Mode", value: knowledge.current_mode }},
          {{ name: "Entities", value: (knowledge.entities || []).join(", ") }},
          {{ name: "Blocked reasons", value: (knowledge.blocked_reasons || []).join(", ") || "none" }},
        ], [
          {{ key: "name", label: "Check" }},
          {{ key: "value", label: "Value", wrap: true }},
        ]);
      }} else {{
        const adapters = ((panels.settings || {{}}).adapters || {{}});
        renderTable("Settings", "Adapter and database policy without secret values", Object.keys(adapters).map((key) => ({{
          name: key,
          value: JSON.stringify(adapters[key]),
        }})), [
          {{ key: "name", label: "Adapter" }},
          {{ key: "value", label: "Policy", wrap: true }},
        ]);
      }}
    }}

    document.querySelectorAll("[data-tab]").forEach((button) => {{
      button.addEventListener("click", () => {{
        state.tab = button.dataset.tab;
        document.querySelectorAll("[data-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
        render();
      }});
    }});
    $("#filters").addEventListener("submit", (event) => {{
      event.preventDefault();
      state.filters = {{
        q: $("#filter-q").value.trim(),
        manager_ref: $("#filter-manager").value.trim(),
        capture_status: $("#filter-capture-status").value,
        scheduler_status: $("#filter-scheduler-status").value,
        scheduler_job_type: $("#filter-job-type").value.trim(),
      }};
      loadDashboard().catch((error) => {{
        $("#mode").textContent = "error";
        $("#mode").className = "badge danger";
        $("#primary").innerHTML = `<div class="empty">${{error.message}}</div>`;
      }});
    }});
    $("#refresh").addEventListener("click", () => loadDashboard().catch((error) => {{
      $("#mode").textContent = "error";
      $("#mode").className = "badge danger";
      $("#primary").innerHTML = `<div class="empty">${{error.message}}</div>`;
    }}));
    loadDashboard().catch((error) => {{
      $("#mode").textContent = "error";
      $("#mode").className = "badge danger";
      $("#primary").innerHTML = `<div class="empty">${{error.message}}</div>`;
    }});
  </script>
</body>
</html>"""


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
