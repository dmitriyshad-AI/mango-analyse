from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.read_api import (
    CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    customer_timeline_read_api_safety_contract,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path


CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION = "customer_timeline_approval_workspace_v1"


@dataclass(frozen=True)
class CustomerTimelineApprovalWorkspaceConfig:
    timeline_db: Path
    allowed_root: Path
    out_json: Optional[Path] = None
    out_html: Optional[Path] = None

    def __post_init__(self) -> None:
        read_config = CustomerTimelineReadApiConfig(timeline_db=self.timeline_db, allowed_root=self.allowed_root)
        out_json = guard_workspace_output_path(self.out_json, read_config) if self.out_json else None
        out_html = guard_workspace_output_path(self.out_html, read_config) if self.out_html else None
        object.__setattr__(self, "timeline_db", read_config.timeline_db)
        object.__setattr__(self, "allowed_root", read_config.allowed_root)
        object.__setattr__(self, "out_json", out_json)
        object.__setattr__(self, "out_html", out_html)


def build_customer_timeline_approval_workspace(
    *,
    config: CustomerTimelineApprovalWorkspaceConfig,
    tenant_id: str,
    customer_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 25,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    read_config = CustomerTimelineReadApiConfig(timeline_db=config.timeline_db, allowed_root=config.allowed_root)
    with CustomerTimelineReadApi.open(read_config) as api:
        health = api.health()
        summary = api.summary(tenant_id, recent_limit=limit)
        customer_search = api.list_customers(tenant_id, q=query, limit=limit)
        selected_customer_id = customer_id or first_customer_id(customer_search)
        selected_profile = api.customer_profile(tenant_id, selected_customer_id, event_limit=limit, bot_context_limit=limit) if selected_customer_id else None
        search = api.search(tenant_id, query, customer_id=selected_customer_id, limit=limit) if query else None
        conflicts = (
            selected_profile["conflicts"]
            if selected_profile and selected_profile.get("found")
            else api.list_conflicts(tenant_id, limit=limit)
        )

    readiness = approval_readiness(selected_profile, conflicts)
    workspace = {
        "schema_version": CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION,
        "read_api_schema_version": CUSTOMER_TIMELINE_READ_API_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "tenant_id": tenant_id,
        "inputs": {
            "timeline_db": str(config.timeline_db),
            "allowed_root": str(config.allowed_root),
            "customer_id": customer_id,
            "query": query,
            "limit": limit,
        },
        "summary": {
            "validation_ok": bool(health.get("validation_ok")),
            "status": readiness["status"],
            "selected_customer_id": selected_customer_id,
            "selected_customer_found": bool(selected_profile and selected_profile.get("found")),
            "customers_visible": len(customer_search.get("items") or ()),
            "open_conflicts": readiness["open_conflicts"],
            "bot_allowed_chunks": readiness["bot_allowed_chunks"],
            "bot_review_required_chunks": readiness["bot_review_required_chunks"],
            "live_actions_available": False,
            "warnings": readiness["warnings"],
            "blocked": readiness["blocked"],
        },
        "panels": {
            "health": health,
            "tenant_summary": summary,
            "customer_search": customer_search,
            "selected_customer": selected_profile,
            "timeline": selected_profile.get("timeline") if selected_profile else None,
            "bot_context": selected_profile.get("bot_context") if selected_profile else None,
            "search": search,
            "conflicts": conflicts,
            "safety_gates": safety_gates(),
        },
        "review_queue": build_review_queue(selected_profile, conflicts),
        "actions": workspace_actions(),
        "safety": customer_timeline_read_api_safety_contract(),
    }
    workspace["validation_ok"] = bool(workspace["summary"]["validation_ok"])
    if config.out_json:
        config.out_json.parent.mkdir(parents=True, exist_ok=True)
        config.out_json.write_text(json.dumps(workspace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.out_html:
        config.out_html.parent.mkdir(parents=True, exist_ok=True)
        config.out_html.write_text(render_customer_timeline_approval_workspace_html(workspace), encoding="utf-8")
    return workspace


def render_customer_timeline_approval_workspace_html(workspace: Mapping[str, Any]) -> str:
    summary = workspace.get("summary") or {}
    panels = workspace.get("panels") or {}
    profile = panels.get("selected_customer") if isinstance(panels.get("selected_customer"), Mapping) else None
    customer = profile.get("customer") if isinstance(profile, Mapping) and isinstance(profile.get("customer"), Mapping) else {}
    customer_search = panels.get("customer_search") if isinstance(panels.get("customer_search"), Mapping) else {}
    timeline = panels.get("timeline") if isinstance(panels.get("timeline"), Mapping) else {}
    bot_context = panels.get("bot_context") if isinstance(panels.get("bot_context"), Mapping) else {}
    conflicts = panels.get("conflicts") if isinstance(panels.get("conflicts"), Mapping) else {}
    safety = panels.get("safety_gates") if isinstance(panels.get("safety_gates"), Mapping) else {}
    title = "Customer Timeline Approval Workspace"
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --surface: #ffffff;
      --surface-soft: #eef2f5;
      --text: #18212f;
      --muted: #647084;
      --border: #d8dee7;
      --accent: #1458d4;
      --ok: #087443;
      --warn: #a15c07;
      --danger: #b42318;
      --shadow: 0 8px 24px rgba(24, 33, 47, 0.07);
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
      z-index: 4;
      background: rgba(247, 248, 250, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }}
    .bar {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }}
    h1 {{ margin: 0; font-size: 20px; line-height: 1.2; }}
    .sub {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 20px 24px 44px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .metric, .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{ min-height: 88px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; }}
    .metric span, .label {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    .metric strong {{ font-size: 24px; line-height: 1; }}
    .layout {{ display: grid; grid-template-columns: 280px minmax(0, 1fr) 340px; gap: 14px; align-items: start; }}
    .panel {{ padding: 14px; min-width: 0; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 15px; line-height: 1.25; }}
    .stack {{ display: grid; gap: 12px; }}
    .row {{ border-top: 1px solid var(--border); padding: 10px 0; }}
    .row:first-child {{ border-top: 0; padding-top: 0; }}
    .row strong {{ display: block; font-size: 13px; line-height: 1.3; overflow-wrap: anywhere; }}
    .row p {{ margin: 5px 0 0; color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
    .kv {{ display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 8px; padding: 7px 0; border-top: 1px solid var(--border); }}
    .kv:first-child {{ border-top: 0; }}
    .pill {{
      display: inline-flex;
      min-height: 26px;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
      background: var(--surface-soft);
      margin: 0 6px 6px 0;
    }}
    .pill.ok {{ color: var(--ok); border-color: rgba(8, 116, 67, 0.25); background: #eef8f3; }}
    .pill.warn {{ color: var(--warn); border-color: rgba(161, 92, 7, 0.28); background: #fff7ed; }}
    .pill.danger {{ color: var(--danger); border-color: rgba(180, 35, 24, 0.24); background: #fff1f0; }}
    .timeline-item {{ display: grid; grid-template-columns: 116px minmax(0, 1fr); gap: 12px; padding: 12px 0; border-top: 1px solid var(--border); }}
    .timeline-item:first-child {{ border-top: 0; padding-top: 0; }}
    .time {{ color: var(--muted); font-size: 12px; line-height: 1.35; }}
    .event-title {{ font-weight: 750; font-size: 13px; }}
    .event-body {{ margin-top: 5px; color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
    .disabled-actions {{ display: grid; gap: 8px; }}
    button {{
      appearance: none;
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #f1f4f8;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: not-allowed;
    }}
    @media (max-width: 1020px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .bar {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>{escape(title)}</h1>
        <p class="sub">Tenant: {e(workspace.get("tenant_id"))} · generated {e(workspace.get("generated_at"))}</p>
      </div>
      <div>{status_pill(summary.get("status"))}{bool_pill("Read-only", True)}{bool_pill("Live actions", False, invert=True)}</div>
    </div>
  </header>
  <main>
    <section class="metrics">
      {metric("Selected", summary.get("selected_customer_found"), "customer found")}
      {metric("Customers", summary.get("customers_visible"), "visible")}
      {metric("Open conflicts", summary.get("open_conflicts"), "review required")}
      {metric("Bot chunks", summary.get("bot_allowed_chunks"), "allowed")}
      {metric("Warnings", summary.get("warnings"), "operator notes")}
    </section>
    <section class="layout">
      <aside class="panel">
        <h2>Customer search</h2>
        {render_customer_rows(customer_search.get("items") or [])}
      </aside>
      <section class="stack">
        <div class="panel">
          <h2>Selected customer</h2>
          {render_customer_card(customer, summary)}
        </div>
        <div class="panel">
          <h2>Timeline</h2>
          {render_timeline(timeline.get("items") or [])}
        </div>
      </section>
      <aside class="stack">
        <div class="panel">
          <h2>Bot context readiness</h2>
          {render_bot_context(bot_context)}
        </div>
        <div class="panel">
          <h2>Conflicts</h2>
          {render_conflicts(conflicts.get("items") or [])}
        </div>
        <div class="panel">
          <h2>Safety gates</h2>
          {render_safety(safety)}
        </div>
      </aside>
    </section>
  </main>
</body>
</html>
"""


def guard_workspace_output_path(path: Optional[Path], read_config: CustomerTimelineReadApiConfig) -> Optional[Path]:
    if path is None:
        return None
    resolved = guard_customer_timeline_output_path(path, read_config.allowed_root)
    if resolved == read_config.timeline_db:
        raise ValueError("workspace output path must not overwrite timeline DB")
    return resolved


def first_customer_id(customer_search: Mapping[str, Any]) -> Optional[str]:
    items = customer_search.get("items") or ()
    if not items:
        return None
    first = items[0]
    return str(first.get("customer_id")) if isinstance(first, Mapping) and first.get("customer_id") else None


def approval_readiness(
    selected_profile: Optional[Mapping[str, Any]],
    conflicts: Mapping[str, Any],
) -> Mapping[str, Any]:
    open_conflicts = int((conflicts.get("summary") or {}).get("open_conflicts") or 0)
    bot_summary: Mapping[str, Any] = {}
    if selected_profile and isinstance(selected_profile.get("bot_context"), Mapping):
        bot_summary = selected_profile["bot_context"].get("summary") or {}
    bot_allowed = int(bot_summary.get("allowed_chunks") or 0)
    review_required = int(bot_summary.get("review_required_chunks") or 0)
    warnings = 0
    blocked = 0
    if not selected_profile or not selected_profile.get("found"):
        warnings += 1
    if open_conflicts:
        blocked += 1
    if review_required:
        warnings += 1
    status = "ready_for_review"
    if blocked:
        status = "blocked_by_conflict"
    elif bot_allowed == 0:
        status = "needs_context"
        warnings += 1
    return {
        "status": status,
        "open_conflicts": open_conflicts,
        "bot_allowed_chunks": bot_allowed,
        "bot_review_required_chunks": review_required,
        "warnings": warnings,
        "blocked": blocked,
    }


def build_review_queue(
    selected_profile: Optional[Mapping[str, Any]],
    conflicts: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    queue: list[Mapping[str, Any]] = []
    conflict_items = conflicts.get("items") if isinstance(conflicts.get("items"), list) else []
    for item in conflict_items:
        queue.append(
            {
                "action": "REVIEW_IDENTITY_CONFLICT",
                "priority": "high",
                "label": item.get("summary") or item.get("conflict_type") or "Identity conflict",
                "live_write": False,
            }
        )
    if selected_profile and selected_profile.get("found"):
        readiness = selected_profile.get("readiness") or {}
        if int(readiness.get("bot_review_required_chunks") or 0):
            queue.append(
                {
                    "action": "REVIEW_BOT_CONTEXT",
                    "priority": "medium",
                    "label": "Bot context contains chunks requiring manager review",
                    "live_write": False,
                }
            )
        if not queue:
            queue.append(
                {
                    "action": "READY_FOR_OPERATOR_APPROVAL_REVIEW",
                    "priority": "normal",
                    "label": "Customer timeline is ready for read-only approval review",
                    "live_write": False,
                }
            )
    return queue


def workspace_actions() -> Mapping[str, Any]:
    return {
        "read_only": True,
        "available": ["open_customer", "inspect_timeline", "inspect_bot_context", "inspect_conflicts"],
        "blocked": {
            "write_crm": False,
            "write_tallanto": False,
            "send_message": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
        },
    }


def safety_gates() -> Mapping[str, Any]:
    safety = customer_timeline_read_api_safety_contract()
    return {
        "read_only_db": True,
        "write_crm": safety["write_crm"],
        "write_tallanto": safety["write_tallanto"],
        "send_email": safety["send_email"],
        "send_messenger": safety["send_messenger"],
        "run_asr": safety["run_asr"],
        "run_ra": safety["run_ra"],
        "write_runtime_db": safety["write_runtime_db"],
        "stable_runtime_writes": safety["stable_runtime_writes"],
    }


def metric(label: str, value: Any, note: str) -> str:
    return f'<div class="metric"><span>{e(label)}</span><strong>{e(value)}</strong><small>{e(note)}</small></div>'


def status_pill(status: Any) -> str:
    text = str(status or "unknown")
    klass = "danger" if "blocked" in text else "warn" if "needs" in text else "ok"
    return f'<span class="pill {klass}">{e(text)}</span>'


def bool_pill(label: str, value: bool, *, invert: bool = False) -> str:
    ok = not value if invert else value
    klass = "ok" if ok else "danger"
    return f'<span class="pill {klass}">{e(label)}: {e("yes" if value else "no")}</span>'


def render_customer_rows(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return '<p class="sub">No customers for current filter.</p>'
    return "".join(
        f'<div class="row"><strong>{e(item.get("display_name") or item.get("customer_id"))}</strong>'
        f'<p>{e(item.get("identity_status"))} · {e(item.get("primary_phone"))} · {e(item.get("primary_email"))}</p></div>'
        for item in items
    )


def render_customer_card(customer: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    if not customer:
        return '<p class="sub">No selected customer. Use query or customer_id to focus the workspace.</p>'
    rows = (
        ("Customer", customer.get("display_name") or customer.get("customer_id")),
        ("Status", customer.get("identity_status")),
        ("Phone", customer.get("primary_phone")),
        ("Email", customer.get("primary_email")),
        ("Touch count", customer.get("touch_count")),
        ("Last seen", customer.get("last_seen_at")),
        ("Workspace", summary.get("status")),
    )
    return "".join(f'<div class="kv"><div class="label">{e(label)}</div><div>{e(value)}</div></div>' for label, value in rows)


def render_timeline(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return '<p class="sub">No timeline events for selected customer.</p>'
    return "".join(
        f'<div class="timeline-item"><div class="time">{e(item.get("event_at"))}<br>{e(item.get("event_type"))}</div>'
        f'<div><div class="event-title">{e(item.get("subject") or item.get("source_system"))}</div>'
        f'<div class="event-body">{e(item.get("summary") or item.get("text_preview"))}</div></div></div>'
        for item in items
    )


def render_bot_context(panel: Mapping[str, Any]) -> str:
    if not panel:
        return '<p class="sub">No selected bot context.</p>'
    summary = panel.get("summary") or {}
    items = panel.get("items") or []
    head = (
        f'{bool_pill("Bot safe", bool(summary.get("allowed_chunks")), invert=False)}'
        f'<span class="pill warn">Review: {e(summary.get("review_required_chunks"))}</span>'
        f'<span class="pill">Blocked: {e(summary.get("blocked_chunks"))}</span>'
    )
    body = "".join(
        f'<div class="row"><strong>{e(item.get("chunk_type"))}</strong><p>{e(item.get("summary") or item.get("text"))}</p></div>'
        for item in items[:6]
    )
    return head + (body or '<p class="sub">No bot-safe chunks.</p>')


def render_conflicts(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return '<span class="pill ok">No open conflicts</span>'
    return "".join(
        f'<div class="row"><strong>{e(item.get("conflict_type"))} · {e(item.get("status"))}</strong>'
        f'<p>{e(item.get("summary"))}</p></div>'
        for item in items[:8]
    )


def render_safety(safety: Mapping[str, Any]) -> str:
    rows = "".join(
        f'<div class="kv"><div class="label">{e(key)}</div><div>{e(value)}</div></div>'
        for key, value in safety.items()
    )
    actions = (
        '<div class="disabled-actions">'
        '<button disabled>Write CRM blocked</button>'
        '<button disabled>Send message blocked</button>'
        '<button disabled>Run ASR/R+A blocked</button>'
        "</div>"
    )
    return rows + actions


def e(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CustomerTimelineApprovalWorkspaceConfig(
            timeline_db=Path(args.timeline_db),
            allowed_root=Path(args.allowed_root),
            out_json=Path(args.out_json) if args.out_json else None,
            out_html=Path(args.out_html) if args.out_html else None,
        )
        workspace = build_customer_timeline_approval_workspace(
            config=config,
            tenant_id=args.tenant_id,
            customer_id=args.customer_id,
            query=args.query,
            limit=args.limit,
        )
        if not args.out_json and not args.out_html:
            print(json.dumps(workspace, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if workspace["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI-facing compact error.
        print(f"customer timeline approval workspace failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build read-only Customer Timeline approval workspace JSON/HTML.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--customer-id")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out-json")
    parser.add_argument("--out-html")
    return parser


__all__ = [
    "CUSTOMER_TIMELINE_APPROVAL_WORKSPACE_SCHEMA_VERSION",
    "CustomerTimelineApprovalWorkspaceConfig",
    "build_customer_timeline_approval_workspace",
    "render_customer_timeline_approval_workspace_html",
    "main",
]
