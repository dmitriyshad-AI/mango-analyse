from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.product_api import ProductApiFacade
from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from mango_mvp.productization.tenant_isolation import build_tenant_isolation_report
from mango_mvp.productization.test_ingest import path_is_relative_to


DEMO_PILOT_PLAYBOOK_SCHEMA_VERSION = "demo_pilot_playbook_v1"


@dataclass(frozen=True)
class DemoPilotPlaybookSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    out_dir: str
    markdown_path: str
    json_path: str
    demo_ready: bool
    pilot_ready_without_processing: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_demo_pilot_playbook(
    product_root: Path,
    product_db_path: Path,
    out_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_dir = (out_dir or product_root / "demo_pilot_playbook").resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_under_root(out_dir, product_root, "demo pilot playbook output")
    out_dir.mkdir(parents=True, exist_ok=True)

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path)
    dashboard = api.appliance_dashboard(capture_limit=10, scheduler_limit=10)
    demo = build_dashboard_demo_readiness(product_root=product_root, product_db_path=product_db_path, panels=dashboard.get("panels", {}))
    scheduler = build_scheduler_health_report(product_db_path=product_db_path, product_root=product_root)
    tenants = build_tenant_isolation_report(product_root=product_root, product_db_path=product_db_path)
    blocked = int(demo["summary"]["blocked"]) + int(tenants["summary"]["blocked"])
    warnings = int(demo["summary"]["warnings"]) + int(scheduler["summary"]["warnings"]) + int(tenants["summary"]["warnings"])
    payload = {
        "schema_version": DEMO_PILOT_PLAYBOOK_SCHEMA_VERSION,
        "dashboard_url": "http://127.0.0.1:8765/dashboard",
        "dashboard_summary": dashboard["summary"],
        "demo_readiness": demo["summary"],
        "scheduler_summary": scheduler["summary"],
        "tenant_summary": tenants["summary"],
        "demo_storyline": demo_storyline(),
        "pilot_checklist": pilot_checklist(),
        "blocked_live_actions": ["ASR execution", "R+A execution", "runtime DB writes", "CRM/Tallanto writeback"],
        "commands": {
            "serve_dashboard": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py "
                f"--product-root {product_root} --product-db {product_db_path} serve --host 127.0.0.1 --port 8765"
            ),
            "build_sanitized_real_demo": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_sanitized_real_demo.py "
                f"--source-product-root {product_root} --source-product-db {product_db_path} "
                f"--demo-product-root {product_root.parent / 'sanitized_real_demo_appliance'} --replace"
            ),
        },
        "safety": safety_contract(),
    }
    markdown_path = out_dir / "demo_pilot_playbook.md"
    json_path = out_dir / "demo_pilot_playbook.json"
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    report = {
        "summary": DemoPilotPlaybookSummary(
            schema_version=DEMO_PILOT_PLAYBOOK_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            out_dir=str(out_dir),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            demo_ready=blocked == 0,
            pilot_ready_without_processing=blocked == 0,
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "playbook": payload,
        "safety": safety_contract(),
    }
    write_json(out_dir / "demo_pilot_playbook_manifest.json", report)
    return report


def demo_storyline() -> list[Mapping[str, str]]:
    return [
        {"step": "Open dashboard", "goal": "Show that the product has a local read-only control panel."},
        {"step": "Capture panel", "goal": "Show real Mango events normalized into product rows."},
        {"step": "CRM Preview", "goal": "Show CRM/Tallanto candidates without writing to CRM."},
        {"step": "Scheduler and Ops", "goal": "Show local appliance health, diagnostics and explicit commands."},
        {"step": "Safety Gates", "goal": "Show that ASR/R+A/runtime/CRM writes are blocked by default."},
    ]


def pilot_checklist() -> list[Mapping[str, str]]:
    return [
        {"item": "Mango API credentials", "status": "required_for_shadow_poll"},
        {"item": "AMO/Tallanto read-only snapshot", "status": "required_for_crm_preview"},
        {"item": "Sanitized real demo root", "status": "recommended_before_external_demo"},
        {"item": "Backup/restore dry-run", "status": "required_before_client_hosted_pilot"},
        {"item": "Processing quality acceptance", "status": "external_blocker_owned_by_processing_dialog"},
    ]


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["dashboard_summary"]
    demo = payload["demo_readiness"]
    lines = [
        "# Mango Analyse Demo/Pilot Playbook",
        "",
        "## Demo Summary",
        "",
        f"- Dashboard URL: `{payload['dashboard_url']}`",
        f"- Product calls: `{summary.get('product_calls')}`",
        f"- Capture ready: `{summary.get('capture_inbox_ready')}`",
        f"- Snapshot files: `{demo.get('snapshot_files')}`",
        f"- Demo artifacts: `{demo.get('demo_artifacts')}`",
        "",
        "## Storyline",
        "",
    ]
    lines.extend(f"- {item['step']}: {item['goal']}" for item in payload["demo_storyline"])
    lines.extend(["", "## Pilot Checklist", ""])
    lines.extend(f"- {item['item']}: `{item['status']}`" for item in payload["pilot_checklist"])
    lines.extend(["", "## Blocked By Default", ""])
    lines.extend(f"- {item}" for item in payload["blocked_live_actions"])
    lines.extend(["", "## Commands", ""])
    for name, command in payload["commands"].items():
        lines.extend([f"### {name}", "", "```zsh", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "read_only": True,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
