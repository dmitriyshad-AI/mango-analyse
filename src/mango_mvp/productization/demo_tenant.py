from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.crm_entity_resolver import build_crm_entity_resolution_report
from mango_mvp.productization.crm_writeback_preview import build_crm_writeback_preview
from mango_mvp.productization.product_db import (
    audit_product_db,
    initialize_product_db,
    now_utc,
    seed_default_retention_policies,
    seed_job_types,
)
from mango_mvp.productization.product_api import build_product_api_readiness_report
from mango_mvp.productization.test_ingest import path_is_relative_to


DEMO_TENANT_SCHEMA_VERSION = "demo_tenant_product_root_v1"


@dataclass(frozen=True)
class DemoTenantSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    crm_snapshot_path: str
    product_calls: int
    capture_inbox_items: int
    crm_entities: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_demo_tenant_product_root(
    product_root: Path,
    out_path: Optional[Path] = None,
    *,
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db = product_root / "mango_product_appliance.sqlite"
    out_path = (out_path or product_root / "demo_tenant_report.json").resolve(strict=False)
    guard_demo_path(product_root, product_root, "product root")
    guard_demo_path(out_path, product_root, "demo report")
    initialize_product_db(product_db, product_root, replace_existing=replace_existing)
    seed_demo_rows(product_db)
    crm_snapshot = write_demo_crm_snapshot(product_root)
    resolver = build_crm_entity_resolution_report(
        product_db_path=product_db,
        product_root=product_root,
        crm_snapshot_path=crm_snapshot,
        out_path=product_root / "crm_entity_resolver_stage6" / "demo_resolution.json",
    )
    preview = build_crm_writeback_preview(
        product_db_path=product_db,
        product_root=product_root,
        crm_snapshot_path=crm_snapshot,
        out_path=product_root / "crm_writeback_preview_stage6" / "demo_preview.json",
    )
    api = build_product_api_readiness_report(
        product_root=product_root,
        product_db_path=product_db,
        out_path=product_root / "product_api_readiness" / "demo_api_readiness.json",
    )
    integrity = audit_product_db(product_db, product_root)
    summary = DemoTenantSummary(
        schema_version=DEMO_TENANT_SCHEMA_VERSION,
        product_root=str(product_root),
        product_db_path=str(product_db),
        crm_snapshot_path=str(crm_snapshot),
        product_calls=int(integrity["summary"]["product_calls"]),
        capture_inbox_items=int(integrity["summary"]["capture_inbox_items"]),
        crm_entities=len(json.loads(crm_snapshot.read_text(encoding="utf-8"))["entities"]),
        validation_ok=bool(integrity["summary"]["validation_ok"]) and bool(api["summary"]["validation_ok"]),
        blocked=int(integrity["summary"]["blocked"]),
        warnings=int(integrity["summary"]["warnings"]) + int(preview["summary"]["warnings"]),
    )
    report = {
        "summary": summary.to_json_dict(),
        "integrity": integrity,
        "crm_resolution": resolver["summary"],
        "writeback_preview": preview["summary"],
        "api_readiness": api["summary"],
        "demo_commands": {
            "serve_dashboard": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 "
                f"scripts/mango_office_product_api_http.py --product-root {product_root} "
                f"--product-db {product_db} serve --host 127.0.0.1 --port 8765"
            ),
            "open_dashboard": "http://127.0.0.1:8765/dashboard",
        },
        "safety": {
            "fake_data_only": True,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "live_crm_reads": False,
            "write_crm": False,
            "run_asr": False,
            "run_ra": False,
        },
    }
    write_json(out_path, report)
    return report


def seed_demo_rows(product_db: Path) -> None:
    now = now_utc()
    with sqlite3.connect(str(product_db)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            """
            INSERT INTO tenants (tenant_id, display_name, status, created_at, updated_at)
            VALUES ('demo_foton', 'Demo Foton', 'active', ?, ?)
            ON CONFLICT(tenant_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (now, now),
        )
        seed_job_types(con, now)
        for manager, owner_id, owner_name in (
            ("101", 9001, "Анна Demo"),
            ("102", 9002, "Олег Demo"),
        ):
            con.execute(
                """
                INSERT INTO tenant_manager_owner_map (
                  tenant_id, telephony_provider, manager_extension,
                  mango_name, mango_email, crm_provider, crm_owner_id, crm_owner_name,
                  crm_owner_email, decision_status, match_status, source_ref, config_ref,
                  notes, created_at, updated_at
                ) VALUES ('demo_foton', 'mango', ?, ?, ?, 'amocrm', ?, ?, ?, 'confirmed_candidate',
                          'demo_match', 'demo', 'demo', 'demo tenant', ?, ?)
                ON CONFLICT(tenant_id, telephony_provider, manager_extension) DO UPDATE SET
                  crm_owner_id = excluded.crm_owner_id,
                  crm_owner_name = excluded.crm_owner_name,
                  updated_at = excluded.updated_at
                """,
                (
                    manager,
                    f"mango_{manager}",
                    f"manager{manager}@demo.local",
                    owner_id,
                    owner_name,
                    f"owner{owner_id}@demo.local",
                    now,
                    now,
                ),
            )
        calls = [
            ("CALL-DEMO-1", "REC-DEMO-1", "2026-05-09T09:00:00+03:00", "101", "79990000000"),
            ("CALL-DEMO-2", "REC-DEMO-2", "2026-05-09T10:00:00+03:00", "101", "79990000001"),
            ("CALL-DEMO-3", "REC-DEMO-3", "2026-05-09T11:00:00+03:00", "102", "79990000002"),
            ("CALL-DEMO-4", "REC-DEMO-4", "2026-05-09T12:00:00+03:00", "102", "79990000003"),
        ]
        for call_id, rec_id, started_at, manager, phone in calls:
            filename = f"2026-05-09__{started_at[11:13]}-00-00__{phone}__mango_{manager}_{call_id}.mp3"
            con.execute(
                """
                INSERT INTO product_calls (
                  tenant_id, telephony_provider, provider_call_id, event_key, recording_id,
                  source_filename, started_at, duration_sec, manager_extension,
                  manager_display_name, crm_owner_id, crm_owner_name, crm_match_status,
                  raw_payload_ref, source_repository_ref, imported_at, updated_at
                ) VALUES ('demo_foton', 'mango', ?, ?, ?, ?, ?, 120.0, ?, ?, ?, ?, 'demo_match',
                          'demo/raw_payload.jsonl', 'demo_repository', ?, ?)
                ON CONFLICT(tenant_id, telephony_provider, provider_call_id) DO UPDATE SET
                  updated_at = excluded.updated_at
                """,
                (
                    call_id,
                    f"demo_foton:mango:{call_id}",
                    rec_id,
                    filename,
                    started_at,
                    manager,
                    f"mango_{manager}",
                    9001 if manager == "101" else 9002,
                    "Анна Demo" if manager == "101" else "Олег Demo",
                    now,
                    now,
                ),
            )
            con.execute(
                """
                INSERT INTO capture_inbox_items (
                  tenant_id, provider, event_key, provider_call_id, status,
                  started_at, direction, client_phone, manager_ref, recording_ref,
                  audio_ref, decision_reason, candidate_json, event_json,
                  first_seen_at, last_seen_at, enqueue_count
                ) VALUES ('demo_foton', 'mango', ?, ?, 'ready_for_capture',
                          ?, 'outbound', ?, ?, ?, ?, 'demo_capture', '{}', '{}', ?, ?, 1)
                ON CONFLICT(tenant_id, provider, event_key) DO UPDATE SET
                  last_seen_at = excluded.last_seen_at
                """,
                (
                    f"demo_foton:mango:{call_id}",
                    call_id,
                    started_at,
                    f"+{phone}",
                    manager,
                    rec_id,
                    rec_id,
                    now,
                    now,
                ),
            )
        con.execute(
            """
            INSERT INTO job_runs (
              job_type, tenant_id, status, planned_at, started_at, finished_at,
              input_ref, output_ref, error, scheduled_for, next_run_at,
              attempt_count, max_attempts
            ) VALUES
              ('shadow_poll', 'demo_foton', 'succeeded', ?, ?, ?, '{}', 'demo/scheduler/succeeded.json', NULL, ?, NULL, 1, 3),
              ('shadow_poll', 'demo_foton', 'planned', ?, NULL, NULL, '{}', NULL, NULL, ?, ?, 0, 3),
              ('shadow_poll', 'demo_foton', 'failed', ?, ?, ?, '{}', NULL, 'demo failure sample', ?, NULL, 3, 3)
            """,
            (now, now, now, now, now, now, now, now, now, now, now),
        )
        seed_default_retention_policies(con, now)
        con.commit()


def write_demo_crm_snapshot(product_root: Path) -> Path:
    path = product_root / "crm_snapshots" / "amocrm_entities.json"
    entities = [
        {"entity_id": "501", "entity_type": "lead", "phone": "+79990000000", "entity_name": "Demo lead A"},
        {"entity_id": "502", "entity_type": "lead", "phone": "+79990000001", "entity_name": "Demo lead B"},
        {"entity_id": "503", "entity_type": "lead", "phone": "+79990000002", "entity_name": "Demo lead C"},
        {"entity_id": "504", "entity_type": "contact", "phone": "+79990000003", "entity_name": "Demo contact D"},
    ]
    write_json(path, {"schema_version": DEMO_TENANT_SCHEMA_VERSION, "entities": entities})
    return path


def guard_demo_path(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if path != product_root and not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
