from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.saas_stage_gates import build_saas_stage_gates_report
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


PRODUCT_API_SCHEMA_VERSION = "product_api_readonly_v1"


@dataclass(frozen=True)
class ProductApiSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    endpoints: int
    read_only: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class ProductApiFacade:
    """Read-only facade for the product appliance.

    This is a dependency-free API skeleton. A future FastAPI layer should call
    these methods instead of reading SQLite files or runtime artifacts directly.
    """

    def __init__(self, product_root: Path, product_db_path: Path, workspace_root: Optional[Path] = None) -> None:
        self.product_root = product_root.resolve(strict=False)
        self.product_db_path = product_db_path.resolve(strict=False)
        self.workspace_root = (workspace_root or Path.cwd()).resolve(strict=False)
        guard_product_api_paths(self.product_root, self.product_db_path)

    def dashboard_summary(self) -> Mapping[str, Any]:
        snapshot = self.product_db_snapshot()
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /dashboard/summary",
            "summary": snapshot["summary"],
            "health": {
                "product_db_present": self.product_db_path.exists(),
                "validation_ok": snapshot["summary"]["validation_ok"],
                "blocked": snapshot["summary"]["blocked"],
                "warnings": snapshot["summary"]["warnings"],
            },
            "actions": read_only_actions(),
        }

    def capture_recent(self, limit: int = 50) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._fetch_rows(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, manager_ref, recording_ref, audio_ref,
                   raw_payload_ref, enqueue_count, last_seen_at
              FROM capture_inbox_items
             ORDER BY last_seen_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
            required_table="capture_inbox_items",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /capture/recent",
            "limit": limit,
            "items": rows,
            "actions": read_only_actions(),
        }

    def scheduler_runs(self, limit: int = 50) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._fetch_rows(
            """
            SELECT id, job_type, tenant_id, status, planned_at,
                   scheduled_for, next_run_at, attempt_count, max_attempts,
                   lock_owner, lock_expires_at, heartbeat_at, output_ref
              FROM job_runs
             ORDER BY id DESC
             LIMIT ?
            """,
            (int(limit),),
            required_table="job_runs",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /scheduler/runs",
            "limit": limit,
            "items": rows,
            "actions": read_only_actions(),
        }

    def asr_gate_status(self) -> Mapping[str, Any]:
        stage25_dir = self.product_root / "asr_worker_sandbox_execution_request_stage25"
        audit_path = stage25_dir / "asr_worker_sandbox_execution_request_stage25_audit.json"
        request_path = stage25_dir / "asr_worker_sandbox_execution_request_stage25.json"
        payload = load_json_if_exists(audit_path) or load_json_if_exists(request_path) or {}
        summary = mapping_or_empty(payload.get("summary"))
        execution_request = mapping_or_empty(payload.get("execution_request"))
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /asr/gates",
            "stage25": {
                "audit_path": str(audit_path) if audit_path.exists() else None,
                "request_path": str(request_path) if request_path.exists() else None,
                "approval_packet_ref": summary.get("approval_packet_ref") or execution_request.get("approval_packet_ref"),
                "execution_request_ready": bool(summary.get("execution_request_ready") or execution_request.get("execution_request_ready")),
                "dispatch_allowed": False,
                "run_asr": False,
                "write_transcripts": False,
                "missing_or_invalid_reasons": mapping_or_empty(payload.get("approval")).get("missing_or_invalid_reasons") or [],
            },
            "actions": read_only_actions(),
        }

    def writeback_previews(self) -> Mapping[str, Any]:
        snapshot = self.product_db_snapshot()
        pending = int(snapshot["summary"].get("pending_owner_mappings") or 0)
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /writeback/previews",
            "current_mode": "preview_only",
            "write_crm": False,
            "write_tallanto": False,
            "blocked_reasons": ["pending_owner_mappings"] if pending else [],
            "required_sequence": ["dry_run_diff", "human_review", "pilot_50", "pilot_300", "full_rollout"],
            "actions": read_only_actions(),
        }

    def processing_queue(self, limit: int = 50) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._fetch_rows(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, manager_ref, recording_ref, audio_ref,
                   raw_payload_ref, enqueue_count, last_seen_at
              FROM capture_inbox_items
             WHERE status = 'ready_for_capture'
             ORDER BY started_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
            required_table="capture_inbox_items",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /queues/processing",
            "limit": limit,
            "items": rows,
            "queue_policy": {
                "auto_trigger_enabled": False,
                "requires_asr_gate": True,
                "run_asr": False,
                "write_runtime_db": False,
            },
            "actions": read_only_actions(),
        }

    def knowledge_playbook(self) -> Mapping[str, Any]:
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /knowledge/playbook",
            "current_mode": "schema_only",
            "entities": [
                "client_identity",
                "opportunity",
                "chain_event",
                "sales_moment",
                "response_quality_score",
                "outcome_link",
                "playbook_item",
            ],
            "items": [],
            "blocked_reasons": ["client_chain_layer_not_materialized", "outcome_linker_not_materialized"],
            "actions": read_only_actions(),
        }

    def settings_adapters(self) -> Mapping[str, Any]:
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /settings/adapters",
            "adapters": {
                "telephony": {
                    "primary": "mango",
                    "credentials_ref": ["env:MANGO_OFFICE_API_KEY", "env:MANGO_OFFICE_API_SALT"],
                    "write_capable": False,
                },
                "crm": {
                    "primary": "amocrm",
                    "write_mode": "preview_only",
                    "write_crm": False,
                },
                "database": {
                    "profile": "sqlite_single_writer_v1",
                    "product_db_path": str(self.product_db_path),
                },
            },
            "actions": read_only_actions(),
        }

    def saas_stage_gates(self) -> Mapping[str, Any]:
        return build_saas_stage_gates_report(
            product_root=self.product_root,
            product_db_path=self.product_db_path,
            workspace_root=self.workspace_root,
        )

    def product_db_snapshot(self) -> Mapping[str, Any]:
        if not self.product_db_path.exists():
            return {"summary": empty_db_summary(self.product_db_path)}
        with self.connect_ro() as con:
            tables = {name: count_relation(con, name) for name in PRODUCT_DB_TABLES}
            missing_relations = [name for name in PRODUCT_DB_TABLES if not relation_exists(con, name)]
            missing_migrations = missing_required_migrations(con)
            blocked = len(missing_relations) + len(missing_migrations)
            summary = {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "db_path": str(self.product_db_path),
                "schema_migrations": tables["schema_migrations"],
                "tenants": tables["tenants"],
                "product_calls": tables["product_calls"],
                "job_runs": tables["job_runs"],
                "capture_inbox_items": tables["capture_inbox_items"],
                "capture_inbox_ready": scalar_int_if_relation(
                    con,
                    "capture_inbox_items",
                    "select count(*) from capture_inbox_items where status = 'ready_for_capture'",
                ),
                "capture_inbox_blocked": scalar_int_if_relation(
                    con,
                    "capture_inbox_items",
                    "select count(*) from capture_inbox_items where status like 'blocked%'",
                ),
                "pending_owner_mappings": scalar_int_if_relation(
                    con,
                    "tenant_manager_owner_map",
                    "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'",
                ),
                "retention_policies": tables["retention_policies"],
                "validation_ok": blocked == 0,
                "blocked": blocked,
                "blocked_reasons": {
                    "missing_relations": missing_relations,
                    "missing_required_migrations": missing_migrations,
                },
                "warnings": scalar_int_if_relation(
                    con,
                    "tenant_manager_owner_map",
                    "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'",
                ),
            }
        return {"summary": summary, "tables": tables}

    def connect_ro(self) -> sqlite3.Connection:
        if not self.product_db_path.exists():
            raise FileNotFoundError(f"product DB not found: {self.product_db_path}")
        uri = f"file:{quote(str(self.product_db_path), safe='/:')}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _fetch_rows(self, sql: str, params: Sequence[Any], required_table: str) -> list[Mapping[str, Any]]:
        if not self.product_db_path.exists():
            return []
        with self.connect_ro() as con:
            if not relation_exists(con, required_table):
                return []
            rows = con.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]


PRODUCT_DB_TABLES = (
    "schema_migrations",
    "tenants",
    "product_calls",
    "job_runs",
    "capture_inbox_items",
    "tenant_manager_owner_map",
    "retention_policies",
)
REQUIRED_PRODUCT_DB_MIGRATIONS = (
    "20260507_001_product_appliance_base",
    "20260507_002_config_history_retention",
    "20260507_003_scheduler_runtime",
    "20260507_004_capture_inbox",
)


def build_product_api_readiness_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_api_paths(product_root, product_db_path, out_path)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)
    endpoints = {
        "dashboard_summary": api.dashboard_summary(),
        "capture_recent": api.capture_recent(limit=25),
        "processing_queue": api.processing_queue(limit=25),
        "scheduler_runs": api.scheduler_runs(limit=25),
        "asr_gate_status": api.asr_gate_status(),
        "writeback_previews": api.writeback_previews(),
        "knowledge_playbook": api.knowledge_playbook(),
        "settings_adapters": api.settings_adapters(),
    }
    dashboard_summary = mapping_or_empty(endpoints["dashboard_summary"].get("summary"))
    product_db_present = product_db_path.exists() and product_db_path.is_file()
    readiness_ok = product_db_present and bool(dashboard_summary.get("validation_ok"))
    blocked_reasons = []
    if not product_db_present:
        blocked_reasons.append("product_db_missing")
    if product_db_present and not bool(dashboard_summary.get("validation_ok")):
        blocked_reasons.append("product_db_invalid")
    blocked = max(int(dashboard_summary.get("blocked") or 0), len(blocked_reasons))
    warnings = int(dashboard_summary.get("warnings") or 0)
    report = {
        "summary": ProductApiSummary(
            schema_version=PRODUCT_API_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            endpoints=len(endpoints),
            read_only=True,
            validation_ok=readiness_ok,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict()
        | {
            "report_generated_ok": True,
            "product_db_present": product_db_present,
            "blocked_reasons": blocked_reasons,
        },
        "endpoints": endpoints,
        "safety": read_only_actions(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def guard_product_api_paths(product_root: Path, product_db_path: Path, out_path: Optional[Path] = None) -> None:
    for label, path in (("product root", product_root), ("product DB", product_db_path), ("product API audit", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"product API audit must stay under product root: {product_root}")


def read_only_actions() -> Mapping[str, bool]:
    return {
        "read_only": True,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "stable_runtime_writes": False,
        "write_crm": False,
        "write_tallanto": False,
        "dispatch_worker": False,
    }


def empty_db_summary(product_db_path: Path) -> Mapping[str, Any]:
    return {
        "schema_version": PRODUCT_API_SCHEMA_VERSION,
        "db_path": str(product_db_path),
        "schema_migrations": 0,
        "tenants": 0,
        "product_calls": 0,
        "job_runs": 0,
        "capture_inbox_items": 0,
        "capture_inbox_ready": 0,
        "capture_inbox_blocked": 0,
        "pending_owner_mappings": 0,
        "retention_policies": 0,
        "validation_ok": False,
        "blocked": 1,
        "blocked_reasons": {
            "missing_relations": list(PRODUCT_DB_TABLES),
            "missing_required_migrations": list(REQUIRED_PRODUCT_DB_MIGRATIONS),
        },
        "warnings": 0,
    }


def count_relation(con: sqlite3.Connection, name: str) -> int:
    if not relation_exists(con, name):
        return 0
    return scalar_int(con, f"select count(*) from {name}")


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where name = ? and type in ('table', 'view')",
        (clean(name),),
    ).fetchone()
    return row is not None


def scalar_int(con: sqlite3.Connection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def scalar_int_if_relation(con: sqlite3.Connection, relation_name: str, sql: str) -> int:
    if not relation_exists(con, relation_name):
        return 0
    return scalar_int(con, sql)


def missing_required_migrations(con: sqlite3.Connection) -> list[str]:
    if not relation_exists(con, "schema_migrations"):
        return list(REQUIRED_PRODUCT_DB_MIGRATIONS)
    present = {
        clean(row[0])
        for row in con.execute("select migration_id from schema_migrations").fetchall()
    }
    return [migration for migration in REQUIRED_PRODUCT_DB_MIGRATIONS if migration not in present]


def load_json_if_exists(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else None


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
