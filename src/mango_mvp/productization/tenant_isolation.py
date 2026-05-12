from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


TENANT_ISOLATION_SCHEMA_VERSION = "tenant_isolation_v1"
TENANT_SCOPED_TABLES = ("product_calls", "capture_inbox_items", "tenant_manager_owner_map", "job_runs")


@dataclass(frozen=True)
class TenantIsolationSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    tenants: int
    tenant_scoped_tables: int
    rows_without_tenant_id: int
    scaffold_written: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_tenant_isolation_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    *,
    scaffold: bool = False,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        guard_under_root(out_path, product_root, "tenant isolation output")

    with sqlite3.connect(readonly_uri(product_db_path), uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        tenants = read_tenants(con)
        table_counts = {table: tenant_counts(con, table) for table in TENANT_SCOPED_TABLES if table_exists(con, table)}
        missing_tenant_rows = {table: rows_without_tenant(con, table) for table in table_counts}
    total_missing = sum(missing_tenant_rows.values())
    tenant_layout = [tenant_layout_item(product_root, tenant["tenant_id"], scaffold=scaffold) for tenant in tenants]
    blocked = sum(missing_tenant_rows.get(table, 0) for table in ("product_calls", "capture_inbox_items", "tenant_manager_owner_map"))
    warnings = missing_tenant_rows.get("job_runs", 0)
    report = {
        "summary": TenantIsolationSummary(
            schema_version=TENANT_ISOLATION_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            tenants=len(tenants),
            tenant_scoped_tables=len(table_counts),
            rows_without_tenant_id=total_missing,
            scaffold_written=scaffold,
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "tenants": tenants,
        "table_counts": table_counts,
        "missing_tenant_rows": missing_tenant_rows,
        "tenant_layout": tenant_layout,
        "policy": {
            "tenant_id_required_for": ["product_calls", "capture_inbox_items", "tenant_manager_owner_map"],
            "tenant_id_recommended_for": ["job_runs"],
            "separate_secret_files_per_tenant": True,
            "separate_snapshots_per_tenant": True,
        },
        "safety": safety_contract() | {"scaffold_written": scaffold},
    }
    if out_path:
        write_json(out_path, report)
    return report


def read_tenants(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    if not table_exists(con, "tenants"):
        return []
    return [
        {
            "tenant_id": clean(row["tenant_id"]),
            "display_name": clean(row["display_name"]) or clean(row["tenant_id"]),
            "status": clean(row["status"]) or None,
        }
        for row in con.execute("SELECT tenant_id, display_name, status FROM tenants ORDER BY tenant_id").fetchall()
    ]


def tenant_counts(con: sqlite3.Connection, table: str) -> Mapping[str, int]:
    if not column_exists(con, table, "tenant_id"):
        return {}
    rows = con.execute(
        f"SELECT COALESCE(NULLIF(TRIM(tenant_id), ''), '<missing>') AS tenant_id, COUNT(*) FROM {table} GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {clean(row[0]): int(row[1]) for row in rows}


def rows_without_tenant(con: sqlite3.Connection, table: str) -> int:
    if not column_exists(con, table, "tenant_id"):
        return 0
    row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE tenant_id IS NULL OR TRIM(tenant_id) = ''").fetchone()
    return int(row[0] or 0)


def tenant_layout_item(product_root: Path, tenant_id: str, *, scaffold: bool) -> Mapping[str, Any]:
    safe_tenant = safe_name(tenant_id)
    root = product_root / "tenants" / safe_tenant
    dirs = {
        "root": root,
        "config": root / "config",
        "crm_snapshots": root / "crm_snapshots",
        "reports": root / "reports",
        "backups": root / "backups",
    }
    if scaffold:
        for path in dirs.values():
            guard_under_root(path, product_root, "tenant scaffold directory")
            path.mkdir(parents=True, exist_ok=True)
        readme = root / "README.md"
        readme.write_text(
            f"# Tenant {tenant_id}\n\nKeep tenant-specific config, snapshots and reports under this directory.\n",
            encoding="utf-8",
        )
    return {"tenant_id": tenant_id, **{key: str(value) for key, value in dirs.items()}}


def safe_name(value: str) -> str:
    text = clean(value) or "unknown_tenant"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80]


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(clean(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


def readonly_uri(path: Path) -> str:
    return f"file:{quote(str(path), safe='/:')}?mode=ro"


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "read_only_db": True,
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
