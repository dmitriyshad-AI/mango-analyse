from __future__ import annotations

import json
import shutil
import sqlite3
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.repository import ManagerRollupItem, ProductCallRecord, ProductRepository
from mango_mvp.productization.tenant_owner_mapping import build_tenant_owner_mapping_draft, load_config
from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


PRODUCT_DB_SCHEMA_VERSION = "product_appliance_sqlite_v1"
OWNER_CONFIG_APPLY_SCHEMA_VERSION = "product_owner_config_apply_v1"
PRODUCT_DB_MIGRATION_ID = "20260507_001_product_appliance_base"
PRODUCT_DB_RETENTION_MIGRATION_ID = "20260507_002_config_history_retention"
PRODUCT_DB_SCHEDULER_MIGRATION_ID = "20260507_003_scheduler_runtime"
PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID = "20260507_004_capture_inbox"
PRODUCT_DB_REQUIRED_MIGRATIONS = (
    PRODUCT_DB_MIGRATION_ID,
    PRODUCT_DB_RETENTION_MIGRATION_ID,
    PRODUCT_DB_SCHEDULER_MIGRATION_ID,
    PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID,
)
PRODUCT_DB_ADMIN_SCHEMA_VERSION = "product_db_admin_v1"
PRODUCT_DB_FILENAMES = {"mango_product_appliance.sqlite", "mango_product_appliance.db"}
SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm")
DEFAULT_RETENTION_POLICIES = (
    ("product_db_backup", 30, "review_delete", 1, "Review product DB backup files older than 30 days."),
    ("audit_json", 180, "review_archive", 1, "Review generated JSON audit files older than 180 days."),
    ("tenant_config_history", 1095, "keep", 1, "Keep tenant config history for at least 3 years."),
    ("product_calls", 1095, "manual_review_only", 0, "Product call rows are never auto-deleted by this policy."),
)


@dataclass(frozen=True)
class ProductDbInitSummary:
    schema_version: str
    db_path: str
    replaced_existing_db: bool
    migrations_applied: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductDbImportSummary:
    schema_version: str
    product_db_path: str
    source_db_path: str
    tenants_upserted: int
    manager_owner_rows_upserted: int
    calls_upserted: int
    pending_owner_mappings: int
    calls_with_crm_owner: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductDbIntegritySummary:
    schema_version: str
    db_path: str
    schema_migrations: int
    tenants: int
    manager_owner_rows: int
    product_calls: int
    job_runs: int
    due_job_runs: int
    running_job_runs: int
    failed_job_runs: int
    capture_inbox_items: int
    capture_inbox_ready: int
    capture_inbox_blocked: int
    calls_with_crm_owner: int
    pending_owner_mappings: int
    raw_payload_refs_present: int
    job_types: int
    tenant_config_history: int
    retention_policies: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductOwnerConfigApplySummary:
    schema_version: str
    product_db_path: str
    config_path: str
    tenant_id: str
    provider: str
    config_entries: int
    manager_owner_rows: int
    complete_owner_entries: int
    missing_owner_entries: int
    would_set_owner: int
    would_confirm_existing: int
    applied: int
    calls_would_gain_owner: int
    calls_with_crm_owner_before: int
    calls_with_crm_owner_after: int
    pending_owner_mappings_before: int
    pending_owner_mappings_after: int
    dry_run: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductDbAdminSummary:
    schema_version: str
    db_path: str
    operation: str
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def initialize_product_db(
    db_path: Path,
    out_allowed_root: Path,
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    guard_product_db_path(db_path, out_allowed_root, must_exist=False)
    replaced = False
    if replace_existing:
        replaced = remove_sqlite_db(db_path, out_allowed_root)
    elif db_path.exists():
        raise FileExistsError(f"product DB already exists: {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        create_product_db_schema(con)
        applied = apply_product_db_migrations(con)
        con.commit()
        integrity = audit_product_db_connection(con, db_path)
    summary = ProductDbInitSummary(
        schema_version=PRODUCT_DB_SCHEMA_VERSION,
        db_path=str(db_path),
        replaced_existing_db=replaced,
        migrations_applied=applied,
        validation_ok=integrity["summary"]["validation_ok"],
        blocked=int(integrity["summary"]["blocked"]),
        warnings=int(integrity["summary"]["warnings"]),
    )
    return {
        "summary": summary.to_json_dict(),
        "integrity": integrity,
    }


def import_repository_snapshot_to_product_db(
    source_repo: ProductRepository,
    product_db_path: Path,
    out_allowed_root: Path,
    tenant_owner_config_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    tenant_owner_config_path = tenant_owner_config_path.resolve(strict=False) if tenant_owner_config_path else None
    guard_product_db_path(product_db_path, out_allowed_root, must_exist=True)
    if tenant_owner_config_path and not path_is_relative_to(tenant_owner_config_path, out_allowed_root):
        raise ValueError(f"tenant owner config must stay under product root: {out_allowed_root}")

    owner_rows = tuple(source_repo.manager_rollup())
    calls = tuple(source_repo.list_calls(limit=1_000_000))
    tenants = sorted({row.tenant_id for row in owner_rows} | {call.tenant_id for call in calls})
    now = now_utc()
    source_ref = str(source_repo.db_path)
    config_ref = str(tenant_owner_config_path) if tenant_owner_config_path else None

    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        create_product_db_schema(con)
        apply_product_db_migrations(con)
        for tenant_id in tenants:
            con.execute(
                """
                INSERT INTO tenants (tenant_id, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (tenant_id, tenant_id, now, now),
            )
        for row in owner_rows:
            upsert_manager_owner_row(con, row, source_ref=source_ref, config_ref=config_ref, now=now)
        for call in calls:
            upsert_product_call(con, call, source_ref=source_ref, now=now)
        seed_job_types(con, now)
        seed_default_retention_policies(con, now)
        if tenant_owner_config_path:
            snapshot_tenant_config_connection(
                con,
                config_path=tenant_owner_config_path,
                snapshot_reason="repository_import",
                now=now,
            )
        con.commit()
        integrity = audit_product_db_connection(con, product_db_path)

    pending = sum(1 for row in owner_rows if row.crm_owner_id is None)
    calls_with_owner = sum(1 for call in calls if call.manager_crm_owner_id is not None)
    blocked = 0 if int(integrity["summary"]["product_calls"]) == len(calls) else 1
    warnings = pending
    summary = ProductDbImportSummary(
        schema_version=PRODUCT_DB_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        source_db_path=str(source_repo.db_path),
        tenants_upserted=len(tenants),
        manager_owner_rows_upserted=len(owner_rows),
        calls_upserted=len(calls),
        pending_owner_mappings=pending,
        calls_with_crm_owner=calls_with_owner,
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    return {
        "summary": summary.to_json_dict(),
        "integrity": integrity,
    }


def bootstrap_product_db_from_repository(
    source_db_path: Path,
    source_allowed_root: Path,
    product_db_path: Path,
    product_root: Path,
    tenant_owner_config_path: Path,
    replace_existing: bool = False,
    audit_out: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    tenant_owner_config_path = tenant_owner_config_path.resolve(strict=False)
    if audit_out:
        audit_out = audit_out.resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=False)
    if not path_is_relative_to(tenant_owner_config_path, product_root):
        raise ValueError(f"tenant owner config must stay under product root: {product_root}")
    if audit_out and not path_is_relative_to(audit_out, product_root):
        raise ValueError(f"product DB audit must stay under product root: {product_root}")

    source_repo = ProductRepository(source_db_path, source_allowed_root)
    owner_draft = build_tenant_owner_mapping_draft(
        db_path=source_db_path,
        out_allowed_root=source_allowed_root,
    )
    write_json(tenant_owner_config_path, owner_draft["config_template"])
    init_report = initialize_product_db(product_db_path, product_root, replace_existing=replace_existing)
    import_report = import_repository_snapshot_to_product_db(
        source_repo=source_repo,
        product_db_path=product_db_path,
        out_allowed_root=product_root,
        tenant_owner_config_path=tenant_owner_config_path,
    )
    integrity = audit_product_db(product_db_path, product_root)
    report = {
        "summary": {
            "schema_version": PRODUCT_DB_SCHEMA_VERSION,
            "validation_ok": bool(
                init_report["summary"]["validation_ok"]
                and import_report["summary"]["validation_ok"]
                and integrity["summary"]["validation_ok"]
            ),
            "product_db_path": str(product_db_path),
            "tenant_owner_config_path": str(tenant_owner_config_path),
            "source_db_path": str(source_repo.db_path),
            "product_calls": integrity["summary"]["product_calls"],
            "manager_owner_rows": integrity["summary"]["manager_owner_rows"],
            "pending_owner_mappings": integrity["summary"]["pending_owner_mappings"],
            "calls_with_crm_owner": integrity["summary"]["calls_with_crm_owner"],
        },
        "init": init_report,
        "import": import_report,
        "integrity": integrity,
        "safety": {
            "stable_runtime_writes": False,
            "runtime_db_writes": False,
            "asr_run": False,
            "ra_run": False,
            "crm_writes": False,
        },
    }
    if audit_out:
        write_json(audit_out, report)
    return report


def audit_product_db(db_path: Path, out_allowed_root: Path) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    guard_product_db_path(db_path, out_allowed_root, must_exist=True)
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        return audit_product_db_connection(con, db_path)


def upgrade_product_db(
    db_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(db_path, out_allowed_root, must_exist=True)
    if out_path and not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"upgrade audit must stay under product root: {out_allowed_root}")

    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        before = audit_product_db_connection(con, db_path)
        create_product_db_schema(con)
        applied = apply_product_db_migrations(con)
        seed_default_retention_policies(con, now_utc())
        con.commit()
        after = audit_product_db_connection(con, db_path)

    report = {
        "summary": ProductDbAdminSummary(
            schema_version=PRODUCT_DB_ADMIN_SCHEMA_VERSION,
            db_path=str(db_path),
            operation="upgrade",
            validation_ok=bool(after["summary"]["validation_ok"]),
            blocked=int(after["summary"]["blocked"]),
            warnings=int(after["summary"]["warnings"]),
        ).to_json_dict()
        | {
            "migrations_applied": applied,
            "schema_migrations_before": before["summary"]["schema_migrations"],
            "schema_migrations_after": after["summary"]["schema_migrations"],
        },
        "before_integrity": before,
        "after_integrity": after,
    }
    if out_path:
        write_json(out_path, report)
    return report


def snapshot_tenant_config(
    product_db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    snapshot_reason: str = "manual_snapshot",
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    config_path = config_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, out_allowed_root, must_exist=True)
    if not path_is_relative_to(config_path, out_allowed_root):
        raise ValueError(f"tenant config must stay under product root: {out_allowed_root}")
    if out_path and not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"tenant config snapshot audit must stay under product root: {out_allowed_root}")

    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        create_product_db_schema(con)
        apply_product_db_migrations(con)
        result = snapshot_tenant_config_connection(
            con,
            config_path=config_path,
            snapshot_reason=snapshot_reason,
            now=now_utc(),
        )
        con.commit()
        integrity = audit_product_db_connection(con, product_db_path)

    report = {
        "summary": ProductDbAdminSummary(
            schema_version=PRODUCT_DB_ADMIN_SCHEMA_VERSION,
            db_path=str(product_db_path),
            operation="snapshot_tenant_config",
            validation_ok=bool(integrity["summary"]["validation_ok"]),
            blocked=int(integrity["summary"]["blocked"]),
            warnings=int(integrity["summary"]["warnings"]),
        ).to_json_dict()
        | result,
        "integrity": integrity,
    }
    if out_path:
        write_json(out_path, report)
    return report


def audit_product_retention(
    db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(db_path, product_root, must_exist=True)
    if out_path and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"retention audit must stay under product root: {product_root}")

    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        create_product_db_schema(con)
        apply_product_db_migrations(con)
        seed_default_retention_policies(con, now_utc())
        con.commit()
        policies = read_retention_policies(con)
        integrity = audit_product_db_connection(con, db_path)

    artifact_rows = retention_artifact_rows(product_root, policies)
    enabled_policies = [policy for policy in policies if int(policy.get("enabled") or 0)]
    review_candidates = [row for row in artifact_rows if row["review_due"]]
    report = {
        "summary": ProductDbAdminSummary(
            schema_version=PRODUCT_DB_ADMIN_SCHEMA_VERSION,
            db_path=str(db_path),
            operation="retention_audit",
            validation_ok=bool(integrity["summary"]["validation_ok"]),
            blocked=int(integrity["summary"]["blocked"]),
            warnings=len(review_candidates) + int(integrity["summary"]["warnings"]),
        ).to_json_dict()
        | {
            "product_root": str(product_root),
            "policies": len(policies),
            "enabled_policies": len(enabled_policies),
            "artifacts_scanned": len(artifact_rows),
            "review_candidates": len(review_candidates),
        },
        "policies": policies,
        "artifact_rows": artifact_rows,
        "review_candidates": review_candidates,
        "integrity": integrity,
        "safety": {
            "deletes_files": False,
            "deletes_db_rows": False,
            "review_only": True,
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def restore_product_db_from_backup(
    backup_path: Path,
    product_db_path: Path,
    out_allowed_root: Path,
    replace_existing: bool = False,
    pre_restore_backup_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    backup_path = backup_path.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    pre_restore_backup_path = pre_restore_backup_path.resolve(strict=False) if pre_restore_backup_path else None
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, out_allowed_root, must_exist=False)
    if product_db_path.name in RUNTIME_DB_FILENAMES or "stable_runtime" in product_db_path.parts:
        raise ValueError("refusing to restore into runtime or stable_runtime DB")
    if not path_is_relative_to(backup_path, out_allowed_root):
        raise ValueError(f"backup must stay under product root: {out_allowed_root}")
    if not backup_path.exists() or not backup_path.is_file():
        raise FileNotFoundError(f"backup DB not found: {backup_path}")
    if pre_restore_backup_path and not path_is_relative_to(pre_restore_backup_path, out_allowed_root):
        raise ValueError(f"pre-restore backup must stay under product root: {out_allowed_root}")
    if out_path and not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"restore audit must stay under product root: {out_allowed_root}")
    if product_db_path.exists() and not replace_existing:
        raise FileExistsError(f"product DB already exists: {product_db_path}")

    pre_restore_backup = None
    if product_db_path.exists() and pre_restore_backup_path:
        pre_restore_backup = backup_product_db(product_db_path, pre_restore_backup_path, out_allowed_root)
    if replace_existing:
        remove_sqlite_db(product_db_path, out_allowed_root)
    product_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, product_db_path)
    upgrade_report = upgrade_product_db(product_db_path, out_allowed_root)
    integrity = audit_product_db(product_db_path, out_allowed_root)
    report = {
        "summary": ProductDbAdminSummary(
            schema_version=PRODUCT_DB_ADMIN_SCHEMA_VERSION,
            db_path=str(product_db_path),
            operation="restore",
            validation_ok=bool(integrity["summary"]["validation_ok"]),
            blocked=int(integrity["summary"]["blocked"]),
            warnings=int(integrity["summary"]["warnings"]),
        ).to_json_dict()
        | {
            "backup_path": str(backup_path),
            "pre_restore_backup_path": pre_restore_backup.get("backup_path") if pre_restore_backup else None,
            "restored_size_bytes": product_db_path.stat().st_size,
        },
        "pre_restore_backup": pre_restore_backup,
        "upgrade": upgrade_report,
        "integrity": integrity,
    }
    if out_path:
        write_json(out_path, report)
    return report


def apply_tenant_owner_config_to_product_db_dry_run(
    product_db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    return _apply_tenant_owner_config_to_product_db(
        product_db_path=product_db_path,
        config_path=config_path,
        out_allowed_root=out_allowed_root,
        out_path=out_path,
        apply_changes=False,
    )


def apply_tenant_owner_config_to_product_db(
    product_db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    return _apply_tenant_owner_config_to_product_db(
        product_db_path=product_db_path,
        config_path=config_path,
        out_allowed_root=out_allowed_root,
        out_path=out_path,
        apply_changes=True,
    )


def _apply_tenant_owner_config_to_product_db(
    product_db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path],
    apply_changes: bool,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    config_path = config_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, out_allowed_root, must_exist=True)
    if not path_is_relative_to(config_path, out_allowed_root):
        raise ValueError(f"tenant owner config must stay under product root: {out_allowed_root}")
    if out_path and not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"owner config audit must stay under product root: {out_allowed_root}")

    config = load_config(config_path)
    tenant_id = clean(config.get("tenant_id"))
    provider = clean(config.get("provider"))
    entries = list(config.get("manager_owner_overrides") or [])
    if not tenant_id:
        raise ValueError("tenant owner config tenant_id must not be empty")
    if not provider:
        raise ValueError("tenant owner config provider must not be empty")

    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        before = audit_product_db_connection(con, product_db_path)
        rows = read_product_owner_rows(con, tenant_id=tenant_id, provider=provider)
        call_counts = read_product_call_counts_by_manager(con, tenant_id=tenant_id, provider=provider)
        actions = build_owner_config_actions(
            entries=entries,
            rows=rows,
            call_counts=call_counts,
            tenant_id=tenant_id,
            provider=provider,
        )
        blocked = sum(1 for action in actions if str(action["action"]).startswith("BLOCK_"))
        applied = 0
        if apply_changes and blocked == 0:
            now = now_utc()
            for action in actions:
                if action["action"] not in {"WOULD_SET_OWNER", "WOULD_CONFIRM_EXISTING"}:
                    continue
                apply_owner_config_action(con, action, config_ref=str(config_path), now=now)
                applied += 1
            snapshot_tenant_config_connection(
                con,
                config_path=config_path,
                snapshot_reason="owner_config_apply",
                now=now,
            )
            con.commit()
        after = audit_product_db_connection(con, product_db_path)

    complete_entries = sum(1 for entry in entries if entry_has_owner(entry))
    missing_entries = len(entries) - complete_entries
    calls_would_gain_owner = sum(
        int(action.get("calls_affected") or 0)
        for action in actions
        if action["action"] == "WOULD_SET_OWNER" and not action.get("previous_crm_owner_id")
    )
    summary = ProductOwnerConfigApplySummary(
        schema_version=OWNER_CONFIG_APPLY_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        config_path=str(config_path),
        tenant_id=tenant_id,
        provider=provider,
        config_entries=len(entries),
        manager_owner_rows=len(rows),
        complete_owner_entries=complete_entries,
        missing_owner_entries=missing_entries,
        would_set_owner=sum(1 for action in actions if action["action"] == "WOULD_SET_OWNER"),
        would_confirm_existing=sum(1 for action in actions if action["action"] == "WOULD_CONFIRM_EXISTING"),
        applied=applied,
        calls_would_gain_owner=calls_would_gain_owner,
        calls_with_crm_owner_before=int(before["summary"]["calls_with_crm_owner"]),
        calls_with_crm_owner_after=int(after["summary"]["calls_with_crm_owner"]),
        pending_owner_mappings_before=int(before["summary"]["pending_owner_mappings"]),
        pending_owner_mappings_after=int(after["summary"]["pending_owner_mappings"]),
        dry_run=not apply_changes,
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=0 if blocked == 0 else blocked,
    )
    report = {
        "summary": summary.to_json_dict(),
        "actions": actions,
        "before_integrity": before,
        "after_integrity": after,
        "safety": {
            "stable_runtime_writes": False,
            "runtime_db_writes": False,
            "asr_run": False,
            "ra_run": False,
            "crm_writes": False,
            "product_db_writes": bool(apply_changes and blocked == 0),
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def read_product_owner_rows(
    con: sqlite3.Connection,
    tenant_id: str,
    provider: str,
) -> Mapping[str, Mapping[str, Any]]:
    rows = con.execute(
        """
        SELECT *
          FROM tenant_manager_owner_map
         WHERE tenant_id = ?
           AND telephony_provider = ?
         ORDER BY manager_extension
        """,
        (tenant_id, provider),
    ).fetchall()
    return {clean(row["manager_extension"]): dict(row) for row in rows}


def read_product_call_counts_by_manager(
    con: sqlite3.Connection,
    tenant_id: str,
    provider: str,
) -> Mapping[str, int]:
    rows = con.execute(
        """
        SELECT manager_extension, count(*) AS n
          FROM product_calls
         WHERE tenant_id = ?
           AND telephony_provider = ?
         GROUP BY manager_extension
        """,
        (tenant_id, provider),
    ).fetchall()
    return {clean(row["manager_extension"]): int(row["n"] or 0) for row in rows}


def build_owner_config_actions(
    entries: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
    call_counts: Mapping[str, int],
    tenant_id: str,
    provider: str,
) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        extension = clean(entry.get("manager_extension"))
        if not extension:
            actions.append({"action": "BLOCK_INVALID_CONFIG", "reason": "manager_extension_required"})
            continue
        if extension in seen:
            actions.append(
                {
                    "action": "BLOCK_DUPLICATE_CONFIG_ENTRY",
                    "reason": "duplicate_manager_extension",
                    "tenant_id": tenant_id,
                    "provider": provider,
                    "manager_extension": extension,
                }
            )
            continue
        seen.add(extension)
        row = rows.get(extension)
        if not row:
            actions.append(
                {
                    "action": "BLOCK_UNKNOWN_MANAGER_EXTENSION",
                    "reason": "manager_extension_not_in_product_db",
                    "tenant_id": tenant_id,
                    "provider": provider,
                    "manager_extension": extension,
                }
            )
            continue
        owner_id = optional_int(entry.get("crm_owner_id"))
        owner_name = clean(entry.get("crm_owner_name"))
        owner_email = clean(entry.get("crm_owner_email")) or None
        if owner_id is None or not owner_name:
            actions.append(
                {
                    "action": "BLOCK_MISSING_OWNER",
                    "reason": "crm_owner_id_and_crm_owner_name_required",
                    "tenant_id": tenant_id,
                    "provider": provider,
                    "manager_extension": extension,
                    "mango_name": clean(row.get("mango_name")) or clean(entry.get("mango_name")) or None,
                    "mango_email": clean(row.get("mango_email")) or clean(entry.get("mango_email")) or None,
                    "calls_affected": int(call_counts.get(extension, 0)),
                }
            )
            continue
        previous_owner = optional_int(row.get("crm_owner_id"))
        base = {
            "tenant_id": tenant_id,
            "provider": provider,
            "manager_extension": extension,
            "mango_name": clean(row.get("mango_name")) or clean(entry.get("mango_name")) or None,
            "mango_email": clean(row.get("mango_email")) or clean(entry.get("mango_email")) or None,
            "crm_owner_id": owner_id,
            "crm_owner_name": owner_name,
            "crm_owner_email": owner_email,
            "previous_crm_owner_id": previous_owner,
            "previous_crm_owner_name": clean(row.get("crm_owner_name")) or None,
            "calls_affected": int(call_counts.get(extension, 0)),
            "confirmed_by": clean(entry.get("confirmed_by")) or None,
            "notes": clean(entry.get("notes")) or "tenant_owner_config",
        }
        if previous_owner == owner_id:
            actions.append({"action": "WOULD_CONFIRM_EXISTING", **base})
        else:
            actions.append({"action": "WOULD_SET_OWNER", **base})

    missing_config = sorted(set(rows) - seen)
    for extension in missing_config:
        row = rows[extension]
        actions.append(
            {
                "action": "BLOCK_MISSING_CONFIG_ENTRY",
                "reason": "manager_extension_missing_from_config",
                "tenant_id": tenant_id,
                "provider": provider,
                "manager_extension": extension,
                "mango_name": clean(row.get("mango_name")) or None,
                "mango_email": clean(row.get("mango_email")) or None,
                "calls_affected": int(call_counts.get(extension, 0)),
            }
        )
    return actions


def apply_owner_config_action(
    con: sqlite3.Connection,
    action: Mapping[str, Any],
    config_ref: str,
    now: str,
) -> None:
    match_status = "config_confirmed" if action["action"] == "WOULD_CONFIRM_EXISTING" else "manual_override"
    con.execute(
        """
        UPDATE tenant_manager_owner_map
           SET crm_owner_id = ?,
               crm_owner_name = ?,
               crm_owner_email = ?,
               decision_status = 'confirmed_candidate',
               match_status = ?,
               config_ref = ?,
               notes = ?,
               updated_at = ?
         WHERE tenant_id = ?
           AND telephony_provider = ?
           AND manager_extension = ?
        """,
        (
            action["crm_owner_id"],
            action["crm_owner_name"],
            action.get("crm_owner_email"),
            match_status,
            config_ref,
            action.get("notes"),
            now,
            action["tenant_id"],
            action["provider"],
            action["manager_extension"],
        ),
    )
    con.execute(
        """
        UPDATE product_calls
           SET crm_owner_id = ?,
               crm_owner_name = ?,
               crm_match_status = ?,
               updated_at = ?
         WHERE tenant_id = ?
           AND telephony_provider = ?
           AND manager_extension = ?
        """,
        (
            action["crm_owner_id"],
            action["crm_owner_name"],
            match_status,
            now,
            action["tenant_id"],
            action["provider"],
            action["manager_extension"],
        ),
    )


def entry_has_owner(entry: Mapping[str, Any]) -> bool:
    return optional_int(entry.get("crm_owner_id")) is not None and bool(clean(entry.get("crm_owner_name")))


def create_product_db_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          migration_id TEXT PRIMARY KEY,
          schema_version TEXT NOT NULL,
          applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tenants (
          tenant_id TEXT PRIMARY KEY,
          display_name TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS provider_accounts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          mode TEXT NOT NULL,
          config_ref TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(tenant_id, provider),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE TABLE IF NOT EXISTS crm_accounts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          mode TEXT NOT NULL,
          config_ref TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(tenant_id, provider),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE TABLE IF NOT EXISTS tenant_manager_owner_map (
          tenant_id TEXT NOT NULL,
          telephony_provider TEXT NOT NULL,
          manager_extension TEXT NOT NULL,
          mango_name TEXT,
          mango_email TEXT,
          crm_provider TEXT NOT NULL DEFAULT 'amocrm',
          crm_owner_id INTEGER,
          crm_owner_name TEXT,
          crm_owner_email TEXT,
          decision_status TEXT NOT NULL,
          match_status TEXT NOT NULL,
          source_ref TEXT,
          config_ref TEXT,
          notes TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(tenant_id, telephony_provider, manager_extension),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE TABLE IF NOT EXISTS product_calls (
          tenant_id TEXT NOT NULL,
          telephony_provider TEXT NOT NULL,
          provider_call_id TEXT NOT NULL,
          event_key TEXT NOT NULL,
          recording_id TEXT,
          source_filename TEXT NOT NULL,
          started_at TEXT,
          duration_sec REAL,
          manager_extension TEXT,
          manager_display_name TEXT,
          crm_owner_id INTEGER,
          crm_owner_name TEXT,
          crm_match_status TEXT,
          raw_payload_ref TEXT,
          source_repository_ref TEXT NOT NULL,
          imported_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(tenant_id, telephony_provider, provider_call_id),
          UNIQUE(event_key),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id),
          FOREIGN KEY(tenant_id, telephony_provider, manager_extension)
            REFERENCES tenant_manager_owner_map(tenant_id, telephony_provider, manager_extension)
        );

        CREATE TABLE IF NOT EXISTS job_types (
          job_type TEXT PRIMARY KEY,
          description TEXT NOT NULL,
          default_mode TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS job_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          job_type TEXT NOT NULL,
          tenant_id TEXT,
          status TEXT NOT NULL,
          planned_at TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          input_ref TEXT,
          output_ref TEXT,
          error TEXT,
          FOREIGN KEY(job_type) REFERENCES job_types(job_type),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE TABLE IF NOT EXISTS tenant_config_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          config_kind TEXT NOT NULL,
          config_ref TEXT NOT NULL,
          source_path TEXT,
          content_hash TEXT NOT NULL,
          content_json TEXT NOT NULL,
          snapshot_reason TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(tenant_id, config_kind, content_hash),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE TABLE IF NOT EXISTS retention_policies (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          target TEXT NOT NULL,
          retention_days INTEGER NOT NULL,
          action TEXT NOT NULL,
          enabled INTEGER NOT NULL,
          notes TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(tenant_id, target, action),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
        );

        CREATE INDEX IF NOT EXISTS ix_product_calls_started_at ON product_calls(started_at);
        CREATE INDEX IF NOT EXISTS ix_product_calls_manager ON product_calls(tenant_id, telephony_provider, manager_extension);
        CREATE INDEX IF NOT EXISTS ix_product_calls_crm_owner ON product_calls(crm_owner_id);
        CREATE INDEX IF NOT EXISTS ix_job_runs_status ON job_runs(status, planned_at);
        CREATE INDEX IF NOT EXISTS ix_tenant_config_history_tenant ON tenant_config_history(tenant_id, config_kind, created_at);
        CREATE INDEX IF NOT EXISTS ix_retention_policies_tenant ON retention_policies(tenant_id, target);
        """
    )


def apply_base_migration(con: sqlite3.Connection) -> int:
    row = con.execute("select 1 from schema_migrations where migration_id = ?", (PRODUCT_DB_MIGRATION_ID,)).fetchone()
    if row:
        return 0
    con.execute(
        """
        INSERT INTO schema_migrations (migration_id, schema_version, applied_at)
        VALUES (?, ?, ?)
        """,
        (PRODUCT_DB_MIGRATION_ID, PRODUCT_DB_SCHEMA_VERSION, now_utc()),
    )
    return 1


def apply_product_db_migrations(con: sqlite3.Connection) -> int:
    applied = 0
    applied += apply_base_migration(con)
    applied += apply_retention_config_history_migration(con)
    applied += apply_scheduler_runtime_migration(con)
    applied += apply_capture_inbox_migration(con)
    return applied


def apply_retention_config_history_migration(con: sqlite3.Connection) -> int:
    row = con.execute(
        "select 1 from schema_migrations where migration_id = ?",
        (PRODUCT_DB_RETENTION_MIGRATION_ID,),
    ).fetchone()
    if row:
        return 0
    create_product_db_schema(con)
    con.execute(
        """
        INSERT INTO schema_migrations (migration_id, schema_version, applied_at)
        VALUES (?, ?, ?)
        """,
        (PRODUCT_DB_RETENTION_MIGRATION_ID, PRODUCT_DB_SCHEMA_VERSION, now_utc()),
    )
    return 1


def apply_scheduler_runtime_migration(con: sqlite3.Connection) -> int:
    row = con.execute(
        "select 1 from schema_migrations where migration_id = ?",
        (PRODUCT_DB_SCHEDULER_MIGRATION_ID,),
    ).fetchone()
    if row:
        return 0
    for column, definition in (
        ("scheduled_for", "TEXT"),
        ("next_run_at", "TEXT"),
        ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ("max_attempts", "INTEGER NOT NULL DEFAULT 3"),
        ("lock_owner", "TEXT"),
        ("lock_expires_at", "TEXT"),
        ("heartbeat_at", "TEXT"),
        ("result_json", "TEXT"),
    ):
        add_column_if_missing(con, "job_runs", column, definition)
    con.execute("CREATE INDEX IF NOT EXISTS ix_job_runs_next_run ON job_runs(status, next_run_at)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_job_runs_lock ON job_runs(lock_owner, lock_expires_at)")
    con.execute(
        """
        INSERT INTO schema_migrations (migration_id, schema_version, applied_at)
        VALUES (?, ?, ?)
        """,
        (PRODUCT_DB_SCHEDULER_MIGRATION_ID, PRODUCT_DB_SCHEMA_VERSION, now_utc()),
    )
    return 1


def apply_capture_inbox_migration(con: sqlite3.Connection) -> int:
    row = con.execute(
        "select 1 from schema_migrations where migration_id = ?",
        (PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID,),
    ).fetchone()
    if row:
        return 0
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS capture_inbox_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          event_key TEXT NOT NULL,
          provider_call_id TEXT NOT NULL,
          status TEXT NOT NULL,
          source_job_run_id INTEGER,
          source_report_ref TEXT,
          raw_payload_ref TEXT,
          started_at TEXT,
          ended_at TEXT,
          direction TEXT,
          client_phone TEXT,
          manager_ref TEXT,
          recording_ref TEXT,
          recording_url TEXT,
          audio_ref TEXT,
          decision_reason TEXT,
          candidate_json TEXT,
          event_json TEXT,
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          enqueue_count INTEGER NOT NULL DEFAULT 1,
          reserved_by TEXT,
          reserved_at TEXT,
          error TEXT,
          UNIQUE(tenant_id, provider, event_key),
          FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id),
          FOREIGN KEY(source_job_run_id) REFERENCES job_runs(id)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS ix_capture_inbox_status ON capture_inbox_items(status, last_seen_at)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_capture_inbox_tenant_status ON capture_inbox_items(tenant_id, status, started_at)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_capture_inbox_source_job ON capture_inbox_items(source_job_run_id)")
    con.execute(
        """
        INSERT INTO schema_migrations (migration_id, schema_version, applied_at)
        VALUES (?, ?, ?)
        """,
        (PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID, PRODUCT_DB_SCHEMA_VERSION, now_utc()),
    )
    return 1


def add_column_if_missing(con: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {clean(row["name"]) for row in con.execute(f"pragma table_info({table})")}
    if column not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def upsert_manager_owner_row(
    con: sqlite3.Connection,
    row: ManagerRollupItem,
    source_ref: str,
    config_ref: Optional[str],
    now: str,
) -> None:
    decision_status = "confirmed_candidate" if row.crm_owner_id is not None else "needs_manual_owner"
    con.execute(
        """
        INSERT INTO tenant_manager_owner_map (
          tenant_id, telephony_provider, manager_extension,
          mango_name, mango_email, crm_provider,
          crm_owner_id, crm_owner_name, crm_owner_email,
          decision_status, match_status, source_ref, config_ref, notes,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'amocrm', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tenant_id, telephony_provider, manager_extension) DO UPDATE SET
          mango_name = excluded.mango_name,
          mango_email = excluded.mango_email,
          crm_owner_id = excluded.crm_owner_id,
          crm_owner_name = excluded.crm_owner_name,
          crm_owner_email = excluded.crm_owner_email,
          decision_status = excluded.decision_status,
          match_status = excluded.match_status,
          source_ref = excluded.source_ref,
          config_ref = excluded.config_ref,
          notes = excluded.notes,
          updated_at = excluded.updated_at
        """,
        (
            row.tenant_id,
            row.provider,
            row.manager_extension,
            row.mango_name,
            row.mango_email,
            row.crm_owner_id,
            row.crm_owner_name,
            row.crm_owner_email,
            decision_status,
            row.crm_match_status,
            source_ref,
            config_ref,
            "tenant_review_required" if row.crm_owner_id is None else None,
            now,
            now,
        ),
    )


def upsert_product_call(con: sqlite3.Connection, call: ProductCallRecord, source_ref: str, now: str) -> None:
    con.execute(
        """
        INSERT INTO product_calls (
          tenant_id, telephony_provider, provider_call_id, event_key, recording_id,
          source_filename, started_at, duration_sec, manager_extension, manager_display_name,
          crm_owner_id, crm_owner_name, crm_match_status, raw_payload_ref,
          source_repository_ref, imported_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tenant_id, telephony_provider, provider_call_id) DO UPDATE SET
          event_key = excluded.event_key,
          recording_id = excluded.recording_id,
          source_filename = excluded.source_filename,
          started_at = excluded.started_at,
          duration_sec = excluded.duration_sec,
          manager_extension = excluded.manager_extension,
          manager_display_name = excluded.manager_display_name,
          crm_owner_id = excluded.crm_owner_id,
          crm_owner_name = excluded.crm_owner_name,
          crm_match_status = excluded.crm_match_status,
          raw_payload_ref = excluded.raw_payload_ref,
          source_repository_ref = excluded.source_repository_ref,
          updated_at = excluded.updated_at
        """,
        (
            call.tenant_id,
            call.provider,
            call.provider_call_id,
            call.event_key,
            call.recording_id,
            call.source_filename,
            call.started_at,
            call.duration_sec,
            call.manager_extension,
            call.manager_display_name,
            call.manager_crm_owner_id,
            call.manager_crm_owner_name,
            call.manager_crm_match_status,
            call.raw_payload_ref,
            source_ref,
            now,
            now,
        ),
    )


def seed_job_types(con: sqlite3.Connection, now: str) -> None:
    rows = [
        ("shadow_poll", "Read-only telephony poll and JSON payload archive", "dry_run"),
        ("capture_download", "Controlled recording download into product storage", "disabled"),
        ("asr", "Speech recognition processing", "disabled"),
        ("ra", "Resolve and analysis processing", "disabled"),
        ("crm_sync", "CRM write queue with explicit approval", "disabled"),
    ]
    for job_type, description, default_mode in rows:
        con.execute(
            """
            INSERT INTO job_types (job_type, description, default_mode, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_type) DO UPDATE SET
              description = excluded.description,
              default_mode = excluded.default_mode
            """,
            (job_type, description, default_mode, now),
        )


def seed_default_retention_policies(con: sqlite3.Connection, now: str) -> None:
    tenants = [clean(row["tenant_id"]) for row in con.execute("select tenant_id from tenants order by tenant_id")]
    for tenant_id in tenants:
        for target, retention_days, action, enabled, notes in DEFAULT_RETENTION_POLICIES:
            con.execute(
                """
                INSERT INTO retention_policies (
                  tenant_id, target, retention_days, action, enabled, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, target, action) DO UPDATE SET
                  retention_days = excluded.retention_days,
                  enabled = excluded.enabled,
                  notes = excluded.notes,
                  updated_at = excluded.updated_at
                """,
                (tenant_id, target, int(retention_days), action, int(enabled), notes, now, now),
            )


def snapshot_tenant_config_connection(
    con: sqlite3.Connection,
    config_path: Path,
    snapshot_reason: str,
    now: str,
) -> Mapping[str, Any]:
    config = load_config(config_path)
    tenant_id = clean(config.get("tenant_id"))
    if not tenant_id:
        raise ValueError("tenant config tenant_id must not be empty")
    canonical = canonical_json(config)
    content_hash = sha256_text(canonical)
    before = scalar_int(
        con,
        """
        SELECT count(*)
          FROM tenant_config_history
         WHERE tenant_id = ?
           AND config_kind = 'tenant_owner_mapping'
           AND content_hash = ?
        """,
        (tenant_id, content_hash),
    )
    con.execute(
        """
        INSERT OR IGNORE INTO tenant_config_history (
          tenant_id, config_kind, config_ref, source_path,
          content_hash, content_json, snapshot_reason, created_at
        ) VALUES (?, 'tenant_owner_mapping', ?, ?, ?, ?, ?, ?)
        """,
        (
            tenant_id,
            str(config_path),
            str(config_path),
            content_hash,
            canonical,
            clean(snapshot_reason) or "snapshot",
            now,
        ),
    )
    after = scalar_int(
        con,
        """
        SELECT count(*)
          FROM tenant_config_history
         WHERE tenant_id = ?
           AND config_kind = 'tenant_owner_mapping'
           AND content_hash = ?
        """,
        (tenant_id, content_hash),
    )
    return {
        "tenant_id": tenant_id,
        "config_kind": "tenant_owner_mapping",
        "config_path": str(config_path),
        "content_hash": content_hash,
        "inserted": 1 if before == 0 and after == 1 else 0,
        "already_present": before > 0,
    }


def audit_product_db_connection(con: sqlite3.Connection, db_path: Path) -> Mapping[str, Any]:
    tables = {
        name: relation_count(con, name)
        for name in (
            "schema_migrations",
            "tenants",
            "provider_accounts",
            "crm_accounts",
            "tenant_manager_owner_map",
            "product_calls",
            "job_types",
            "job_runs",
            "capture_inbox_items",
            "tenant_config_history",
            "retention_policies",
        )
    }
    calls_with_owner = scalar_int(con, "select count(*) from product_calls where crm_owner_id is not null")
    job_runs = tables["job_runs"]
    due_job_runs = count_due_job_runs(con)
    running_job_runs = scalar_int(con, "select count(*) from job_runs where status = 'running'")
    failed_job_runs = scalar_int(con, "select count(*) from job_runs where status = 'failed'")
    capture_inbox_items = tables["capture_inbox_items"]
    capture_inbox_ready = scalar_int(con, "select count(*) from capture_inbox_items where status = 'ready_for_capture'")
    capture_inbox_blocked = scalar_int(con, "select count(*) from capture_inbox_items where status like 'blocked%'")
    pending = scalar_int(con, "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'")
    raw_refs = scalar_int(con, "select count(*) from product_calls where raw_payload_ref is not null and raw_payload_ref != ''")
    orphan_calls = scalar_int(
        con,
        """
        SELECT count(*)
          FROM product_calls pc
          LEFT JOIN tenant_manager_owner_map mom
            ON mom.tenant_id = pc.tenant_id
           AND mom.telephony_provider = pc.telephony_provider
           AND mom.manager_extension = pc.manager_extension
         WHERE mom.manager_extension IS NULL
        """,
    )
    duplicate_events = scalar_int(
        con,
        """
        SELECT count(*)
          FROM (
            SELECT event_key
              FROM product_calls
             GROUP BY event_key
            HAVING count(*) > 1
          )
        """,
    )
    missing_migrations = missing_required_migrations(con)
    blocked_reasons = {
        "missing_required_migrations": len(missing_migrations),
        "orphan_calls_without_owner_map": orphan_calls,
        "duplicate_event_keys": duplicate_events,
    }
    warning_reasons = {
        "pending_owner_mappings": pending,
    }
    summary = ProductDbIntegritySummary(
        schema_version=PRODUCT_DB_SCHEMA_VERSION,
        db_path=str(db_path),
        schema_migrations=tables["schema_migrations"],
        tenants=tables["tenants"],
        manager_owner_rows=tables["tenant_manager_owner_map"],
        product_calls=tables["product_calls"],
        job_runs=job_runs,
        due_job_runs=due_job_runs,
        running_job_runs=running_job_runs,
        failed_job_runs=failed_job_runs,
        capture_inbox_items=capture_inbox_items,
        capture_inbox_ready=capture_inbox_ready,
        capture_inbox_blocked=capture_inbox_blocked,
        calls_with_crm_owner=calls_with_owner,
        pending_owner_mappings=pending,
        raw_payload_refs_present=raw_refs,
        job_types=tables["job_types"],
        tenant_config_history=tables["tenant_config_history"],
        retention_policies=tables["retention_policies"],
        validation_ok=sum(blocked_reasons.values()) == 0,
        blocked=sum(blocked_reasons.values()),
        warnings=sum(warning_reasons.values()),
    )
    return {
        "summary": summary.to_json_dict(),
        "tables": tables,
        "blocked_reasons": blocked_reasons,
        "missing_migrations": missing_migrations,
        "warning_reasons": warning_reasons,
        "manager_owner_status_counts": manager_owner_status_counts(con),
        "call_owner_status_counts": call_owner_status_counts(con),
        "job_status_counts": job_status_counts(con),
        "capture_inbox_status_counts": capture_inbox_status_counts(con),
    }


def manager_owner_status_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    rows = con.execute(
        """
        SELECT decision_status, count(*) AS n
          FROM tenant_manager_owner_map
         GROUP BY decision_status
         ORDER BY decision_status
        """
    ).fetchall()
    return {clean(row["decision_status"]): int(row["n"] or 0) for row in rows}


def call_owner_status_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    rows = con.execute(
        """
        SELECT
          CASE WHEN crm_owner_id IS NULL THEN 'missing_owner' ELSE 'has_owner' END AS status,
          count(*) AS n
        FROM product_calls
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()
    return {clean(row["status"]): int(row["n"] or 0) for row in rows}


def job_status_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    if not relation_exists(con, "job_runs"):
        return {}
    rows = con.execute(
        """
        SELECT status, count(*) AS n
          FROM job_runs
         GROUP BY status
         ORDER BY status
        """
    ).fetchall()
    return {clean(row["status"]): int(row["n"] or 0) for row in rows}


def capture_inbox_status_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    if not relation_exists(con, "capture_inbox_items"):
        return {}
    rows = con.execute(
        """
        SELECT status, count(*) AS n
          FROM capture_inbox_items
         GROUP BY status
         ORDER BY status
        """
    ).fetchall()
    return {clean(row["status"]): int(row["n"] or 0) for row in rows}


def count_due_job_runs(con: sqlite3.Connection) -> int:
    if not relation_exists(con, "job_runs"):
        return 0
    columns = {clean(row["name"]) for row in con.execute("pragma table_info(job_runs)")}
    if "next_run_at" not in columns:
        return 0
    now = now_utc()
    return scalar_int(
        con,
        """
        SELECT count(*)
          FROM job_runs
         WHERE status IN ('planned', 'retry_wait')
           AND coalesce(next_run_at, planned_at) <= ?
        """,
        (now,),
    )


def relation_count(con: sqlite3.Connection, name: str) -> int:
    if not relation_exists(con, name):
        return 0
    return scalar_int(con, f"select count(*) from {name}")


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute("select 1 from sqlite_master where type = 'table' and name = ?", (name,)).fetchone()
    return row is not None


def migration_applied(con: sqlite3.Connection) -> bool:
    if not relation_exists(con, "schema_migrations"):
        return False
    row = con.execute("select 1 from schema_migrations where migration_id = ?", (PRODUCT_DB_MIGRATION_ID,)).fetchone()
    return row is not None


def missing_required_migrations(con: sqlite3.Connection) -> list[str]:
    if not relation_exists(con, "schema_migrations"):
        return list(PRODUCT_DB_REQUIRED_MIGRATIONS)
    rows = con.execute("select migration_id from schema_migrations").fetchall()
    applied = {clean(row["migration_id"] if isinstance(row, sqlite3.Row) else row[0]) for row in rows}
    return [migration_id for migration_id in PRODUCT_DB_REQUIRED_MIGRATIONS if migration_id not in applied]


def scalar_int(con: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> int:
    try:
        row = con.execute(sql, tuple(params)).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row else 0


def read_retention_policies(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    if not relation_exists(con, "retention_policies"):
        return []
    rows = con.execute(
        """
        SELECT tenant_id, target, retention_days, action, enabled, notes, created_at, updated_at
          FROM retention_policies
         ORDER BY tenant_id, target, action
        """
    ).fetchall()
    return [dict(row) for row in rows]


def retention_artifact_rows(product_root: Path, policies: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    enabled_by_target = {clean(policy.get("target")): policy for policy in policies if int(policy.get("enabled") or 0)}
    for path in sorted(product_root.rglob("*")):
        if not path.is_file():
            continue
        target = retention_target_for_path(path, product_root)
        policy = enabled_by_target.get(target)
        if not policy:
            continue
        age_days = file_age_days(path)
        retention_days = int(policy.get("retention_days") or 0)
        rows.append(
            {
                "path": str(path),
                "target": target,
                "size_bytes": path.stat().st_size,
                "age_days": age_days,
                "retention_days": retention_days,
                "action": clean(policy.get("action")),
                "review_due": age_days > retention_days if retention_days >= 0 else False,
            }
        )
    return rows


def retention_target_for_path(path: Path, product_root: Path) -> str:
    rel = path.resolve(strict=False).relative_to(product_root.resolve(strict=False))
    parts = rel.parts
    suffix = path.suffix.lower()
    if parts and parts[0] == "backups" and suffix in {".sqlite", ".db"}:
        return "product_db_backup"
    if suffix == ".json":
        return "audit_json"
    if path.name in PRODUCT_DB_FILENAMES:
        return "product_db"
    return "other"


def file_age_days(path: Path) -> int:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - mtime).total_seconds() // 86400))


def canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def guard_product_db_path(db_path: Path, out_allowed_root: Path, must_exist: bool) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts:
        raise ValueError("refusing product DB under stable_runtime")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"product DB must stay under allowed root: {out_allowed_root}")
    if must_exist and (not db_path.exists() or not db_path.is_file()):
        raise FileNotFoundError(f"product DB not found: {db_path}")
    if not must_exist and db_path.name not in PRODUCT_DB_FILENAMES:
        raise ValueError(f"product DB filename must be one of: {sorted(PRODUCT_DB_FILENAMES)}")


def remove_sqlite_db(db_path: Path, out_allowed_root: Path) -> bool:
    removed = False
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        path = Path(f"{db_path}{suffix}")
        if not path.exists():
            continue
        if not path_is_relative_to(path.resolve(strict=False), out_allowed_root):
            raise ValueError(f"refusing to remove SQLite sidecar outside allowed root: {path}")
        if path.is_dir():
            raise ValueError(f"refusing to remove directory while replacing product DB: {path}")
        path.unlink()
        removed = True
    return removed


def backup_product_db(db_path: Path, backup_path: Path, out_allowed_root: Path) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    backup_path = backup_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    guard_product_db_path(db_path, out_allowed_root, must_exist=True)
    if not path_is_relative_to(backup_path, out_allowed_root):
        raise ValueError(f"backup must stay under allowed root: {out_allowed_root}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, backup_path)
    return {
        "schema_version": PRODUCT_DB_SCHEMA_VERSION,
        "db_path": str(db_path),
        "backup_path": str(backup_path),
        "size_bytes": backup_path.stat().st_size,
        "validation_ok": backup_path.exists(),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
