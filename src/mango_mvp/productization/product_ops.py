from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote

from mango_mvp.productization.appliance_config_wizard import build_appliance_config_wizard_report
from mango_mvp.productization.product_db import (
    audit_product_db,
    backup_product_db,
    guard_product_db_path,
)
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


PRODUCT_OPS_SCHEMA_VERSION = "product_ops_v1"


@dataclass(frozen=True)
class ProductOpsSummary:
    schema_version: str
    operation: str
    product_root: str
    product_db_path: str
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_product_ops_healthcheck(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    *,
    backup_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root, product_db_path, out_path = resolve_ops_paths(product_root, product_db_path, out_path)
    backup_dir = (backup_dir or product_root / "backups").resolve(strict=False)
    guard_under_root(backup_dir, product_root, "backup dir")
    checks = []
    integrity = audit_product_db(product_db_path, product_root)
    checks.append({"name": "product_db_integrity", "ok": bool(integrity["summary"]["validation_ok"]), "summary": integrity["summary"]})
    checks.append({"name": "backup_dir_writable", "ok": probe_backup_dir(backup_dir), "path": str(backup_dir)})
    checks.append({"name": "sqlite_sidecars", "ok": True, "files": sqlite_sidecars(product_db_path)})
    blocked = sum(1 for check in checks if not check["ok"])
    warnings = int(integrity["summary"]["warnings"])
    report = {
        "summary": ProductOpsSummary(
            schema_version=PRODUCT_OPS_SCHEMA_VERSION,
            operation="healthcheck",
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "checks": checks,
        "integrity": integrity,
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def run_product_db_backup(
    product_root: Path,
    product_db_path: Path,
    backup_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root, product_db_path, out_path = resolve_ops_paths(product_root, product_db_path, out_path)
    backup_path = backup_path.resolve(strict=False)
    guard_under_root(backup_path, product_root, "backup")
    backup = backup_product_db(product_db_path, backup_path, product_root)
    verify = verify_product_db_backup(product_root, product_db_path, backup_path)
    report = {
        "summary": ProductOpsSummary(
            schema_version=PRODUCT_OPS_SCHEMA_VERSION,
            operation="backup",
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            validation_ok=bool(backup["validation_ok"]) and bool(verify["summary"]["validation_ok"]),
            blocked=0 if bool(backup["validation_ok"]) and bool(verify["summary"]["validation_ok"]) else 1,
            warnings=0,
        ).to_json_dict()
        | {"backup_path": str(backup_path), "size_bytes": backup["size_bytes"]},
        "backup": backup,
        "verify": verify,
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def verify_product_db_backup(
    product_root: Path,
    product_db_path: Path,
    backup_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    backup_path = backup_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_under_root(backup_path, product_root, "backup")
    if out_path:
        guard_under_root(out_path, product_root, "backup verify output")
    blocked = 0
    checks = []
    if not backup_path.exists() or not backup_path.is_file():
        blocked += 1
        checks.append({"name": "backup_exists", "ok": False})
    else:
        checks.append({"name": "backup_exists", "ok": True, "size_bytes": backup_path.stat().st_size})
        try:
            quick_check = sqlite_quick_check(backup_path)
            checks.append({"name": "sqlite_quick_check", "ok": quick_check == "ok", "result": quick_check})
            blocked += 0 if quick_check == "ok" else 1
        except Exception as exc:
            blocked += 1
            checks.append({"name": "sqlite_quick_check", "ok": False, "error": str(exc)})
    report = {
        "summary": ProductOpsSummary(
            schema_version=PRODUCT_OPS_SCHEMA_VERSION,
            operation="verify_backup",
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=0,
        ).to_json_dict()
        | {"backup_path": str(backup_path)},
        "checks": checks,
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_restore_dry_run(
    product_root: Path,
    product_db_path: Path,
    backup_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    verify = verify_product_db_backup(product_root, product_db_path, backup_path)
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    backup_path = backup_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    if out_path:
        guard_under_root(out_path, product_root, "restore dry-run output")
    report = {
        "summary": ProductOpsSummary(
            schema_version=PRODUCT_OPS_SCHEMA_VERSION,
            operation="restore_dry_run",
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            validation_ok=bool(verify["summary"]["validation_ok"]),
            blocked=int(verify["summary"]["blocked"]),
            warnings=0,
        ).to_json_dict()
        | {
            "backup_path": str(backup_path),
            "would_replace_product_db": True,
            "execute_restore": False,
        },
        "verify": verify,
        "restore_plan": {
            "pre_restore_backup_required": True,
            "copy_backup_to_product_db": str(product_db_path),
            "run_upgrade_after_restore": True,
        },
        "safety": safety_contract() | {"restore_executed": False},
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_product_ops_diagnostics_bundle(
    product_root: Path,
    product_db_path: Path,
    out_dir: Optional[Path] = None,
    *,
    backup_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root, product_db_path, _ = resolve_ops_paths(product_root, product_db_path, None)
    out_dir = (out_dir or product_root / "ops" / "diagnostics_bundle").resolve(strict=False)
    backup_dir = (backup_dir or product_root / "backups").resolve(strict=False)
    guard_under_root(out_dir, product_root, "diagnostics bundle")
    guard_under_root(backup_dir, product_root, "backup dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    health = build_product_ops_healthcheck(
        product_root=product_root,
        product_db_path=product_db_path,
        out_path=out_dir / "healthcheck.json",
        backup_dir=backup_dir,
    )
    scheduler = build_scheduler_health_report(
        product_db_path=product_db_path,
        product_root=product_root,
        out_path=out_dir / "scheduler_health.json",
    )
    wizard = build_appliance_config_wizard_report(
        product_root=product_root,
        product_db_path=product_db_path,
        out_path=out_dir / "appliance_config_wizard.json",
        backup_dir=backup_dir,
    )
    demo = build_dashboard_demo_readiness(product_root=product_root, product_db_path=product_db_path)
    demo_path = out_dir / "demo_readiness.json"
    write_json(demo_path, demo)

    checks = [
        {"name": "healthcheck", "path": str(out_dir / "healthcheck.json"), "summary": health["summary"]},
        {"name": "scheduler_health", "path": str(out_dir / "scheduler_health.json"), "summary": scheduler["summary"]},
        {"name": "appliance_config_wizard", "path": str(out_dir / "appliance_config_wizard.json"), "summary": wizard["summary"]},
        {"name": "demo_readiness", "path": str(demo_path), "summary": demo["summary"]},
    ]
    blocked = sum(int(check["summary"].get("blocked") or 0) for check in checks)
    warnings = sum(int(check["summary"].get("warnings") or 0) for check in checks)
    report = {
        "summary": ProductOpsSummary(
            schema_version=PRODUCT_OPS_SCHEMA_VERSION,
            operation="diagnostics_bundle",
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict()
        | {"out_dir": str(out_dir), "artifacts": len(checks), "backup_executed": False, "restore_executed": False},
        "checks": checks,
        "commands": {
            "backup": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py "
                f"--product-root {product_root} --product-db {product_db_path} backup "
                f"--backup {backup_dir / 'mango_product_appliance_backup.sqlite'}"
            ),
            "restore_dry_run": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py "
                f"--product-root {product_root} --product-db {product_db_path} restore-dry-run "
                f"--backup {backup_dir / 'mango_product_appliance_backup.sqlite'}"
            ),
        },
        "safety": safety_contract() | {"backup_executed": False, "restore_executed": False},
    }
    write_json(out_dir / "diagnostics_manifest.json", report)
    return report


def sqlite_quick_check(db_path: Path) -> str:
    uri = f"file:{quote(str(db_path), safe='/:')}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as con:
        row = con.execute("PRAGMA quick_check").fetchone()
    return clean(row[0]) if row else ""


def probe_backup_dir(backup_dir: Path) -> bool:
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        probe = backup_dir / ".healthcheck_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def sqlite_sidecars(product_db_path: Path) -> list[str]:
    return [str(path) for path in (product_db_path, Path(f"{product_db_path}-wal"), Path(f"{product_db_path}-shm")) if path.exists()]


def resolve_ops_paths(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path]]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        guard_under_root(out_path, product_root, "ops output")
    return product_root, product_db_path, out_path


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "write_crm": False,
        "write_tallanto": False,
        "run_asr": False,
        "run_ra": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
