from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.crm_entity_resolver import build_crm_entity_resolution_report
from mango_mvp.productization.product_db import audit_product_db, audit_product_retention, guard_product_db_path
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


APPLIANCE_CONFIG_WIZARD_SCHEMA_VERSION = "appliance_config_wizard_v1"
CHECK_OK = "OK"
CHECK_WARN = "WARN"
CHECK_BLOCK = "BLOCK"


@dataclass(frozen=True)
class ApplianceConfigWizardSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    checks: int
    ok: int
    warn: int
    block: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_appliance_config_wizard_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    *,
    crm_snapshot_path: Optional[Path] = None,
    backup_dir: Optional[Path] = None,
    require_mango_credentials: bool = False,
    write_templates: bool = False,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    crm_snapshot_path = crm_snapshot_path.resolve(strict=False) if crm_snapshot_path else default_crm_snapshot(product_root)
    backup_dir = (backup_dir or product_root / "backups").resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=False)
    for label, path in (("wizard output", out_path), ("CRM snapshot", crm_snapshot_path), ("backup dir", backup_dir)):
        if path is None:
            continue
        guard_under_root(path, product_root, label)

    checks = []
    checks.append(check_product_root(product_root))
    checks.append(check_product_db(product_db_path, product_root))
    checks.append(check_mango_credentials(require=require_mango_credentials))
    checks.append(check_crm_snapshot(product_db_path, product_root, crm_snapshot_path))
    checks.append(check_backup_dir(backup_dir))
    checks.append(check_retention(product_db_path, product_root))
    checks.append(check_runtime_separation(product_root, product_db_path))
    checks.append(check_demo_readiness(product_root, product_db_path))
    counts = Counter(check["status"] for check in checks)
    blocked = int(counts[CHECK_BLOCK])
    warnings = int(counts[CHECK_WARN])
    templates = build_config_templates(product_root, product_db_path, crm_snapshot_path, backup_dir)
    written_templates = write_config_templates(product_root, templates) if write_templates else []
    report = {
        "summary": ApplianceConfigWizardSummary(
            schema_version=APPLIANCE_CONFIG_WIZARD_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            checks=len(checks),
            ok=int(counts[CHECK_OK]),
            warn=warnings,
            block=blocked,
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "checks": checks,
        "config_templates": templates,
        "written_templates": written_templates,
        "install_profile": build_install_profile(product_root, product_db_path, backup_dir),
        "next_actions": next_actions(checks),
        "safety": safety_contract() | {"product_config_template_writes": bool(write_templates)},
    }
    if out_path:
        write_json(out_path, report)
    return report


def check_product_root(product_root: Path) -> Mapping[str, Any]:
    if "stable_runtime" in product_root.parts:
        return check("product_root", CHECK_BLOCK, "product_root_under_stable_runtime", product_root)
    if not product_root.exists():
        return check("product_root", CHECK_WARN, "product_root_missing_will_be_created_by_setup", product_root)
    if not product_root.is_dir():
        return check("product_root", CHECK_BLOCK, "product_root_is_not_directory", product_root)
    return check("product_root", CHECK_OK, "product_root_ready", product_root)


def check_product_db(product_db_path: Path, product_root: Path) -> Mapping[str, Any]:
    try:
        guard_product_db_path(product_db_path, product_root, must_exist=True)
        audit = audit_product_db(product_db_path, product_root)
    except Exception as exc:
        return check("product_db", CHECK_BLOCK, str(exc), product_db_path)
    status = CHECK_OK if audit["summary"]["validation_ok"] else CHECK_BLOCK
    return check("product_db", status, "product_db_valid" if status == CHECK_OK else "product_db_invalid", product_db_path) | {
        "audit_summary": audit["summary"]
    }


def check_mango_credentials(*, require: bool) -> Mapping[str, Any]:
    present = bool(clean(os.getenv("MANGO_OFFICE_API_KEY")) and clean(os.getenv("MANGO_OFFICE_API_SALT")))
    if present:
        return check("mango_credentials", CHECK_OK, "env_credentials_present", "env:MANGO_OFFICE_API_KEY")
    return check(
        "mango_credentials",
        CHECK_BLOCK if require else CHECK_WARN,
        "env_credentials_missing",
        "env:MANGO_OFFICE_API_KEY",
    )


def check_crm_snapshot(product_db_path: Path, product_root: Path, crm_snapshot_path: Optional[Path]) -> Mapping[str, Any]:
    if crm_snapshot_path is None or not crm_snapshot_path.exists():
        return check("crm_snapshot", CHECK_WARN, "crm_snapshot_missing", crm_snapshot_path or product_root / "crm_snapshots")
    try:
        resolution = build_crm_entity_resolution_report(product_db_path, product_root, crm_snapshot_path, limit=50)
    except Exception as exc:
        return check("crm_snapshot", CHECK_BLOCK, str(exc), crm_snapshot_path)
    return check("crm_snapshot", CHECK_OK, "crm_snapshot_readable", crm_snapshot_path) | {
        "resolution_summary": resolution["summary"]
    }


def check_backup_dir(backup_dir: Path) -> Mapping[str, Any]:
    if "stable_runtime" in backup_dir.parts:
        return check("backup_dir", CHECK_BLOCK, "backup_dir_under_stable_runtime", backup_dir)
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        probe = backup_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return check("backup_dir", CHECK_BLOCK, f"backup_dir_not_writable:{exc}", backup_dir)
    return check("backup_dir", CHECK_OK, "backup_dir_writable", backup_dir)


def check_retention(product_db_path: Path, product_root: Path) -> Mapping[str, Any]:
    try:
        audit = audit_product_retention(product_db_path, product_root)
    except Exception as exc:
        return check("retention", CHECK_WARN, str(exc), product_db_path)
    status = CHECK_OK if audit["summary"]["validation_ok"] else CHECK_WARN
    return check("retention", status, "retention_policy_readable", product_db_path) | {
        "retention_summary": audit["summary"]
    }


def check_runtime_separation(product_root: Path, product_db_path: Path) -> Mapping[str, Any]:
    if "stable_runtime" in product_root.parts or "stable_runtime" in product_db_path.parts:
        return check("runtime_separation", CHECK_BLOCK, "product_paths_overlap_stable_runtime", product_root)
    return check("runtime_separation", CHECK_OK, "product_paths_are_separate_from_stable_runtime", product_root)


def check_demo_readiness(product_root: Path, product_db_path: Path) -> Mapping[str, Any]:
    try:
        readiness = build_dashboard_demo_readiness(product_root=product_root, product_db_path=product_db_path)
    except Exception as exc:
        return check("demo_readiness", CHECK_WARN, str(exc), product_root)
    status = CHECK_OK if readiness["summary"]["validation_ok"] else CHECK_WARN
    return check("demo_readiness", status, "demo_readiness_contract_available", product_root) | {
        "demo_summary": readiness["summary"],
        "warning_reasons": readiness.get("warning_reasons", []),
    }


def check(name: str, status: str, reason: str, path: object) -> Mapping[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "path": str(path) if path is not None else None,
    }


def next_actions(checks: list[Mapping[str, Any]]) -> list[str]:
    actions = []
    for item in checks:
        if item["status"] == CHECK_OK:
            continue
        if item["name"] == "product_db":
            actions.append("Initialize or repair the product DB under product root.")
        elif item["name"] == "crm_snapshot":
            actions.append("Run the read-only AMO snapshot exporter or provide a local CRM snapshot.")
        elif item["name"] == "mango_credentials":
            actions.append("Set MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT before live shadow polling.")
        elif item["name"] == "backup_dir":
            actions.append("Create a writable backups directory under product root.")
        elif item["name"] == "demo_readiness":
            actions.append("Build a sanitized real demo root or add local CRM/Tallanto snapshots before client demos.")
        else:
            actions.append(f"Review check: {item['name']}")
    return actions


def default_crm_snapshot(product_root: Path) -> Optional[Path]:
    for relative in (
        "crm_snapshots/amocrm_entities.json",
        "crm_snapshots/amocrm_entities.jsonl",
        "crm_snapshots/amocrm_entities.csv",
    ):
        path = product_root / relative
        if path.exists():
            return path
    return None


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def build_config_templates(
    product_root: Path,
    product_db_path: Path,
    crm_snapshot_path: Optional[Path],
    backup_dir: Path,
) -> Mapping[str, Any]:
    return {
        "env_example": "\n".join(
            [
                "# Mango Analyse appliance local profile",
                f"MANGO_PRODUCT_ROOT={product_root}",
                f"MANGO_PRODUCT_DB={product_db_path}",
                "MANGO_OFFICE_API_KEY=<put-client-vpbx-code-here>",
                "MANGO_OFFICE_API_SALT=<put-client-signature-key-here>",
                "CRM_AMO_BASE_URL=<https://example.amocrm.ru>",
                "CRM_AMO_API_TOKEN=<put-read-only-token-here>",
                f"MANGO_CRM_SNAPSHOT={crm_snapshot_path or product_root / 'crm_snapshots' / 'amocrm_entities.json'}",
                f"MANGO_BACKUP_DIR={backup_dir}",
                "MANGO_ENABLE_LIVE_CRM_WRITE=0",
                "MANGO_ENABLE_ASR_EXECUTION=0",
                "",
            ]
        ),
        "paths_json": {
            "product_root": str(product_root),
            "product_db": str(product_db_path),
            "crm_snapshot": str(crm_snapshot_path) if crm_snapshot_path else str(product_root / "crm_snapshots" / "amocrm_entities.json"),
            "backup_dir": str(backup_dir),
            "dashboard_url": "http://127.0.0.1:8765/dashboard",
        },
    }


def write_config_templates(product_root: Path, templates: Mapping[str, Any]) -> list[Mapping[str, str]]:
    env_path = product_root / "config" / "appliance.env.example"
    paths_path = product_root / "config" / "appliance_paths.json"
    for label, path in (("env template", env_path), ("paths template", paths_path)):
        guard_under_root(path, product_root, label)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(str(templates["env_example"]), encoding="utf-8")
    paths_path.write_text(json.dumps(templates["paths_json"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [
        {"kind": "env_example", "path": str(env_path)},
        {"kind": "paths_json", "path": str(paths_path)},
    ]


def build_install_profile(product_root: Path, product_db_path: Path, backup_dir: Path) -> Mapping[str, Any]:
    return {
        "mode": "client_hosted_sqlite_appliance",
        "product_root": str(product_root),
        "product_db_path": str(product_db_path),
        "backup_dir": str(backup_dir),
        "commands": {
            "serve_dashboard": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py "
                f"--product-root {product_root} --product-db {product_db_path} serve --host 127.0.0.1 --port 8765"
            ),
            "healthcheck": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py "
                f"--product-root {product_root} --product-db {product_db_path} healthcheck"
            ),
        },
        "disabled_by_default": ["ASR execution", "R+A execution", "CRM writeback", "runtime DB writes"],
    }


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "live_crm_reads": False,
        "write_crm": False,
        "write_tallanto": False,
        "run_asr": False,
        "run_ra": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
