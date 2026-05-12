from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


APPLIANCE_COMMAND_SURFACE_SCHEMA_VERSION = "appliance_command_surface_v1"


@dataclass(frozen=True)
class ApplianceCommandSurfaceSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    commands: int
    missing_scripts: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_appliance_command_surface(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    workspace_root = (workspace_root or Path.cwd()).resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=False)
    guard_path(product_root, product_root, "product root", allow_root=True)
    if out_path:
        guard_path(out_path, product_root, "appliance command surface output")
    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535")
    host = clean(host) or "127.0.0.1"

    commands = command_groups(product_root=product_root, product_db_path=product_db_path, host=host, port=port)
    missing = missing_scripts(commands, workspace_root=workspace_root)
    blocked = len(missing)
    warnings = 0 if product_db_path.exists() else 1
    report = {
        "summary": ApplianceCommandSurfaceSummary(
            schema_version=APPLIANCE_COMMAND_SURFACE_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            commands=sum(len(group["commands"]) for group in commands),
            missing_scripts=len(missing),
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict(),
        "groups": commands,
        "missing_scripts": missing,
        "operator_flow": [
            "demo_or_bootstrap",
            "config_wizard",
            "healthcheck",
            "scheduler_health",
            "serve_dashboard",
            "backup_verify",
            "service_pack",
            "tenant_isolation",
            "demo_pilot_playbook",
            "processing_acceptance_gates",
        ],
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def command_groups(*, product_root: Path, product_db_path: Path, host: str, port: int) -> list[Mapping[str, Any]]:
    return [
        {
            "id": "bootstrap",
            "label": "Create or refresh demo/client product root",
            "commands": [
                command(
                    "build_demo_tenant",
                    "scripts/mango_office_demo_tenant.py",
                    f"--product-root {shell_path(product_root)} --replace",
                    "Creates anonymized demo product data only.",
                ),
                command(
                    "build_sanitized_real_demo",
                    "scripts/mango_office_sanitized_real_demo.py",
                    (
                        f"--source-product-root {shell_path(product_root)} "
                        f"--source-product-db {shell_path(product_db_path)} "
                        f"--demo-product-root {shell_path(product_root.parent / 'sanitized_real_demo_appliance')} --replace"
                    ),
                    "Creates a masked demo root from real product DB rows.",
                ),
                command(
                    "product_api_readiness",
                    "scripts/mango_office_product_api_readiness.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Checks read-only Product API contracts.",
                ),
            ],
        },
        {
            "id": "configure",
            "label": "Validate appliance configuration",
            "commands": [
                command(
                    "appliance_config_wizard",
                    "scripts/mango_office_appliance_config_wizard.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Checks paths, Mango env, CRM snapshot, product DB, retention, backup dir.",
                ),
                command(
                    "amo_snapshot_export",
                    "scripts/mango_office_amo_snapshot_export.py",
                    f"--product-root {shell_path(product_root)}",
                    "Reads amoCRM and writes only local crm_snapshots/amocrm_entities.json.",
                ),
            ],
        },
        {
            "id": "operate",
            "label": "Run safe diagnostics and local UI",
            "commands": [
                command(
                    "ops_healthcheck",
                    "scripts/mango_office_product_ops.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} healthcheck",
                    "Verifies product DB and backup directory.",
                ),
                command(
                    "ops_diagnostics",
                    "scripts/mango_office_product_ops.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} diagnostics",
                    "Builds a local diagnostics bundle without running restore or workers.",
                ),
                command(
                    "scheduler_health",
                    "scripts/mango_office_scheduler_health.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Reports due, failed, locked and stale scheduler jobs.",
                ),
                command(
                    "serve_dashboard",
                    "scripts/mango_office_product_api_http.py",
                    (
                        f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
                        f"serve --host {host} --port {port}"
                    ),
                    "Starts read-only dashboard/API service.",
                ),
            ],
        },
        {
            "id": "package",
            "label": "Package client-hosted appliance",
            "commands": [
                command(
                    "service_pack",
                    "scripts/mango_office_appliance_service_pack.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Generates launchd/systemd templates without installing or starting them.",
                ),
                command(
                    "tenant_isolation",
                    "scripts/mango_office_tenant_isolation.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Reports tenant isolation and optional tenant layout scaffold.",
                ),
            ],
        },
        {
            "id": "pilot_gate",
            "label": "Prepare demo/pilot acceptance",
            "commands": [
                command(
                    "demo_pilot_playbook",
                    "scripts/mango_office_demo_pilot_playbook.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Builds a client demo and pilot playbook from product-safe data.",
                ),
                command(
                    "processing_acceptance_gates",
                    "scripts/mango_office_processing_acceptance_gates.py",
                    f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)}",
                    "Checks gates before connecting processing quality output.",
                ),
            ],
        },
        {
            "id": "backup",
            "label": "Back up and verify product DB",
            "commands": [
                command(
                    "backup_product_db",
                    "scripts/mango_office_product_ops.py",
                    (
                        f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
                        f"backup --backup {shell_path(product_root / 'backups' / 'mango_product_appliance_backup.sqlite')}"
                    ),
                    "Creates a local product DB backup under product root.",
                ),
                command(
                    "restore_dry_run",
                    "scripts/mango_office_product_ops.py",
                    (
                        f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
                        f"restore-dry-run --backup {shell_path(product_root / 'backups' / 'mango_product_appliance_backup.sqlite')}"
                    ),
                    "Verifies restore plan without replacing the live product DB.",
                ),
            ],
        },
    ]


def command(command_id: str, script: str, args: str, purpose: str) -> Mapping[str, Any]:
    return {
        "id": command_id,
        "script": script,
        "command": f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 {script} {args}".strip(),
        "purpose": purpose,
        "read_only_or_guarded": True,
    }


def missing_scripts(groups: list[Mapping[str, Any]], workspace_root: Path) -> list[str]:
    missing = []
    for group in groups:
        for item in group["commands"]:
            script_path = workspace_root / clean(item.get("script"))
            if not script_path.exists():
                missing.append(clean(item.get("script")))
    return sorted(set(missing))


def shell_path(path: Path) -> str:
    text = str(path)
    if not text or all(ch not in text for ch in (" ", "'", '"', "(", ")")):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def guard_path(path: Path, product_root: Path, label: str, *, allow_root: bool = False) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if allow_root and path == product_root:
        return
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
        "executes_commands": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
