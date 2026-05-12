from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.product_api import ProductApiFacade
from mango_mvp.productization.product_db import audit_product_db, guard_product_db_path
from mango_mvp.productization.product_ops import build_product_ops_healthcheck
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from mango_mvp.productization.tenant_isolation import build_tenant_isolation_report
from mango_mvp.productization.test_ingest import path_is_relative_to


PROCESSING_ACCEPTANCE_GATES_SCHEMA_VERSION = "processing_acceptance_gates_v1"


@dataclass(frozen=True)
class ProcessingAcceptanceGatesSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    gates: int
    passed: int
    blocked: int
    warnings: int
    processing_quality_external_ready: bool
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_processing_acceptance_gates_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    *,
    processing_quality_ready: bool = False,
    processing_quality_report_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    processing_quality_report_path = processing_quality_report_path.resolve(strict=False) if processing_quality_report_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        guard_under_root(out_path, product_root, "processing acceptance gates output")
    if processing_quality_report_path and "stable_runtime" in processing_quality_report_path.parts:
        raise ValueError("processing quality evidence must not point into stable_runtime from this productization gate")

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path)
    dashboard = api.appliance_dashboard(capture_limit=10, scheduler_limit=10)
    product_db = audit_product_db(product_db_path, product_root)
    demo = build_dashboard_demo_readiness(product_root=product_root, product_db_path=product_db_path, panels=dashboard.get("panels", {}))
    scheduler = build_scheduler_health_report(product_db_path=product_db_path, product_root=product_root)
    ops = build_product_ops_healthcheck(product_root=product_root, product_db_path=product_db_path)
    tenants = build_tenant_isolation_report(product_root=product_root, product_db_path=product_db_path)
    evidence_ready = processing_quality_ready or bool(processing_quality_report_path and processing_quality_report_path.exists())
    gates = [
        gate("PRODUCT_DB_VALID", product_db["summary"]["validation_ok"], "Product DB schema and migrations are valid.", severity="block"),
        gate("DASHBOARD_CONTRACT_READY", demo["summary"]["validation_ok"], "Dashboard has required read-only panels.", severity="block"),
        gate("OPS_HEALTHCHECK_OK", ops["summary"]["validation_ok"], "Local product DB and backup directory are operational.", severity="block"),
        gate("TENANT_ISOLATION_OK", tenants["summary"]["validation_ok"], "Tenant-scoped product rows have tenant_id.", severity="block"),
        gate("SCHEDULER_NO_FAILED_OR_STALE", scheduler["summary"]["blocked"] == 0, "Scheduler has no failed or stale-running jobs.", severity="block"),
        gate("CRM_SNAPSHOT_AVAILABLE", demo["summary"]["snapshot_files"] > 0, "CRM/Tallanto snapshots are available for previews.", severity="warn"),
        gate("CAPTURE_DATA_AVAILABLE", dashboard["summary"]["capture_inbox_items"] > 0 or dashboard["summary"]["product_calls"] > 0, "Product DB has call/capture rows.", severity="warn"),
        gate("PROCESSING_QUALITY_EXTERNAL_READY", evidence_ready, "Processing dialog has accepted transcript/analysis quality.", severity="block"),
        gate("LIVE_ACTIONS_DISABLED", True, "ASR, R+A, runtime DB writes and CRM writes remain disabled by default.", severity="block"),
    ]
    blocked = sum(1 for item in gates if item["severity"] == "block" and not item["passed"])
    warnings = sum(1 for item in gates if item["severity"] == "warn" and not item["passed"])
    report = {
        "summary": ProcessingAcceptanceGatesSummary(
            schema_version=PROCESSING_ACCEPTANCE_GATES_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            gates=len(gates),
            passed=sum(1 for item in gates if item["passed"]),
            blocked=blocked,
            warnings=warnings,
            processing_quality_external_ready=evidence_ready,
            validation_ok=blocked == 0,
        ).to_json_dict(),
        "gates": gates,
        "inputs": {
            "processing_quality_ready": processing_quality_ready,
            "processing_quality_report_path": str(processing_quality_report_path) if processing_quality_report_path else None,
        },
        "source_summaries": {
            "product_db": product_db["summary"],
            "dashboard": dashboard["summary"],
            "demo": demo["summary"],
            "scheduler": scheduler["summary"],
            "ops": ops["summary"],
            "tenants": tenants["summary"],
        },
        "next_actions": next_actions(gates),
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def gate(gate_id: str, passed: bool, reason: str, *, severity: str) -> Mapping[str, Any]:
    return {
        "gate": gate_id,
        "passed": bool(passed),
        "severity": severity,
        "reason": reason,
    }


def next_actions(gates: list[Mapping[str, Any]]) -> list[str]:
    actions = []
    for item in gates:
        if item["passed"]:
            continue
        gate_id = item["gate"]
        if gate_id == "PROCESSING_QUALITY_EXTERNAL_READY":
            actions.append("Wait for processing dialog to close transcript/analysis quality acceptance, then pass evidence explicitly.")
        elif gate_id == "CRM_SNAPSHOT_AVAILABLE":
            actions.append("Run read-only AMO/Tallanto snapshot export or provide local snapshots.")
        elif gate_id == "CAPTURE_DATA_AVAILABLE":
            actions.append("Run safe Mango shadow poll/capture ingest into product DB before demo or pilot.")
        elif gate_id == "SCHEDULER_NO_FAILED_OR_STALE":
            actions.append("Review scheduler health before enabling appliance loop.")
        else:
            actions.append(f"Resolve gate: {gate_id}")
    return actions


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
