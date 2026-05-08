from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.product_api import ProductApiFacade, read_only_actions
from mango_mvp.productization.test_ingest import path_is_relative_to


APPLIANCE_LOOP_SCHEMA_VERSION = "autonomous_appliance_loop_dry_run_v1"


@dataclass(frozen=True)
class ApplianceLoopSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    capture_ready: int
    scheduler_runs: int
    planned_actions: int
    blocked_actions: int
    loop_ready: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_autonomous_appliance_loop_dry_run(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_appliance_loop_paths(product_root=product_root, product_db_path=product_db_path, out_path=out_path)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)
    dashboard = api.dashboard_summary()
    capture = api.capture_recent(limit=50)
    processing = api.processing_queue(limit=50)
    scheduler = api.scheduler_runs(limit=50)
    asr_gate = api.asr_gate_status()
    writeback = api.writeback_previews()
    db_summary = dashboard["summary"]
    capture_ready = int(db_summary.get("capture_inbox_ready") or 0)
    scheduler_runs = int(db_summary.get("job_runs") or 0)
    actions = build_loop_actions(
        product_db_present=bool(dashboard["health"].get("product_db_present")),
        dashboard_ok=bool(dashboard["health"].get("validation_ok")),
        capture_ready=capture_ready,
        scheduler_runs=scheduler_runs,
        asr_gate=asr_gate,
        writeback=writeback,
    )
    action_counts = Counter(action["action"] for action in actions)
    blocked_actions = sum(1 for action in actions if action["status"] == "blocked")
    dry_run_inputs_ready = (
        bool(dashboard["health"].get("product_db_present"))
        and bool(dashboard["health"].get("validation_ok"))
        and scheduler_runs > 0
        and capture_ready >= 0
    )
    loop_ready = dry_run_inputs_ready and blocked_actions == 0
    report = {
        "summary": ApplianceLoopSummary(
            schema_version=APPLIANCE_LOOP_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            capture_ready=capture_ready,
            scheduler_runs=scheduler_runs,
            planned_actions=len(actions),
            blocked_actions=blocked_actions,
            loop_ready=loop_ready,
            validation_ok=bool(dashboard["health"].get("validation_ok")),
            warnings=int(db_summary.get("warnings") or 0) + blocked_actions,
        ).to_json_dict()
        | {
            "dry_run_inputs_ready": dry_run_inputs_ready,
            "loop_blocked_reason": "dangerous_actions_blocked_by_policy" if dry_run_inputs_ready and blocked_actions else None,
        },
        "action_counts": dict(sorted(action_counts.items())),
        "actions": actions,
        "inputs": {
            "dashboard_summary": dashboard,
            "capture_recent_count": len(capture["items"]),
            "processing_queue_count": len(processing["items"]),
            "scheduler_runs_count": len(scheduler["items"]),
            "asr_gate_status": asr_gate,
            "writeback_previews": writeback,
        },
        "loop_contract": {
            "poll_mango": "read_only_shadow_poll",
            "capture_inbox": "single_writer_backend_required_for_writes",
            "recording_download": "disabled_until_policy_gate",
            "processing_queue": "dry_run_only",
            "asr": "requires_explicit_approval",
            "crm": "preview_only",
        },
        "safety": appliance_loop_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_loop_actions(
    product_db_present: bool,
    dashboard_ok: bool,
    capture_ready: int,
    scheduler_runs: int,
    asr_gate: Mapping[str, Any],
    writeback: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    actions.append(
        action(
            "CHECK_PRODUCT_DB",
            "ready" if product_db_present and dashboard_ok else "blocked",
            "product DB is available for appliance loop" if product_db_present and dashboard_ok else "product DB missing or invalid",
        )
    )
    actions.append(
        action(
            "CHECK_SCHEDULER_RUNTIME",
            "ready" if scheduler_runs > 0 else "blocked",
            f"{scheduler_runs} scheduler job rows observed",
        )
    )
    actions.append(
        action(
            "PLAN_MANGO_SHADOW_POLL",
            "ready" if product_db_present else "blocked",
            "polling remains read-only and stores no secrets in job payload",
        )
    )
    actions.append(
        action(
            "PLAN_CAPTURE_INBOX_REVIEW",
            "ready" if capture_ready >= 0 else "blocked",
            f"{capture_ready} capture inbox rows ready for downstream review",
        )
    )
    actions.append(
        action(
            "BLOCK_RECORDING_DOWNLOAD_AUTO_TRIGGER",
            "blocked",
            "recording download requires an explicit future policy gate",
        )
    )
    actions.append(
        action(
            "BLOCK_ASR_AUTO_TRIGGER",
            "blocked",
            "ASR execution remains disabled until explicit human approval and launcher gate",
            details=mapping_or_empty(asr_gate.get("stage25")),
        )
    )
    actions.append(
        action(
            "BLOCK_CRM_WRITEBACK",
            "blocked",
            "CRM writeback remains preview-only",
            details={"blocked_reasons": writeback.get("blocked_reasons") or []},
        )
    )
    return actions


def action(name: str, status: str, reason: str, details: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    return {
        "action": name,
        "status": status,
        "reason": reason,
        "download_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "write_crm": False,
        "details": details or {},
    }


def appliance_loop_safety() -> Mapping[str, bool]:
    safety = dict(read_only_actions())
    safety.update({"writes_report_json": True, "product_db_writes": False, "download_audio": False})
    return safety


def guard_appliance_loop_paths(product_root: Path, product_db_path: Path, out_path: Optional[Path]) -> None:
    for label, path in (("product root", product_root), ("product DB", product_db_path), ("appliance loop audit", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"appliance loop audit must stay under product root: {product_root}")


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
