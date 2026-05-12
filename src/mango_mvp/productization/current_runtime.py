from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.call_processing_readiness import build_call_processing_readiness_report


CURRENT_RUNTIME_SCHEMA_VERSION = "current_runtime_contract_v1"
DEFAULT_CURRENT_RUNTIME_PATH = Path("stable_runtime/CURRENT_RUNTIME.json")


@dataclass(frozen=True)
class CurrentRuntimeSummary:
    schema_version: str
    generated_at: str
    project_root: str
    active_export_name: Optional[str]
    validation_ok: bool
    blocked: int
    warnings: int
    canonical_actionable_calls: int
    canonical_missing_asr: int
    canonical_missing_ra: int
    amo_ready_rows: int
    safe_writeback_pending_rows: int
    stage1_writeback_complete: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_current_runtime_contract(
    *,
    project_root: Path,
    out_path: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build the machine-readable contract for the currently active runtime layer.

    The contract is intentionally read-only. It pins all paths that operator UI,
    writeback gates and future SaaS service code should treat as the current
    source of truth, so new code does not infer state from old folders.
    """

    project_root = project_root.resolve(strict=False)
    out_path = _resolve_optional(project_root, out_path or DEFAULT_CURRENT_RUNTIME_PATH)
    now = generated_at or datetime.now(timezone.utc)

    pointer_path = project_root / "stable_runtime" / "CANONICAL_EXPORT.txt"
    active_export_root = _resolve_active_export_root(project_root, pointer_path)
    export_summary_path = active_export_root / "summary.json" if active_export_root else None
    export_summary = _load_json_if_exists(export_summary_path)

    canonical_db = _path_from_value(export_summary.get("canonical_db"))
    canonical_summary_path = canonical_db.parent / "summary.json" if canonical_db else None
    stage15_summary_path = _path_from_value(export_summary.get("stage15_summary"))
    amo_export_csv = _path_from_value(_mapping(export_summary.get("output_files")).get("amo_export_ready_csv"))
    crm_quality_summary_path = _find_matching_summary(
        project_root / "stable_runtime",
        "crm_writeback_quality_gate_*/summary.json",
        key="input",
        expected_path=amo_export_csv,
    )
    amo_queue_summary_path = _find_matching_summary(
        project_root / "stable_runtime",
        "amo_writeback_queue_*/summary.json",
        key="input_csv",
        expected_path=amo_export_csv,
    )
    product_root = project_root / "_local_archive_mango_api_downloads_20260507" / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    default_quarantine_manifest = project_root / "_cleanup_quarantine_20260510_stage2" / "MANIFEST.csv"
    quarantine_manifest = default_quarantine_manifest if default_quarantine_manifest.exists() else None

    readiness = build_call_processing_readiness_report(
        project_root=project_root,
        canonical_export_pointer=pointer_path,
        export_summary_path=export_summary_path,
        canonical_summary_path=canonical_summary_path,
        stage15_summary_path=stage15_summary_path,
        crm_quality_summary_path=crm_quality_summary_path,
        amo_queue_summary_path=amo_queue_summary_path,
        quarantine_manifest_path=quarantine_manifest,
    )
    readiness_summary = _mapping(readiness.get("summary"))
    readiness_gates = list(readiness.get("gates") or [])
    local_gates = _contract_gates(
        pointer_path=pointer_path,
        active_export_root=active_export_root,
        export_summary_path=export_summary_path,
        canonical_db=canonical_db,
        stage15_summary_path=stage15_summary_path,
        amo_export_csv=amo_export_csv,
        crm_quality_summary_path=crm_quality_summary_path,
        amo_queue_summary_path=amo_queue_summary_path,
        product_root=product_root,
        product_db=product_db,
        quarantine_manifest=quarantine_manifest,
        readiness_summary=readiness_summary,
    )
    blocked = sum(1 for gate in local_gates if gate["severity"] == "block" and not gate["passed"])
    warnings = sum(1 for gate in local_gates if gate["severity"] == "warn" and not gate["passed"])
    validation_ok = blocked == 0 and bool(readiness_summary.get("validation_ok"))

    summary = CurrentRuntimeSummary(
        schema_version=CURRENT_RUNTIME_SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        project_root=str(project_root),
        active_export_name=active_export_root.name if active_export_root else None,
        validation_ok=validation_ok,
        blocked=blocked,
        warnings=warnings,
        canonical_actionable_calls=_int(readiness_summary.get("canonical_actionable_calls")),
        canonical_missing_asr=_int(readiness_summary.get("canonical_missing_asr")),
        canonical_missing_ra=_int(readiness_summary.get("canonical_missing_ra")),
        amo_ready_rows=_int(readiness_summary.get("amo_ready_rows")),
        safe_writeback_pending_rows=_int(readiness_summary.get("safe_writeback_pending_rows")),
        stage1_writeback_complete=bool(readiness_summary.get("stage1_writeback_complete")),
    )
    contract: dict[str, Any] = {
        "summary": summary.to_json_dict(),
        "paths": {
            "canonical_export_pointer": _string_path(pointer_path),
            "active_export_root": _string_path(active_export_root),
            "active_export_summary": _string_path(export_summary_path),
            "canonical_db": _string_path(canonical_db),
            "canonical_summary": _string_path(canonical_summary_path),
            "stage15_summary": _string_path(stage15_summary_path),
            "amo_export_ready_csv": _string_path(amo_export_csv),
            "crm_quality_summary": _string_path(crm_quality_summary_path),
            "amo_queue_summary": _string_path(amo_queue_summary_path),
            "product_root": _string_path(product_root),
            "product_db": _string_path(product_db),
            "quarantine_manifest": _string_path(quarantine_manifest),
        },
        "gates": local_gates,
        "readiness": {
            "summary": readiness_summary,
            "gates": readiness_gates,
            "next_actions": list(readiness.get("next_actions") or []),
        },
        "contracts": {
            "runtime_source_of_truth": "stable_runtime/CURRENT_RUNTIME.json",
            "active_export_pointer": "stable_runtime/CANONICAL_EXPORT.txt",
            "old_april_exports_allowed": False,
            "live_crm_write_without_explicit_operator_stage": False,
            "live_mango_download_without_operator_stage": False,
        },
        "safety": {
            "read_only": True,
            "downloads_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def load_current_runtime_contract(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contract_gates(**kwargs: Any) -> list[Mapping[str, Any]]:
    pointer_path: Path = kwargs["pointer_path"]
    active_export_root: Optional[Path] = kwargs["active_export_root"]
    export_summary_path: Optional[Path] = kwargs["export_summary_path"]
    canonical_db: Optional[Path] = kwargs["canonical_db"]
    stage15_summary_path: Optional[Path] = kwargs["stage15_summary_path"]
    amo_export_csv: Optional[Path] = kwargs["amo_export_csv"]
    crm_quality_summary_path: Optional[Path] = kwargs["crm_quality_summary_path"]
    amo_queue_summary_path: Optional[Path] = kwargs["amo_queue_summary_path"]
    product_root: Path = kwargs["product_root"]
    product_db: Path = kwargs["product_db"]
    quarantine_manifest: Optional[Path] = kwargs["quarantine_manifest"]
    readiness_summary: Mapping[str, Any] = kwargs["readiness_summary"]
    active_name = active_export_root.name if active_export_root else ""
    return [
        _gate("CURRENT_POINTER_EXISTS", pointer_path.exists(), "Active export pointer exists.", "block"),
        _gate("ACTIVE_EXPORT_EXISTS", bool(active_export_root and active_export_root.exists()), "Active export root exists.", "block"),
        _gate("ACTIVE_EXPORT_NOT_LEGACY_APRIL", "20260424" not in active_name, "Active export is not the old April layer.", "block"),
        _gate("ACTIVE_EXPORT_SUMMARY_EXISTS", bool(export_summary_path and export_summary_path.exists()), "Active export summary exists.", "block"),
        _gate("CANONICAL_DB_EXISTS", bool(canonical_db and canonical_db.exists()), "Canonical DB exists.", "block"),
        _gate("STAGE15_SUMMARY_EXISTS", bool(stage15_summary_path and stage15_summary_path.exists()), "Stage15 quality summary exists.", "block"),
        _gate("AMO_READY_CSV_EXISTS", bool(amo_export_csv and amo_export_csv.exists()), "AMO-ready CSV exists.", "block"),
        _gate("CRM_QUALITY_SUMMARY_EXISTS", bool(crm_quality_summary_path and crm_quality_summary_path.exists()), "CRM writeback quality summary exists.", "block"),
        _gate("AMO_QUEUE_SUMMARY_EXISTS", bool(amo_queue_summary_path and amo_queue_summary_path.exists()), "AMO queue summary exists.", "block"),
        _gate("CALL_PROCESSING_READINESS_GREEN", bool(readiness_summary.get("validation_ok")), "Call-processing readiness gate is green.", "block"),
        _gate("PRODUCT_ROOT_EXISTS", product_root.exists(), "Product appliance root exists.", "warn"),
        _gate("PRODUCT_DB_EXISTS", product_db.exists(), "Product appliance SQLite DB exists.", "warn"),
        _gate(
            "CLEANUP_QUARANTINE_CLOSED_OR_RESTORE_MANIFEST",
            quarantine_manifest is None or quarantine_manifest.exists(),
            (
                "Cleanup quarantine is closed; restore manifest is no longer expected."
                if quarantine_manifest is None
                else "Cleanup quarantine restore manifest exists."
            ),
            "warn",
        ),
    ]


def _gate(gate_id: str, passed: bool, reason: str, severity: str) -> Mapping[str, Any]:
    return {"gate": gate_id, "passed": bool(passed), "severity": severity, "reason": reason}


def _resolve_active_export_root(project_root: Path, pointer_path: Path) -> Optional[Path]:
    if not pointer_path.exists():
        return None
    text = pointer_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = project_root / "stable_runtime" / path
    return path.resolve(strict=False)


def _find_matching_summary(root: Path, pattern: str, *, key: str, expected_path: Optional[Path]) -> Optional[Path]:
    candidates = sorted(root.glob(pattern))
    if expected_path is None:
        return candidates[-1] if candidates else None
    matches: list[Path] = []
    for path in candidates:
        data = _load_json_if_exists(path)
        value = _path_from_value(data.get(key))
        if value and value.resolve(strict=False) == expected_path.resolve(strict=False):
            matches.append(path)
    return matches[-1] if matches else None


def _load_json_if_exists(path: Optional[Path]) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_value(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def _resolve_optional(project_root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _string_path(path: Optional[Path]) -> Optional[str]:
    return str(path) if path else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CURRENT_RUNTIME_SCHEMA_VERSION",
    "DEFAULT_CURRENT_RUNTIME_PATH",
    "build_current_runtime_contract",
    "load_current_runtime_contract",
]
