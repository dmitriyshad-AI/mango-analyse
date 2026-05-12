from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


CALL_PROCESSING_READINESS_SCHEMA_VERSION = "call_processing_readiness_v1"


@dataclass(frozen=True)
class CallProcessingReadinessSummary:
    schema_version: str
    project_root: str
    active_export_root: Optional[str]
    gates: int
    passed: int
    blocked: int
    warnings: int
    canonical_actionable_calls: int
    canonical_missing_asr: int
    canonical_missing_ra: int
    amo_ready_rows: int
    safe_writeback_pending_rows: int
    stage1_writeback_complete: bool
    processing_pipeline_ready: bool
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_call_processing_readiness_report(
    *,
    project_root: Path,
    out_path: Optional[Path] = None,
    canonical_export_pointer: Optional[Path] = None,
    canonical_summary_path: Optional[Path] = None,
    stage15_summary_path: Optional[Path] = None,
    export_summary_path: Optional[Path] = None,
    crm_quality_summary_path: Optional[Path] = None,
    amo_queue_summary_path: Optional[Path] = None,
    quarantine_manifest_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Build a read-only industrial readiness report for the call-processing pipeline.

    This gate intentionally performs no live actions. It binds together the current
    post-backfill canonical layer, transcript quality gate, CRM text gate, AMO
    writeback queue and readback evidence so future stages cannot accidentally
    use an old export or skip a quality/readback check.
    """

    project_root = project_root.resolve(strict=False)
    out_path = _resolve_optional(project_root, out_path)
    pointer_path = _resolve_optional(
        project_root,
        canonical_export_pointer or Path("stable_runtime/CANONICAL_EXPORT.txt"),
    )
    active_export_root = _active_export_root(project_root, pointer_path)

    export_summary_path = _resolve_optional(
        project_root,
        export_summary_path or ((active_export_root / "summary.json") if active_export_root else None),
    )
    export_summary = _load_json(export_summary_path) if export_summary_path else {}

    canonical_summary_path = _resolve_optional(
        project_root,
        canonical_summary_path or _summary_from_db_path(export_summary.get("canonical_db")),
    )
    stage15_summary_path = _resolve_optional(
        project_root,
        stage15_summary_path or _path_from_value(export_summary.get("stage15_summary")),
    )
    crm_quality_summary_path = _resolve_optional(
        project_root,
        crm_quality_summary_path or _find_latest_crm_quality_summary(project_root, export_summary),
    )
    amo_queue_summary_path = _resolve_optional(
        project_root,
        amo_queue_summary_path or _find_latest_amo_queue_summary(project_root, export_summary),
    )
    quarantine_manifest_path = _resolve_optional(
        project_root,
        quarantine_manifest_path or Path("_cleanup_quarantine_20260510_stage2/MANIFEST.csv"),
    )

    canonical_summary = _load_json(canonical_summary_path) if canonical_summary_path else {}
    stage15_summary = _load_json(stage15_summary_path) if stage15_summary_path else {}
    crm_quality_summary = _load_json(crm_quality_summary_path) if crm_quality_summary_path else {}
    amo_queue_summary = _load_json(amo_queue_summary_path) if amo_queue_summary_path else {}
    readback = _collect_readback_evidence(project_root)

    gates = _build_gates(
        project_root=project_root,
        pointer_path=pointer_path,
        active_export_root=active_export_root,
        export_summary_path=export_summary_path,
        export_summary=export_summary,
        canonical_summary_path=canonical_summary_path,
        canonical_summary=canonical_summary,
        stage15_summary_path=stage15_summary_path,
        stage15_summary=stage15_summary,
        crm_quality_summary_path=crm_quality_summary_path,
        crm_quality_summary=crm_quality_summary,
        amo_queue_summary_path=amo_queue_summary_path,
        amo_queue_summary=amo_queue_summary,
        quarantine_manifest_path=quarantine_manifest_path,
        readback=readback,
    )
    blocked = sum(1 for item in gates if item["severity"] == "block" and not item["passed"])
    warnings = sum(1 for item in gates if item["severity"] == "warn" and not item["passed"])
    passed = sum(1 for item in gates if item["passed"])

    canonical_missing_asr = _int(canonical_summary.get("missing_asr_actionable"))
    canonical_missing_ra = _int(canonical_summary.get("missing_full_ra_actionable"))
    canonical_actionable = _int(canonical_summary.get("actionable_source_audio"))
    bucket_counts = _dict(amo_queue_summary.get("bucket_counts"))
    safe_pending = _int(bucket_counts.get("ready_single_contact_not_written"))
    stage1_complete = bool(amo_queue_summary) and safe_pending == 0 and readback["passed_expected_count_readbacks"] >= 2

    summary = CallProcessingReadinessSummary(
        schema_version=CALL_PROCESSING_READINESS_SCHEMA_VERSION,
        project_root=str(project_root),
        active_export_root=str(active_export_root) if active_export_root else None,
        gates=len(gates),
        passed=passed,
        blocked=blocked,
        warnings=warnings,
        canonical_actionable_calls=canonical_actionable,
        canonical_missing_asr=canonical_missing_asr,
        canonical_missing_ra=canonical_missing_ra,
        amo_ready_rows=_int(export_summary.get("amo_export_ready_rows")),
        safe_writeback_pending_rows=safe_pending,
        stage1_writeback_complete=stage1_complete,
        processing_pipeline_ready=blocked == 0,
        validation_ok=blocked == 0,
    )
    report: dict[str, Any] = {
        "summary": summary.to_json_dict(),
        "gates": gates,
        "inputs": {
            "canonical_export_pointer": str(pointer_path) if pointer_path else None,
            "export_summary": str(export_summary_path) if export_summary_path else None,
            "canonical_summary": str(canonical_summary_path) if canonical_summary_path else None,
            "stage15_summary": str(stage15_summary_path) if stage15_summary_path else None,
            "crm_quality_summary": str(crm_quality_summary_path) if crm_quality_summary_path else None,
            "amo_queue_summary": str(amo_queue_summary_path) if amo_queue_summary_path else None,
            "quarantine_manifest": str(quarantine_manifest_path) if quarantine_manifest_path else None,
        },
        "source_summaries": {
            "canonical": _project_safe_summary(canonical_summary),
            "export": _project_safe_summary(export_summary),
            "stage15": _project_safe_summary(stage15_summary),
            "crm_quality": _project_safe_summary(crm_quality_summary),
            "amo_queue": _project_safe_summary(amo_queue_summary),
            "readback": readback,
        },
        "next_actions": _next_actions(gates, safe_pending=safe_pending),
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
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _build_gates(**kwargs: Any) -> list[Mapping[str, Any]]:
    project_root: Path = kwargs["project_root"]
    pointer_path: Optional[Path] = kwargs["pointer_path"]
    active_export_root: Optional[Path] = kwargs["active_export_root"]
    export_summary_path: Optional[Path] = kwargs["export_summary_path"]
    export_summary: Mapping[str, Any] = kwargs["export_summary"]
    canonical_summary_path: Optional[Path] = kwargs["canonical_summary_path"]
    canonical_summary: Mapping[str, Any] = kwargs["canonical_summary"]
    stage15_summary_path: Optional[Path] = kwargs["stage15_summary_path"]
    stage15_summary: Mapping[str, Any] = kwargs["stage15_summary"]
    crm_quality_summary_path: Optional[Path] = kwargs["crm_quality_summary_path"]
    crm_quality_summary: Mapping[str, Any] = kwargs["crm_quality_summary"]
    amo_queue_summary_path: Optional[Path] = kwargs["amo_queue_summary_path"]
    amo_queue_summary: Mapping[str, Any] = kwargs["amo_queue_summary"]
    quarantine_manifest_path: Optional[Path] = kwargs["quarantine_manifest_path"]
    readback: Mapping[str, Any] = kwargs["readback"]

    export_csv = _path_from_value(_dict(export_summary.get("output_files")).get("amo_export_ready_csv"))
    crm_input = _path_from_value(crm_quality_summary.get("input"))
    queue_input = _path_from_value(amo_queue_summary.get("input_csv"))
    canonical_db = _path_from_value(export_summary.get("canonical_db"))
    canonical_summary_db = _path_from_value(_dict(canonical_summary.get("canonical_db")).get("path"))
    bucket_counts = _dict(amo_queue_summary.get("bucket_counts"))
    bucket_total = sum(_int(value) for value in bucket_counts.values())

    return [
        _gate("POINTER_EXISTS", bool(pointer_path and pointer_path.exists()), "Current export pointer file exists.", "block"),
        _gate("POINTER_EXPORT_SUMMARY_EXISTS", bool(active_export_root and export_summary_path and export_summary_path.exists()), "Pointer resolves to an export root with summary.json.", "block"),
        _gate(
            "CANONICAL_VALIDATED",
            bool(_dict(canonical_summary.get("validation")).get("passed")) and bool(_dict(canonical_summary.get("canonical_db")).get("passed")),
            "Canonical master validation passed.",
            "block",
        ),
        _gate(
            "NO_MISSING_ASR_OR_RA",
            _int(canonical_summary.get("missing_asr_actionable")) == 0 and _int(canonical_summary.get("missing_full_ra_actionable")) == 0,
            "All actionable source audio has ASR and full Resolve+Analyze.",
            "block",
        ),
        _gate(
            "EXPORT_BOUND_TO_CANONICAL",
            bool(canonical_db and canonical_summary_db and canonical_db.resolve(strict=False) == canonical_summary_db.resolve(strict=False)),
            "Strict CRM export references the current canonical DB.",
            "block",
        ),
        _gate(
            "STAGE15_CRM_READY",
            bool(stage15_summary.get("passed")) and bool(_dict(stage15_summary.get("readiness")).get("crm_quality_writeback_ready")),
            "Transcript-quality Stage15 gate passed for CRM writeback.",
            "block",
        ),
        _gate(
            "EXPORT_STAGE15_MATCHES",
            bool(stage15_summary_path and _path_from_value(export_summary.get("stage15_summary")) and stage15_summary_path.resolve(strict=False) == _path_from_value(export_summary.get("stage15_summary")).resolve(strict=False)),
            "Export summary references the same Stage15 gate used by readiness.",
            "block",
        ),
        _gate(
            "CRM_QUALITY_GATE_PASSED",
            bool(crm_quality_summary.get("passed")) and _int(crm_quality_summary.get("blocking_rows")) == 0,
            "CRM writeback quality gate passed with zero blocking rows.",
            "block",
        ),
        _gate(
            "CRM_GATE_INPUT_MATCHES_EXPORT",
            bool(export_csv and crm_input and export_csv.resolve(strict=False) == crm_input.resolve(strict=False)),
            "CRM quality gate input matches active AMO export CSV.",
            "block",
        ),
        _gate(
            "POPULATION_RECALL_GREEN",
            bool(_dict(crm_quality_summary.get("population_recall")).get("passed_for_live")),
            "Independent population-level recall guard is green.",
            "block",
        ),
        _gate(
            "AMO_QUEUE_MATCHES_EXPORT",
            bool(export_csv and queue_input and export_csv.resolve(strict=False) == queue_input.resolve(strict=False)),
            "AMO writeback queue was built from active strict export.",
            "block",
        ),
        _gate(
            "AMO_QUEUE_CLASSIFIED_ALL_ROWS",
            bool(bucket_counts) and bucket_total == _int(export_summary.get("amo_export_ready_rows")),
            "AMO queue buckets account for every AMO-ready row.",
            "block",
        ),
        _gate(
            "NO_SAFE_WRITEBACK_PENDING",
            _int(bucket_counts.get("ready_single_contact_not_written")) == 0,
            "No remaining single-contact ready row is waiting for live writeback in current Stage1 scope.",
            "warn",
        ),
        _gate(
            "READBACK_EXPECTED_COUNTS_PASSED",
            _int(readback.get("passed_expected_count_readbacks")) >= 2,
            "Recent staged live writebacks have expected-count readback gates.",
            "block",
        ),
        _gate(
            "QUARANTINE_MANIFEST_EXISTS",
            bool(quarantine_manifest_path and quarantine_manifest_path.exists()),
            "Stage2 cleanup quarantine has a restore manifest.",
            "warn",
        ),
        _gate("READ_ONLY_GATE", True, "This readiness command does not mutate runtime or CRM.", "block"),
    ]


def _gate(gate_id: str, passed: bool, reason: str, severity: str) -> Mapping[str, Any]:
    return {"gate": gate_id, "passed": bool(passed), "severity": severity, "reason": reason}


def _next_actions(gates: list[Mapping[str, Any]], *, safe_pending: int) -> list[str]:
    actions: list[str] = []
    for gate in gates:
        if gate["passed"]:
            continue
        gate_id = str(gate["gate"])
        if gate_id == "NO_SAFE_WRITEBACK_PENDING" and safe_pending > 0:
            actions.append("Build a staged dry-run/live/readback pack for the remaining safe AMO rows.")
        elif gate_id == "QUARANTINE_MANIFEST_EXISTS":
            actions.append("Create or restore Stage2 cleanup quarantine manifest before physical cleanup continues.")
        elif gate_id == "READBACK_EXPECTED_COUNTS_PASSED":
            actions.append("Run post-writeback readback with expected-count guard for every live stage.")
        else:
            actions.append(f"Resolve readiness gate: {gate_id}")
    if not actions:
        actions.append("Processing/readiness layer is green. Continue only with an explicitly scoped next stage.")
    return actions


def _collect_readback_evidence(project_root: Path) -> Mapping[str, Any]:
    root = project_root / "stable_runtime" / "amocrm_runtime" / "contact_writebacks"
    if not root.exists():
        return {"summaries_seen": 0, "passed_readbacks": 0, "passed_expected_count_readbacks": 0, "latest_passed": []}
    summaries = sorted(root.glob("*/readback*/summary.json"))
    passed = []
    expected = []
    for path in summaries:
        try:
            data = _load_json(path)
        except Exception:
            continue
        if data.get("passed") is True:
            item = {
                "path": str(path),
                "evaluated_rows": _int(data.get("evaluated_rows")),
                "expected_evaluated": data.get("expected_evaluated"),
                "expected_count_mismatch": bool(data.get("expected_count_mismatch")),
            }
            passed.append(item)
            if data.get("expected_evaluated") is not None and not data.get("expected_count_mismatch"):
                expected.append(item)
    return {
        "summaries_seen": len(summaries),
        "passed_readbacks": len(passed),
        "passed_expected_count_readbacks": len(expected),
        "latest_passed": passed[-5:],
    }


def _find_latest_crm_quality_summary(project_root: Path, export_summary: Mapping[str, Any]) -> Optional[Path]:
    export_csv = _path_from_value(_dict(export_summary.get("output_files")).get("amo_export_ready_csv"))
    candidates = sorted((project_root / "stable_runtime").glob("crm_writeback_quality_gate_*/summary.json"))
    matches = []
    for path in candidates:
        try:
            data = _load_json(path)
        except Exception:
            continue
        input_path = _path_from_value(data.get("input"))
        if export_csv and input_path and input_path.resolve(strict=False) == export_csv.resolve(strict=False):
            matches.append(path)
    return matches[-1] if matches else (candidates[-1] if candidates else None)


def _find_latest_amo_queue_summary(project_root: Path, export_summary: Mapping[str, Any]) -> Optional[Path]:
    export_csv = _path_from_value(_dict(export_summary.get("output_files")).get("amo_export_ready_csv"))
    candidates = sorted((project_root / "stable_runtime").glob("amo_writeback_queue_*/summary.json"))
    matches = []
    for path in candidates:
        try:
            data = _load_json(path)
        except Exception:
            continue
        input_path = _path_from_value(data.get("input_csv"))
        if export_csv and input_path and input_path.resolve(strict=False) == export_csv.resolve(strict=False):
            matches.append(path)
    return matches[-1] if matches else (candidates[-1] if candidates else None)


def _active_export_root(project_root: Path, pointer_path: Optional[Path]) -> Optional[Path]:
    if not pointer_path or not pointer_path.exists():
        return None
    text = pointer_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = project_root / "stable_runtime" / path
    return path.resolve(strict=False)


def _summary_from_db_path(value: Any) -> Optional[Path]:
    path = _path_from_value(value)
    if not path:
        return None
    return path.parent / "summary.json"


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


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _project_safe_summary(data: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = (
        "generated_at",
        "schema_version",
        "passed",
        "validation_ok",
        "blocking_rows",
        "rows",
        "amo_export_ready_rows",
        "manual_review_rows",
        "actionable_source_audio",
        "missing_asr_actionable",
        "missing_full_ra_actionable",
        "bucket_counts",
        "readiness",
        "population_recall",
        "crm_text_quality",
    )
    return {key: data[key] for key in keys if key in data}


__all__ = [
    "CALL_PROCESSING_READINESS_SCHEMA_VERSION",
    "build_call_processing_readiness_report",
]
