from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.quarantine_import import (
    build_quarantine_import_plan,
    materialize_quarantine_package,
)
from mango_mvp.productization.recording_capture_download import guard_download_path
from mango_mvp.productization.test_ingest import clean


RECORDING_QUARANTINE_PACKAGE_SCHEMA_VERSION = "recording_quarantine_package_v1"


@dataclass(frozen=True)
class RecordingQuarantinePlanSummary:
    schema_version: str
    source_bridge_plan_path: str
    normalized_bridge_plan_path: str
    quarantine_plan_path: str
    package_root: str
    quarantine_dir: str
    metadata_csv_path: str
    total_bridge_items: int
    ready: int
    blocked: int
    metadata_rows: int
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_recording_quarantine_plan(
    source_bridge_plan_path: Path,
    product_root: Path,
    package_root: Path,
    quarantine_dir: Path,
    metadata_csv_path: Path,
    plan_path: Path,
    normalized_bridge_plan_path: Path,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    paths = resolve_plan_paths(
        source_bridge_plan_path=source_bridge_plan_path,
        product_root=product_root,
        package_root=package_root,
        quarantine_dir=quarantine_dir,
        metadata_csv_path=metadata_csv_path,
        plan_path=plan_path,
        normalized_bridge_plan_path=normalized_bridge_plan_path,
    )
    source_bridge_plan_path = paths["source_bridge_plan_path"]
    product_root = paths["product_root"]
    package_root = paths["package_root"]
    quarantine_dir = paths["quarantine_dir"]
    metadata_csv_path = paths["metadata_csv_path"]
    plan_path = paths["plan_path"]
    normalized_bridge_plan_path = paths["normalized_bridge_plan_path"]

    bridge_plan = normalize_bridge_plan(source_bridge_plan_path)
    path_audit = audit_bridge_plan_paths(bridge_plan, product_root=product_root)
    if path_audit["blocked"]:
        raise ValueError(f"bridge plan has unsafe paths: {path_audit}")
    write_json(normalized_bridge_plan_path, bridge_plan)

    plan = build_quarantine_import_plan(
        bridge_plan_path=normalized_bridge_plan_path,
        quarantine_dir=quarantine_dir,
        metadata_csv_path=metadata_csv_path,
        verify_checksum=verify_checksum,
    )
    plan_path_audit = audit_quarantine_plan_paths(plan, product_root=product_root)
    if plan_path_audit["blocked"]:
        raise ValueError(f"quarantine plan has unsafe paths: {plan_path_audit}")

    summary = RecordingQuarantinePlanSummary(
        schema_version=RECORDING_QUARANTINE_PACKAGE_SCHEMA_VERSION,
        source_bridge_plan_path=str(source_bridge_plan_path),
        normalized_bridge_plan_path=str(normalized_bridge_plan_path),
        quarantine_plan_path=str(plan_path),
        package_root=str(package_root),
        quarantine_dir=str(quarantine_dir),
        metadata_csv_path=str(metadata_csv_path),
        total_bridge_items=int(plan["summary"].get("total_bridge_items") or 0),
        ready=int(plan["summary"].get("ready") or 0),
        blocked=int(plan["summary"].get("blocked") or 0),
        metadata_rows=int(plan["summary"].get("metadata_rows") or 0),
        validation_ok=int(plan["summary"].get("blocked") or 0) == 0,
        warnings=int(plan["summary"].get("skipped_non_import_status") or 0),
    )
    report = {
        "summary": summary.to_json_dict(),
        "quarantine_plan": plan,
        "path_audit": {
            "bridge_plan": path_audit,
            "quarantine_plan": plan_path_audit,
        },
        "safety": safety_contract(materialize_audio=False),
    }
    write_json(plan_path, report)
    return report


def materialize_recording_quarantine_package(
    plan_path: Path,
    product_root: Path,
    out_path: Path,
    mode: str = "copy",
    verify_checksum: bool = True,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    plan_path = plan_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False)
    guard_download_path(plan_path, product_root, "recording quarantine plan")
    guard_download_path(out_path, product_root, "recording quarantine materialization audit")

    report = json.loads(plan_path.read_text(encoding="utf-8"))
    quarantine_plan = extract_quarantine_plan(report)
    plan_path_audit = audit_quarantine_plan_paths(quarantine_plan, product_root=product_root)
    if plan_path_audit["blocked"]:
        raise ValueError(f"quarantine plan has unsafe paths: {plan_path_audit}")

    materialize_input_path = plan_path.with_name(f"{plan_path.stem}__materialize_input.json")
    guard_download_path(materialize_input_path.resolve(strict=False), product_root, "materialization input plan")
    write_json(materialize_input_path, quarantine_plan)

    materialized = materialize_quarantine_package(
        plan_path=materialize_input_path,
        mode=mode,
        verify_checksum=verify_checksum,
        overwrite=overwrite,
    )
    materialized_path_audit = audit_materialized_paths(materialized, product_root=product_root)
    if materialized_path_audit["blocked"]:
        raise ValueError(f"materialized package has unsafe paths: {materialized_path_audit}")

    result = {
        "summary": {
            **materialized["summary"],
            "schema_version": RECORDING_QUARANTINE_PACKAGE_SCHEMA_VERSION,
            "recording_quarantine_plan_path": str(plan_path),
            "materialize_input_path": str(materialize_input_path),
            "out_path": str(out_path),
            "validation_ok": int(materialized["summary"].get("blocked") or 0) == 0,
        },
        "materialization": materialized,
        "path_audit": {
            "quarantine_plan": plan_path_audit,
            "materialized": materialized_path_audit,
        },
        "safety": safety_contract(materialize_audio=True),
    }
    write_json(out_path, result)
    return result


def normalize_bridge_plan(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping) and isinstance(data.get("items"), list):
        return dict(data)
    if isinstance(data, Mapping) and isinstance(data.get("bridge"), Mapping):
        bridge = data["bridge"]
        if isinstance(bridge.get("items"), list):
            return dict(bridge)
    raise ValueError("bridge plan must contain top-level items or nested bridge.items")


def extract_quarantine_plan(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report.get("quarantine_plan"), Mapping):
        return dict(report["quarantine_plan"])
    if isinstance(report.get("items"), list) and isinstance(report.get("summary"), Mapping):
        return dict(report)
    raise ValueError("recording quarantine plan report does not contain quarantine_plan")


def audit_bridge_plan_paths(bridge_plan: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    unsafe = []
    stable_runtime_refs = []
    for item in bridge_plan.get("items", []):
        if not isinstance(item, Mapping):
            continue
        path_text = clean(item.get("local_audio_path"))
        if not path_text:
            continue
        path = Path(path_text).resolve(strict=False)
        if "stable_runtime" in path.parts:
            stable_runtime_refs.append(path_text)
        if not is_under(path, product_root):
            unsafe.append(path_text)
    return {
        "items": len(bridge_plan.get("items", [])),
        "local_audio_paths_outside_product_root": len(unsafe),
        "stable_runtime_refs": len(stable_runtime_refs),
        "blocked": len(unsafe) + len(stable_runtime_refs),
        "samples": {
            "local_audio_paths_outside_product_root": unsafe[:20],
            "stable_runtime_refs": stable_runtime_refs[:20],
        },
    }


def audit_quarantine_plan_paths(plan: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    summary = plan.get("summary") or {}
    paths = [
        ("quarantine_dir", summary.get("quarantine_dir")),
        ("metadata_csv_path", summary.get("metadata_csv_path")),
    ]
    for item in plan.get("items", []):
        if not isinstance(item, Mapping):
            continue
        paths.append(("source_audio_path", item.get("source_audio_path")))
        paths.append(("target_audio_path", item.get("target_audio_path")))
    return audit_named_paths(paths, product_root)


def audit_materialized_paths(report: Mapping[str, Any], product_root: Path) -> Mapping[str, Any]:
    paths = []
    summary = report.get("summary") or {}
    paths.append(("quarantine_dir", summary.get("quarantine_dir")))
    for item in report.get("items", []):
        if not isinstance(item, Mapping):
            continue
        paths.append(("source_audio_path", item.get("source_audio_path")))
        paths.append(("target_audio_path", item.get("target_audio_path")))
    return audit_named_paths(paths, product_root)


def audit_named_paths(paths: list[tuple[str, Any]], product_root: Path) -> Mapping[str, Any]:
    unsafe = []
    stable_runtime_refs = []
    checked = 0
    for label, raw_path in paths:
        path_text = clean(raw_path)
        if not path_text:
            continue
        checked += 1
        path = Path(path_text).resolve(strict=False)
        if "stable_runtime" in path.parts:
            stable_runtime_refs.append({"label": label, "path": path_text})
        if not is_under(path, product_root):
            unsafe.append({"label": label, "path": path_text})
    return {
        "checked_paths": checked,
        "paths_outside_product_root": len(unsafe),
        "stable_runtime_refs": len(stable_runtime_refs),
        "blocked": len(unsafe) + len(stable_runtime_refs),
        "samples": {
            "paths_outside_product_root": unsafe[:20],
            "stable_runtime_refs": stable_runtime_refs[:20],
        },
    }


def resolve_plan_paths(
    source_bridge_plan_path: Path,
    product_root: Path,
    package_root: Path,
    quarantine_dir: Path,
    metadata_csv_path: Path,
    plan_path: Path,
    normalized_bridge_plan_path: Path,
) -> Mapping[str, Path]:
    product_root = product_root.resolve(strict=False)
    paths = {
        "product_root": product_root,
        "source_bridge_plan_path": source_bridge_plan_path.resolve(strict=False),
        "package_root": package_root.resolve(strict=False),
        "quarantine_dir": quarantine_dir.resolve(strict=False),
        "metadata_csv_path": metadata_csv_path.resolve(strict=False),
        "plan_path": plan_path.resolve(strict=False),
        "normalized_bridge_plan_path": normalized_bridge_plan_path.resolve(strict=False),
    }
    for label in (
        "source_bridge_plan_path",
        "package_root",
        "quarantine_dir",
        "metadata_csv_path",
        "plan_path",
        "normalized_bridge_plan_path",
    ):
        guard_download_path(paths[label], product_root, label)
    return paths


def safety_contract(materialize_audio: bool) -> Mapping[str, bool]:
    return {
        "materialize_audio_into_quarantine": materialize_audio,
        "copy_audio_to_legacy_source": False,
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
