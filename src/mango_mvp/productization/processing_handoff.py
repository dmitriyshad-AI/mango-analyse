from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.recording_asset_ingest import READY_STATUS, sha256_file
from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


PROCESSING_HANDOFF_SCHEMA_VERSION = "processing_handoff_contract_v1"
ASR_HANDOFF_STATUS = "ready_for_asr"


@dataclass(frozen=True)
class ProcessingHandoffSummary:
    schema_version: str
    product_root: str
    asset_db_path: str
    out_dir: str
    manifest_path: str
    package_ref: Optional[str]
    source_assets_seen: int
    selected_assets: int
    ready_for_asr: int
    blocked: int
    skipped_not_ready: int
    warnings: int
    manifest_rows: int
    manifest_sha256: str
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_processing_handoff_dry_run(
    asset_db_path: Path,
    product_root: Path,
    out_dir: Path,
    manifest_path: Path,
    out_path: Optional[Path] = None,
    package_ref: Optional[str] = None,
    limit: Optional[int] = None,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    paths = resolve_handoff_paths(
        asset_db_path=asset_db_path,
        product_root=product_root,
        out_dir=out_dir,
        manifest_path=manifest_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    asset_db_path = paths["asset_db_path"]
    out_dir = paths["out_dir"]
    manifest_path = paths["manifest_path"]
    out_path = paths.get("out_path")
    package_ref = clean(package_ref) or None

    assets = read_recording_assets(asset_db_path, package_ref=package_ref, limit=limit)
    items = [plan_handoff_item(asset, product_root=product_root, out_dir=out_dir, verify_checksum=verify_checksum) for asset in assets]
    items = apply_queue_duplicate_blocks(items)
    manifest_rows = [manifest_item(item) for item in items if item["action"] == "PLAN_ASR_HANDOFF"]
    write_jsonl(manifest_path, manifest_rows)
    manifest_sha = sha256_file(manifest_path)
    action_counts = action_counts_for(items)
    warnings = count_warnings(items)
    blocked = int(action_counts.get("BLOCK_ASR_HANDOFF") or 0)
    ready = len(manifest_rows)
    summary = ProcessingHandoffSummary(
        schema_version=PROCESSING_HANDOFF_SCHEMA_VERSION,
        product_root=str(product_root),
        asset_db_path=str(asset_db_path),
        out_dir=str(out_dir),
        manifest_path=str(manifest_path),
        package_ref=package_ref,
        source_assets_seen=len(assets),
        selected_assets=len(items),
        ready_for_asr=ready,
        blocked=blocked,
        skipped_not_ready=int(action_counts.get("SKIP_NOT_READY_FOR_ASR") or 0),
        warnings=warnings,
        manifest_rows=len(manifest_rows),
        manifest_sha256=manifest_sha,
        validation_ok=blocked == 0 and ready == len(manifest_rows),
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts,
        "items": items,
        "manifest_samples": manifest_rows[:20],
        "contract": asr_handoff_contract(),
        "safety": {
            "product_db_writes": False,
            "asset_db_writes": False,
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "downloads_audio": False,
            "copies_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def read_recording_assets(
    asset_db_path: Path,
    package_ref: Optional[str],
    limit: Optional[int],
) -> list[Mapping[str, Any]]:
    params: list[Any] = []
    where = ""
    if package_ref:
        where = "WHERE package_ref = ?"
        params.append(package_ref)
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))
    uri = f"file:{quote(str(asset_db_path), safe='/:')}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        ensure_asset_db_schema(con)
        rows = con.execute(
            f"""
            SELECT
              id,
              tenant_id,
              provider,
              event_key,
              provider_call_id,
              recording_id,
              source,
              source_audio_path,
              audio_path,
              source_filename,
              checksum_sha256,
              size_bytes,
              duration_sec,
              started_at,
              direction,
              client_phone,
              manager_ref,
              manager_name,
              status,
              package_ref,
              metadata_json,
              first_ingested_at,
              updated_at
            FROM captured_recording_assets
            {where}
            ORDER BY started_at, id
            {limit_sql}
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def plan_handoff_item(
    asset: Mapping[str, Any],
    product_root: Path,
    out_dir: Path,
    verify_checksum: bool,
) -> dict[str, Any]:
    audio_path = Path(clean(asset.get("audio_path"))).resolve(strict=False)
    queue_item_id = build_queue_item_id(asset)
    planned_outputs = planned_asr_outputs(out_dir, asset, queue_item_id)
    item: dict[str, Any] = {
        "action": "PLAN_ASR_HANDOFF",
        "reason": "recording_asset_ready_for_asr_dry_run",
        "queue_status": ASR_HANDOFF_STATUS,
        "queue_item_id": queue_item_id,
        "asset_id": int(asset.get("id") or 0),
        "tenant_id": clean(asset.get("tenant_id")),
        "provider": clean(asset.get("provider")),
        "event_key": clean(asset.get("event_key")),
        "provider_call_id": clean(asset.get("provider_call_id")),
        "recording_id": clean(asset.get("recording_id")),
        "package_ref": clean(asset.get("package_ref")),
        "audio_path": str(audio_path),
        "source_filename": clean(asset.get("source_filename")),
        "checksum_sha256": clean(asset.get("checksum_sha256")).lower(),
        "size_bytes": optional_int(asset.get("size_bytes")),
        "duration_sec": optional_float(asset.get("duration_sec")),
        "started_at": clean(asset.get("started_at")) or None,
        "direction": clean(asset.get("direction")) or None,
        "client_phone": clean(asset.get("client_phone")) or None,
        "manager_ref": clean(asset.get("manager_ref")) or None,
        "manager_name": clean(asset.get("manager_name")) or None,
        "asset_status": clean(asset.get("status")),
        "planned_outputs": planned_outputs,
        "blocked_reasons": [],
        "warnings": [],
    }
    if item["asset_status"] != READY_STATUS:
        item["action"] = "SKIP_NOT_READY_FOR_ASR"
        item["reason"] = f"asset_status_is_{item['asset_status'] or 'empty'}"
        return item

    for field in ("tenant_id", "provider", "event_key", "provider_call_id", "recording_id", "checksum_sha256"):
        if not clean(item.get(field)):
            item["blocked_reasons"].append(f"missing_{field}")
    item["blocked_reasons"].extend(validate_audio_for_handoff(audio_path, product_root=product_root))
    if item["duration_sec"] is None:
        item["warnings"].append("duration_sec_missing")
    if not item["manager_ref"]:
        item["warnings"].append("manager_ref_missing")
    if not item["client_phone"]:
        item["warnings"].append("client_phone_missing")

    if not item["blocked_reasons"] and verify_checksum:
        actual_checksum = sha256_file(audio_path)
        item["actual_checksum_sha256"] = actual_checksum
        if clean(item["checksum_sha256"]) != actual_checksum:
            item["blocked_reasons"].append("checksum_sha256_mismatch")
    if item["blocked_reasons"]:
        item["action"] = "BLOCK_ASR_HANDOFF"
        item["reason"] = ",".join(sorted(set(item["blocked_reasons"])))
    return item


def validate_audio_for_handoff(audio_path: Path, product_root: Path) -> list[str]:
    blocked: list[str] = []
    if "stable_runtime" in audio_path.parts:
        blocked.append("audio_under_stable_runtime")
    if not path_is_relative_to(audio_path, product_root):
        blocked.append("audio_outside_product_root")
    if audio_path.suffix.lower() != ".mp3":
        blocked.append("unsupported_audio_extension")
    if not audio_path.exists():
        blocked.append("audio_missing")
    elif not audio_path.is_file():
        blocked.append("audio_not_file")
    elif audio_path.stat().st_size <= 0:
        blocked.append("zero_size_audio")
    return blocked


def apply_queue_duplicate_blocks(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    result: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item["action"] == "PLAN_ASR_HANDOFF" and counts.get(clean(item.get("queue_item_id")), 0) > 1:
            blocked = list(item.get("blocked_reasons") or [])
            blocked.append("duplicate_queue_item_id")
            item["blocked_reasons"] = sorted(set(blocked))
            item["action"] = "BLOCK_ASR_HANDOFF"
            item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def manifest_item(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema_version": PROCESSING_HANDOFF_SCHEMA_VERSION,
        "queue_status": ASR_HANDOFF_STATUS,
        "queue_item_id": item["queue_item_id"],
        "asset_id": item["asset_id"],
        "tenant_id": item["tenant_id"],
        "provider": item["provider"],
        "event_key": item["event_key"],
        "provider_call_id": item["provider_call_id"],
        "recording_id": item["recording_id"],
        "package_ref": item["package_ref"],
        "audio_path": item["audio_path"],
        "source_filename": item["source_filename"],
        "checksum_sha256": item["checksum_sha256"],
        "size_bytes": item["size_bytes"],
        "duration_sec": item["duration_sec"],
        "started_at": item["started_at"],
        "direction": item["direction"],
        "client_phone": item["client_phone"],
        "manager_ref": item["manager_ref"],
        "manager_name": item["manager_name"],
        "planned_outputs": item["planned_outputs"],
    }


def planned_asr_outputs(out_dir: Path, asset: Mapping[str, Any], queue_item_id: str) -> Mapping[str, str]:
    tenant = safe_slug(clean(asset.get("tenant_id")) or "tenant")
    provider = safe_slug(clean(asset.get("provider")) or "provider")
    stem = f"{clean(asset.get('started_at'))[:10] or 'undated'}__{queue_item_id[:16]}"
    root = out_dir / "planned_asr_outputs" / tenant / provider
    return {
        "transcript_json": str((root / f"{stem}.transcript.json").resolve(strict=False)),
        "transcript_txt": str((root / f"{stem}.transcript.txt").resolve(strict=False)),
        "asr_audit_json": str((root / f"{stem}.asr_audit.json").resolve(strict=False)),
    }


def build_queue_item_id(asset: Mapping[str, Any]) -> str:
    raw = "|".join(
        (
            clean(asset.get("tenant_id")),
            clean(asset.get("provider")),
            clean(asset.get("recording_id")),
            clean(asset.get("event_key")),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def asr_handoff_contract() -> Mapping[str, Any]:
    return {
        "schema_version": PROCESSING_HANDOFF_SCHEMA_VERSION,
        "queue_status": ASR_HANDOFF_STATUS,
        "worker_must_verify": ["audio_path_exists", "checksum_sha256", "runtime_target_approval"],
        "worker_must_not_do_in_dry_run": ["run_asr", "write_runtime_db", "write_crm"],
        "required_manifest_fields": [
            "queue_item_id",
            "tenant_id",
            "provider",
            "event_key",
            "recording_id",
            "audio_path",
            "checksum_sha256",
            "planned_outputs",
        ],
        "expected_worker_outputs": ["transcript_json", "transcript_txt", "asr_audit_json"],
    }


def resolve_handoff_paths(
    asset_db_path: Path,
    product_root: Path,
    out_dir: Path,
    manifest_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "asset_db_path": asset_db_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "manifest_path": manifest_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_handoff_paths(**paths)
    return paths


def guard_handoff_paths(
    product_root: Path,
    asset_db_path: Path,
    out_dir: Path,
    manifest_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("asset DB", asset_db_path),
        ("handoff out dir", out_dir),
        ("handoff manifest", manifest_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if asset_db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {asset_db_path.name}")
    if not asset_db_path.exists() or not asset_db_path.is_file():
        raise FileNotFoundError(f"asset DB not found: {asset_db_path}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing handoff audit output under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"handoff audit output must stay under product root: {product_root}")


def ensure_asset_db_schema(con: sqlite3.Connection) -> None:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'captured_recording_assets'"
    ).fetchone()
    if not row:
        raise ValueError("asset DB does not contain captured_recording_assets")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))


def count_warnings(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(item.get("warnings") or []) for item in items)


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def optional_float(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_slug(value: str) -> str:
    result = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            result.append(char)
        else:
            result.append("_")
    return "".join(result).strip("_") or "unknown"
