from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.recording_asset_ingest import READY_STATUS
from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


PROCESSING_LIFECYCLE_SCHEMA_VERSION = "processing_lifecycle_v1"
CANDIDATE_ASR_HANDOFF_DRY_RUN = "CANDIDATE_ASR_HANDOFF_DRY_RUN"
SKIP_ALREADY_IN_HANDOFF_MANIFEST = "SKIP_ALREADY_IN_HANDOFF_MANIFEST"
WAIT_RECORDING_ASSET = "WAIT_RECORDING_ASSET"
WAIT_RECORDING_DOWNLOAD = "WAIT_RECORDING_DOWNLOAD"
WAIT_ASSET_READY = "WAIT_ASSET_READY"
BLOCK_CAPTURE_STATUS = "BLOCK_CAPTURE_STATUS"
BLOCK_MISSING_RECORDING_REF = "BLOCK_MISSING_RECORDING_REF"
BLOCK_DUPLICATE_PROVIDER_CALL_ID = "BLOCK_DUPLICATE_PROVIDER_CALL_ID"
BLOCK_DUPLICATE_RECORDING_ID = "BLOCK_DUPLICATE_RECORDING_ID"


@dataclass(frozen=True)
class ProcessingLifecycleSummary:
    schema_version: str
    product_db_path: str
    product_root: str
    asset_db_path: Optional[str]
    handoff_manifest_path: Optional[str]
    capture_items_seen: int
    candidate_asr_handoff_dry_run: int
    skip_already_in_handoff_manifest: int
    wait_recording_asset: int
    wait_recording_download: int
    wait_asset_ready: int
    blocked_capture_status: int
    blocked_missing_recording_ref: int
    blocked_duplicate_provider_call_id: int
    blocked_duplicate_recording_id: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_processing_lifecycle_report(
    product_db_path: Path,
    product_root: Path,
    asset_db_path: Optional[Path] = None,
    handoff_manifest_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    *,
    limit: Optional[int] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root, asset_db_path, handoff_manifest_path, out_path = resolve_lifecycle_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        asset_db_path=asset_db_path,
        handoff_manifest_path=handoff_manifest_path,
        out_path=out_path,
    )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    capture_items = read_capture_items(product_db_path, limit=limit)
    assets = read_asset_index(asset_db_path) if asset_db_path else empty_asset_index()
    handoff = read_handoff_manifest_index(handoff_manifest_path) if handoff_manifest_path else empty_handoff_index()

    items = [
        classify_lifecycle_item(
            row,
            asset_index=assets,
            handoff_index=handoff,
        )
        for row in capture_items
    ]
    items = apply_duplicate_lifecycle_blocks(items)
    action_counts = Counter(clean(item.get("action")) for item in items)
    blocked = sum(
        int(action_counts[action])
        for action in (
            BLOCK_CAPTURE_STATUS,
            BLOCK_MISSING_RECORDING_REF,
            BLOCK_DUPLICATE_PROVIDER_CALL_ID,
            BLOCK_DUPLICATE_RECORDING_ID,
        )
    )
    warnings = int(
        action_counts[WAIT_RECORDING_ASSET]
        + action_counts[WAIT_RECORDING_DOWNLOAD]
        + action_counts[WAIT_ASSET_READY]
        + action_counts[SKIP_ALREADY_IN_HANDOFF_MANIFEST]
    )
    summary = ProcessingLifecycleSummary(
        schema_version=PROCESSING_LIFECYCLE_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        product_root=str(product_root),
        asset_db_path=str(asset_db_path) if asset_db_path else None,
        handoff_manifest_path=str(handoff_manifest_path) if handoff_manifest_path else None,
        capture_items_seen=len(capture_items),
        candidate_asr_handoff_dry_run=int(action_counts[CANDIDATE_ASR_HANDOFF_DRY_RUN]),
        skip_already_in_handoff_manifest=int(action_counts[SKIP_ALREADY_IN_HANDOFF_MANIFEST]),
        wait_recording_asset=int(action_counts[WAIT_RECORDING_ASSET]),
        wait_recording_download=int(action_counts[WAIT_RECORDING_DOWNLOAD]),
        wait_asset_ready=int(action_counts[WAIT_ASSET_READY]),
        blocked_capture_status=int(action_counts[BLOCK_CAPTURE_STATUS]),
        blocked_missing_recording_ref=int(action_counts[BLOCK_MISSING_RECORDING_REF]),
        blocked_duplicate_provider_call_id=int(action_counts[BLOCK_DUPLICATE_PROVIDER_CALL_ID]),
        blocked_duplicate_recording_id=int(action_counts[BLOCK_DUPLICATE_RECORDING_ID]),
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "items": items,
        "handoff_contract": {
            "auto_trigger_enabled": False,
            "requires_human_approval": True,
            "requires_asr_gate": True,
            "dry_run_actions": [CANDIDATE_ASR_HANDOFF_DRY_RUN],
            "worker_must_not_do": ["run_asr", "write_runtime_db", "write_crm"],
        },
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def classify_lifecycle_item(
    row: Mapping[str, Any],
    *,
    asset_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    handoff_index: Mapping[str, set[str]],
) -> Mapping[str, Any]:
    event_key = clean(row.get("event_key"))
    provider_call_id = clean(row.get("provider_call_id"))
    recording_id = clean(row.get("recording_ref")) or clean(row.get("audio_ref"))
    status = clean(row.get("status"))
    queue_item_id = build_queue_item_id(row, recording_id=recording_id)
    asset = find_asset(row, recording_id=recording_id, asset_index=asset_index)
    base = {
        "schema_version": PROCESSING_LIFECYCLE_SCHEMA_VERSION,
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": event_key,
        "provider_call_id": provider_call_id,
        "capture_inbox_item_id": int(row.get("id") or 0),
        "capture_status": status,
        "recording_id": recording_id or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "started_at": clean(row.get("started_at")) or None,
        "queue_item_id": queue_item_id,
        "asset_id": optional_int(asset.get("id")) if asset else None,
        "asset_status": clean(asset.get("status")) if asset else None,
        "asset_audio_path": clean(asset.get("audio_path")) if asset else None,
        "auto_trigger_enabled": False,
        "requires_human_approval": True,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "write_crm": False,
    }
    if status != "ready_for_capture":
        return base | {"action": BLOCK_CAPTURE_STATUS, "reason": f"capture_status_is_{status or 'empty'}"}
    if not recording_id:
        return base | {"action": BLOCK_MISSING_RECORDING_REF, "reason": "recording_ref_missing"}
    if event_key in handoff_index["event_keys"] or recording_id in handoff_index["recording_ids"]:
        return base | {"action": SKIP_ALREADY_IN_HANDOFF_MANIFEST, "reason": "already_present_in_handoff_manifest"}
    if asset is None:
        return base | {"action": WAIT_RECORDING_ASSET, "reason": "recording_asset_not_ingested_yet"}
    if clean(asset.get("status")) != READY_STATUS:
        return base | {"action": WAIT_ASSET_READY, "reason": f"asset_status_is_{clean(asset.get('status')) or 'empty'}"}
    if not clean(asset.get("audio_path")):
        return base | {"action": WAIT_RECORDING_DOWNLOAD, "reason": "asset_audio_path_missing"}
    return base | {"action": CANDIDATE_ASR_HANDOFF_DRY_RUN, "reason": "asset_ready_for_asr_handoff_dry_run"}


def apply_duplicate_lifecycle_blocks(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    provider_counts = Counter(clean(item.get("provider_call_id")) for item in items if clean(item.get("provider_call_id")))
    recording_counts = Counter(clean(item.get("recording_id")) for item in items if clean(item.get("recording_id")))
    result = []
    for source in items:
        item = dict(source)
        if clean(item.get("action")) in {
            CANDIDATE_ASR_HANDOFF_DRY_RUN,
            WAIT_RECORDING_ASSET,
            WAIT_RECORDING_DOWNLOAD,
            WAIT_ASSET_READY,
        }:
            if provider_counts.get(clean(item.get("provider_call_id")), 0) > 1:
                item["action"] = BLOCK_DUPLICATE_PROVIDER_CALL_ID
                item["reason"] = "provider_call_id_seen_multiple_times_in_capture_inbox"
            elif recording_counts.get(clean(item.get("recording_id")), 0) > 1:
                item["action"] = BLOCK_DUPLICATE_RECORDING_ID
                item["reason"] = "recording_id_seen_multiple_times_in_capture_inbox"
        result.append(item)
    return result


def read_capture_items(product_db_path: Path, limit: Optional[int]) -> list[Mapping[str, Any]]:
    limit_sql = "LIMIT ?" if limit is not None else ""
    params: tuple[Any, ...] = (int(limit),) if limit is not None else ()
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        ensure_table(con, "capture_inbox_items")
        rows = con.execute(
            f"""
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, direction, client_phone, manager_ref,
                   recording_ref, audio_ref, raw_payload_ref, enqueue_count,
                   first_seen_at, last_seen_at
              FROM capture_inbox_items
             ORDER BY started_at, id
             {limit_sql}
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def read_asset_index(asset_db_path: Path) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    uri = f"file:{quote(str(asset_db_path), safe='/:')}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        ensure_table(con, "captured_recording_assets")
        rows = con.execute(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id,
                   recording_id, audio_path, checksum_sha256, status, package_ref
              FROM captured_recording_assets
             ORDER BY id
            """
        ).fetchall()
    by_event: dict[str, Mapping[str, Any]] = {}
    by_recording: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item = dict(row)
        if clean(item.get("event_key")):
            by_event[clean(item.get("event_key"))] = item
        if clean(item.get("recording_id")):
            by_recording[clean(item.get("recording_id"))] = item
    return {"by_event": by_event, "by_recording": by_recording}


def read_handoff_manifest_index(path: Path) -> Mapping[str, set[str]]:
    event_keys: set[str] = set()
    recording_ids: set[str] = set()
    queue_item_ids: set[str] = set()
    if not path.exists():
        return {"event_keys": event_keys, "recording_ids": recording_ids, "queue_item_ids": queue_item_ids}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            continue
        if clean(value.get("event_key")):
            event_keys.add(clean(value.get("event_key")))
        if clean(value.get("recording_id")):
            recording_ids.add(clean(value.get("recording_id")))
        if clean(value.get("queue_item_id")):
            queue_item_ids.add(clean(value.get("queue_item_id")))
    return {"event_keys": event_keys, "recording_ids": recording_ids, "queue_item_ids": queue_item_ids}


def empty_asset_index() -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    return {"by_event": {}, "by_recording": {}}


def empty_handoff_index() -> Mapping[str, set[str]]:
    return {"event_keys": set(), "recording_ids": set(), "queue_item_ids": set()}


def find_asset(
    row: Mapping[str, Any],
    *,
    recording_id: str,
    asset_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Optional[Mapping[str, Any]]:
    return asset_index["by_event"].get(clean(row.get("event_key"))) or asset_index["by_recording"].get(recording_id)


def build_queue_item_id(row: Mapping[str, Any], *, recording_id: str) -> str:
    raw = "|".join(
        (
            clean(row.get("tenant_id")),
            clean(row.get("provider")),
            clean(recording_id),
            clean(row.get("event_key")),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_lifecycle_paths(
    product_db_path: Path,
    product_root: Path,
    asset_db_path: Optional[Path],
    handoff_manifest_path: Optional[Path],
    out_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path], Optional[Path], Optional[Path]]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    asset_db_path = asset_db_path.resolve(strict=False) if asset_db_path else None
    handoff_manifest_path = handoff_manifest_path.resolve(strict=False) if handoff_manifest_path else None
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    for label, path in (
        ("asset DB", asset_db_path),
        ("handoff manifest", handoff_manifest_path),
        ("lifecycle output", out_path),
    ):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"{label} must not be under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if asset_db_path is not None:
        if asset_db_path.name in RUNTIME_DB_FILENAMES:
            raise ValueError(f"refusing runtime-looking DB filename: {asset_db_path.name}")
        if not asset_db_path.exists() or not asset_db_path.is_file():
            raise FileNotFoundError(f"asset DB not found: {asset_db_path}")
    return product_db_path, product_root, asset_db_path, handoff_manifest_path, out_path


def ensure_table(con: sqlite3.Connection, name: str) -> None:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (clean(name),),
    ).fetchone()
    if row is None:
        raise ValueError(f"required table not found: {name}")


def optional_int(value: Any) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safety_contract() -> Mapping[str, bool]:
    return {
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
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
