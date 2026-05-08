from __future__ import annotations

import csv
import json
import sqlite3
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.capture_staging import (
    CaptureManifestStore,
    ManifestEntry,
    file_sha256,
    optional_float,
    optional_int,
)
from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata
from mango_mvp.utils.phone import normalize_phone


DEFAULT_BRIDGE_TZ = ZoneInfo("Europe/Moscow")


class BridgeStatus(str, Enum):
    WOULD_IMPORT = "would_import"
    ALREADY_PRESENT_AUDIO = "already_present_audio"
    ALREADY_PRESENT_DB = "already_present_db"
    BLOCKED_MANIFEST_STATUS = "blocked_manifest_status"
    BLOCKED_MISSING_FILE = "blocked_missing_file"
    BLOCKED_ZERO_SIZE_FILE = "blocked_zero_size_file"
    BLOCKED_CHECKSUM_MISMATCH = "blocked_checksum_mismatch"
    BLOCKED_DURATION_MISSING = "blocked_duration_missing"
    BLOCKED_STARTED_AT_INVALID = "blocked_started_at_invalid"


@dataclass(frozen=True)
class IndexedAudio:
    started_ts: int
    started_at: str
    phone: str
    path: str
    source_filename: str


@dataclass(frozen=True)
class IndexedDbCall:
    started_ts: int
    started_at: str
    phone: str
    db_path: str
    row_id: Optional[int]
    source_filename: Optional[str]
    source_file: Optional[str]
    source_call_id: Optional[str]


@dataclass(frozen=True)
class BridgePlanItem:
    event_key: str
    provider_call_id: str
    recording_id: Optional[str]
    status: str
    reason: str
    started_at: str
    started_at_msk: Optional[str]
    direction: str
    client_phone: Optional[str]
    manager_ref: Optional[str]
    local_audio_path: Optional[str]
    size_bytes: Optional[int]
    checksum_sha256: Optional[str]
    duration_sec: Optional[float]
    proposed_filename: Optional[str]
    proposed_metadata: Mapping[str, Any]
    matched_audio_path: Optional[str] = None
    matched_audio_delta_sec: Optional[int] = None
    matched_db_path: Optional[str] = None
    matched_db_row_id: Optional[int] = None
    matched_db_source_filename: Optional[str] = None
    matched_db_delta_sec: Optional[int] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class BridgePlanSummary:
    manifest_path: str
    source_dir: str
    db_paths: Sequence[str]
    tolerance_sec: int
    total_manifest_events: int
    bridge_status_counts: Mapping[str, int]
    manifest_status_counts: Mapping[str, int]
    source_audio_indexed: int
    db_calls_indexed: int
    checksum_verified: int
    checksum_skipped: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_pipeline_bridge_plan(
    manifest_path: Path,
    source_dir: Path,
    db_paths: Optional[Sequence[Path]] = None,
    tolerance_sec: int = 120,
    verify_checksum: bool = True,
) -> Mapping[str, Any]:
    store = CaptureManifestStore(manifest_path)
    entries = list(store.latest_by_event_key().values())
    audio_index = build_audio_index(source_dir)
    db_index = build_db_index(db_paths or ())
    status_counts = Counter()
    manifest_status_counts = Counter(entry.status for entry in entries)
    items = []
    checksum_verified = 0
    checksum_skipped = 0

    for entry in sorted(entries, key=lambda item: (item.started_at, item.event_key)):
        item, checksum_checked = plan_manifest_entry(
            entry=entry,
            audio_index=audio_index,
            db_index=db_index,
            source_dir=source_dir,
            tolerance_sec=tolerance_sec,
            verify_checksum=verify_checksum,
        )
        if checksum_checked:
            checksum_verified += 1
        else:
            checksum_skipped += 1
        status_counts[item.status] += 1
        items.append(item)

    summary = BridgePlanSummary(
        manifest_path=str(manifest_path),
        source_dir=str(source_dir),
        db_paths=tuple(str(path) for path in (db_paths or ())),
        tolerance_sec=tolerance_sec,
        total_manifest_events=len(entries),
        bridge_status_counts=dict(sorted(status_counts.items())),
        manifest_status_counts=dict(sorted(manifest_status_counts.items())),
        source_audio_indexed=count_index_items(audio_index),
        db_calls_indexed=count_index_items(db_index),
        checksum_verified=checksum_verified,
        checksum_skipped=checksum_skipped,
    )
    return {
        "summary": summary.to_json_dict(),
        "items": [item.to_json_dict() for item in items],
        "audit": audit_bridge_items(items),
    }


def plan_manifest_entry(
    entry: ManifestEntry,
    audio_index: Mapping[str, Sequence[IndexedAudio]],
    db_index: Mapping[str, Sequence[IndexedDbCall]],
    source_dir: Path,
    tolerance_sec: int,
    verify_checksum: bool,
) -> tuple[BridgePlanItem, bool]:
    base = base_item_kwargs(entry)
    started = parse_manifest_datetime(entry.started_at)
    if started is None:
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_STARTED_AT_INVALID.value,
                reason="manifest_started_at_invalid",
                started_at_msk=None,
                proposed_filename=None,
                proposed_metadata={},
            ),
            False,
        )

    started_msk = started.astimezone(DEFAULT_BRIDGE_TZ)
    proposed_filename = build_legacy_candidate_filename(entry=entry, started_msk=started_msk)
    proposed_metadata = build_proposed_metadata(entry=entry, started_msk=started_msk)

    if entry.status != "downloaded":
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_MANIFEST_STATUS.value,
                reason=f"manifest_status_is_{entry.status}",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
            ),
            False,
        )

    if not entry.local_audio_path:
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_MISSING_FILE.value,
                reason="manifest_local_audio_path_missing",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
            ),
            False,
        )

    local_path = Path(entry.local_audio_path)
    if not local_path.exists():
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_MISSING_FILE.value,
                reason="local_audio_file_missing",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
            ),
            False,
        )
    if local_path.stat().st_size <= 0:
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_ZERO_SIZE_FILE.value,
                reason="local_audio_file_zero_size",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
            ),
            False,
        )

    checksum_checked = False
    if verify_checksum and entry.checksum_sha256:
        checksum_checked = True
        actual_checksum = file_sha256(local_path)
        if actual_checksum != entry.checksum_sha256:
            return (
                BridgePlanItem(
                    **base,
                    status=BridgeStatus.BLOCKED_CHECKSUM_MISMATCH.value,
                    reason="checksum_mismatch",
                    started_at_msk=started_msk.isoformat(),
                    proposed_filename=proposed_filename,
                    proposed_metadata={**proposed_metadata, "actual_checksum_sha256": actual_checksum},
                ),
                checksum_checked,
            )

    if entry.duration_sec is None:
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.BLOCKED_DURATION_MISSING.value,
                reason="manifest_duration_missing",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
            ),
            checksum_checked,
        )

    phone = normalize_phone(entry.client_phone or "") if entry.client_phone else None
    started_ts = int(started_msk.timestamp())
    audio_match = find_nearest_match(audio_index, phone, started_ts, tolerance_sec)
    if audio_match is not None:
        audio, delta = audio_match
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.ALREADY_PRESENT_AUDIO.value,
                reason="phone_and_start_time_match_in_source_dir",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
                matched_audio_path=audio.path,
                matched_audio_delta_sec=delta,
            ),
            checksum_checked,
        )

    db_match = find_nearest_match(db_index, phone, started_ts, tolerance_sec)
    if db_match is not None:
        db_call, delta = db_match
        return (
            BridgePlanItem(
                **base,
                status=BridgeStatus.ALREADY_PRESENT_DB.value,
                reason="phone_and_start_time_match_in_readonly_db",
                started_at_msk=started_msk.isoformat(),
                proposed_filename=proposed_filename,
                proposed_metadata=proposed_metadata,
                matched_db_path=db_call.db_path,
                matched_db_row_id=db_call.row_id,
                matched_db_source_filename=db_call.source_filename,
                matched_db_delta_sec=delta,
            ),
            checksum_checked,
        )

    return (
        BridgePlanItem(
            **base,
            status=BridgeStatus.WOULD_IMPORT.value,
            reason="validated_capture_not_found_in_source_dir_or_db",
            started_at_msk=started_msk.isoformat(),
            proposed_filename=proposed_filename,
            proposed_metadata=proposed_metadata,
        ),
        checksum_checked,
    )


def build_audio_index(source_dir: Path) -> Mapping[str, Sequence[IndexedAudio]]:
    index: dict[str, list[IndexedAudio]] = defaultdict(list)
    if not source_dir.exists():
        return {}
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = parse_filename_metadata(path.name)
        started_at = meta.get("started_at")
        phone = normalize_phone(meta.get("phone")) if meta.get("phone") else None
        if not started_at or not phone:
            continue
        started_msk = started_at.replace(tzinfo=DEFAULT_BRIDGE_TZ)
        index[phone].append(
            IndexedAudio(
                started_ts=int(started_msk.timestamp()),
                started_at=started_at.isoformat(sep=" "),
                phone=phone,
                path=str(path),
                source_filename=path.name,
            )
        )
    return sort_index(index)


def build_db_index(db_paths: Sequence[Path]) -> Mapping[str, Sequence[IndexedDbCall]]:
    index: dict[str, list[IndexedDbCall]] = defaultdict(list)
    for db_path in db_paths:
        if not db_path.exists():
            continue
        scan_db_into_index(db_path, index)
    return sort_index(index)


def scan_db_into_index(db_path: Path, index: dict[str, list[IndexedDbCall]]) -> None:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        if not has_table(conn, "call_records"):
            return
        columns = table_columns(conn, "call_records")
        select_columns = [
            "id" if "id" in columns else "NULL",
            "source_filename" if "source_filename" in columns else "NULL",
            "source_file" if "source_file" in columns else "NULL",
            "source_call_id" if "source_call_id" in columns else "NULL",
            "phone" if "phone" in columns else "NULL",
            "started_at" if "started_at" in columns else "NULL",
        ]
        query = f"SELECT {', '.join(select_columns)} FROM call_records"
        for row in conn.execute(query):
            row_id, source_filename, source_file, source_call_id, phone_raw, started_raw = row
            phone = normalize_phone(phone_raw) if phone_raw else None
            started = parse_db_datetime(started_raw)
            if not phone or started is None:
                continue
            if started.tzinfo is None or started.utcoffset() is None:
                started_msk = started.replace(tzinfo=DEFAULT_BRIDGE_TZ)
            else:
                started_msk = started.astimezone(DEFAULT_BRIDGE_TZ)
            index[phone].append(
                IndexedDbCall(
                    started_ts=int(started_msk.timestamp()),
                    started_at=started_msk.isoformat(),
                    phone=phone,
                    db_path=str(db_path),
                    row_id=optional_int(row_id),
                    source_filename=optional_str(source_filename),
                    source_file=optional_str(source_file),
                    source_call_id=optional_str(source_call_id),
                )
            )


def find_nearest_match(
    index: Mapping[str, Sequence[Any]],
    phone: Optional[str],
    started_ts: int,
    tolerance_sec: int,
) -> Optional[tuple[Any, int]]:
    if not phone:
        return None
    items = index.get(phone)
    if not items:
        return None
    keys = [item.started_ts for item in items]
    pos = bisect_left(keys, started_ts)
    candidates = []
    if pos < len(items):
        candidates.append(items[pos])
    if pos > 0:
        candidates.append(items[pos - 1])
    best = None
    best_delta = None
    for candidate in candidates:
        delta = abs(candidate.started_ts - started_ts)
        if best_delta is None or delta < best_delta:
            best = candidate
            best_delta = delta
    if best is not None and best_delta is not None and best_delta <= tolerance_sec:
        return best, best_delta
    return None


def audit_bridge_items(items: Sequence[BridgePlanItem]) -> Mapping[str, Any]:
    by_status = Counter(item.status for item in items)
    by_day = Counter((item.started_at_msk or item.started_at)[:10] for item in items)
    would_import = [item for item in items if item.status == BridgeStatus.WOULD_IMPORT.value]
    blocked = [item for item in items if item.status.startswith("blocked_")]
    return {
        "by_status": dict(sorted(by_status.items())),
        "by_day": dict(sorted(by_day.items())),
        "would_import": len(would_import),
        "blocked": len(blocked),
        "samples": {
            "would_import": [sample_item(item) for item in would_import[:20]],
            "blocked": [sample_item(item) for item in blocked[:20]],
        },
    }


def write_bridge_plan_csv(plan: Mapping[str, Any], csv_path: Path) -> None:
    items = plan.get("items", [])
    fieldnames = [
        "status",
        "reason",
        "started_at_msk",
        "client_phone",
        "manager_ref",
        "duration_sec",
        "provider_call_id",
        "recording_id",
        "local_audio_path",
        "proposed_filename",
        "matched_audio_path",
        "matched_audio_delta_sec",
        "matched_db_path",
        "matched_db_row_id",
        "matched_db_source_filename",
        "matched_db_delta_sec",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def build_legacy_candidate_filename(entry: ManifestEntry, started_msk: datetime) -> str:
    date_part = started_msk.strftime("%Y-%m-%d")
    time_part = started_msk.strftime("%H-%M-%S")
    phone = normalize_phone(entry.client_phone or "") or "unknown_phone"
    phone_part = phone.lstrip("+")
    manager = sanitize_filename_part(f"mango_{entry.manager_ref or 'unknown'}")
    call_id = sanitize_filename_part(entry.provider_call_id)
    return f"{date_part}__{time_part}__{phone_part}__{manager}_{call_id}.mp3"


def build_proposed_metadata(entry: ManifestEntry, started_msk: datetime) -> Mapping[str, Any]:
    return {
        "source": "mango_api_capture",
        "tenant_id": entry.tenant_id,
        "provider": entry.provider,
        "event_key": entry.event_key,
        "provider_call_id": entry.provider_call_id,
        "recording_id": entry.recording_id,
        "started_at_msk": started_msk.isoformat(),
        "client_phone": entry.client_phone,
        "manager_ref": entry.manager_ref,
        "direction": entry.direction,
        "duration_sec": entry.duration_sec,
        "checksum_sha256": entry.checksum_sha256,
    }


def base_item_kwargs(entry: ManifestEntry) -> Mapping[str, Any]:
    return {
        "event_key": entry.event_key,
        "provider_call_id": entry.provider_call_id,
        "recording_id": entry.recording_id,
        "started_at": entry.started_at,
        "direction": entry.direction,
        "client_phone": entry.client_phone,
        "manager_ref": entry.manager_ref,
        "local_audio_path": entry.local_audio_path,
        "size_bytes": entry.size_bytes,
        "checksum_sha256": entry.checksum_sha256,
        "duration_sec": entry.duration_sec,
    }


def sample_item(item: BridgePlanItem) -> Mapping[str, Any]:
    return {
        "status": item.status,
        "reason": item.reason,
        "started_at_msk": item.started_at_msk,
        "client_phone": item.client_phone,
        "manager_ref": item.manager_ref,
        "provider_call_id": item.provider_call_id,
        "recording_id": item.recording_id,
        "proposed_filename": item.proposed_filename,
    }


def parse_manifest_datetime(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_db_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    )


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}


def sort_index(index: Mapping[str, Sequence[Any]]) -> Mapping[str, Sequence[Any]]:
    return {key: tuple(sorted(items, key=lambda item: item.started_ts)) for key, items in index.items()}


def count_index_items(index: Mapping[str, Sequence[Any]]) -> int:
    return sum(len(items) for items in index.values())


def sanitize_filename_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"+", "_", ".", "-", "="} else "_" for ch in value.strip())
    return cleaned.strip("._")[:120] or "unknown"


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
