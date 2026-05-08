from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.contracts import Direction, TelephonyCallEvent


CAPTURE_MANIFEST_SCHEMA_VERSION = "capture_manifest_v1"
DEFAULT_CAPTURE_FILENAME_TZ = ZoneInfo("Europe/Moscow")
TERMINAL_EVENT_STATUSES = {
    "downloaded",
    "duplicate_recording",
    "skipped_no_recording",
}
ASSET_STATUSES = {"downloaded"}


class RecordingDownloader(Protocol):
    def download(self, recording_id: str, target_path: Path) -> int:
        """Download recording_id into target_path and return downloaded size in bytes."""


@dataclass(frozen=True)
class AudioValidation:
    size_bytes: int
    checksum_sha256: str
    duration_sec: Optional[float] = None
    codec_name: Optional[str] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class ManifestEntry:
    schema_version: str
    created_at: str
    tenant_id: str
    provider: str
    event_key: str
    provider_call_id: str
    recording_id: Optional[str]
    started_at: str
    ended_at: Optional[str]
    direction: str
    client_phone: Optional[str]
    manager_ref: Optional[str]
    status: str
    local_audio_path: Optional[str] = None
    canonical_event_key: Optional[str] = None
    canonical_recording_id: Optional[str] = None
    canonical_audio_path: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum_sha256: Optional[str] = None
    duration_sec: Optional[float] = None
    codec_name: Optional[str] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None
    dry_run: bool = False

    def to_json_dict(self) -> Mapping[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class CaptureStageSummary:
    total_events: int
    downloaded: int
    reused_existing_file: int
    duplicate_recording: int
    skipped_no_recording: int
    already_manifested: int
    dry_run_download: int
    failed: int
    manifest_path: str
    recordings_dir: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class CaptureManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read_entries(self) -> Sequence[ManifestEntry]:
        if not self.path.exists():
            return ()
        entries = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            entries.append(entry_from_json(json.loads(line)))
        return tuple(entries)

    def latest_by_event_key(self) -> Mapping[str, ManifestEntry]:
        latest = {}
        for entry in self.read_entries():
            latest[entry.event_key] = entry
        return latest

    def latest_assets_by_recording_id(self) -> Mapping[str, ManifestEntry]:
        latest = {}
        for entry in self.read_entries():
            if entry.status not in ASSET_STATUSES or not entry.recording_id:
                continue
            if not entry.local_audio_path:
                continue
            latest[entry.recording_id] = entry
        return latest

    def append(self, entry: ManifestEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def stage_capture_events(
    events: Iterable[TelephonyCallEvent],
    manifest_store: CaptureManifestStore,
    recordings_dir: Path,
    downloader: Optional[RecordingDownloader] = None,
    dry_run: bool = False,
    sleep_sec: float = 0.0,
    validator: Optional[Callable[[Path], AudioValidation]] = None,
) -> CaptureStageSummary:
    recordings_dir.mkdir(parents=True, exist_ok=True)
    validate = validator or validate_audio_file
    latest_by_event = dict(manifest_store.latest_by_event_key())
    assets_by_recording = dict(manifest_store.latest_assets_by_recording_id())

    counts = Counter()
    total = 0
    for event in events:
        total += 1
        existing = latest_by_event.get(event.event_key)
        if existing is not None and existing.status in TERMINAL_EVENT_STATUSES:
            counts["already_manifested"] += 1
            continue

        recording_id = event.recording_ref or event.recording_url
        if not recording_id:
            entry = manifest_entry_from_event(event, status="skipped_no_recording")
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["skipped_no_recording"] += 1
            continue

        canonical = assets_by_recording.get(recording_id)
        if canonical is not None:
            entry = manifest_entry_from_event(
                event,
                status="duplicate_recording",
                canonical_event_key=canonical.event_key,
                canonical_recording_id=canonical.recording_id,
                canonical_audio_path=canonical.local_audio_path,
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["duplicate_recording"] += 1
            continue

        target_path = recordings_dir / build_capture_audio_filename(event, recording_id)
        if dry_run:
            entry = manifest_entry_from_event(
                event,
                status="dry_run_download",
                local_audio_path=str(target_path),
                dry_run=True,
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["dry_run_download"] += 1
            continue

        try:
            reused_existing = target_path.exists() and target_path.stat().st_size > 0
            if not reused_existing:
                if downloader is None:
                    raise RuntimeError("downloader is required when target file does not exist")
                downloader.download(recording_id=recording_id, target_path=target_path)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            audio = validate(target_path)
            entry = manifest_entry_from_event(
                event,
                status="downloaded",
                local_audio_path=str(target_path),
                audio=audio,
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            assets_by_recording[recording_id] = entry
            counts["reused_existing_file" if reused_existing else "downloaded"] += 1
        except Exception as exc:
            entry = manifest_entry_from_event(
                event,
                status="failed",
                local_audio_path=str(target_path),
                error=str(exc),
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["failed"] += 1

    return CaptureStageSummary(
        total_events=total,
        downloaded=counts["downloaded"],
        reused_existing_file=counts["reused_existing_file"],
        duplicate_recording=counts["duplicate_recording"],
        skipped_no_recording=counts["skipped_no_recording"],
        already_manifested=counts["already_manifested"],
        dry_run_download=counts["dry_run_download"],
        failed=counts["failed"],
        manifest_path=str(manifest_store.path),
        recordings_dir=str(recordings_dir),
    )


def manifest_entry_from_event(
    event: TelephonyCallEvent,
    status: str,
    local_audio_path: Optional[str] = None,
    canonical_event_key: Optional[str] = None,
    canonical_recording_id: Optional[str] = None,
    canonical_audio_path: Optional[str] = None,
    audio: Optional[AudioValidation] = None,
    error: Optional[str] = None,
    dry_run: bool = False,
) -> ManifestEntry:
    return ManifestEntry(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        tenant_id=event.tenant.tenant_id,
        provider=event.provider,
        event_key=event.event_key,
        provider_call_id=event.provider_call_id,
        recording_id=event.recording_ref or event.recording_url,
        started_at=event.started_at.isoformat(),
        ended_at=event.ended_at.isoformat() if event.ended_at else None,
        direction=event.direction.value,
        client_phone=event.client_phone,
        manager_ref=event.manager_ref,
        status=status,
        local_audio_path=local_audio_path,
        canonical_event_key=canonical_event_key,
        canonical_recording_id=canonical_recording_id,
        canonical_audio_path=canonical_audio_path,
        size_bytes=audio.size_bytes if audio else None,
        checksum_sha256=audio.checksum_sha256 if audio else None,
        duration_sec=audio.duration_sec if audio else None,
        codec_name=audio.codec_name if audio else None,
        channels=audio.channels if audio else None,
        sample_rate=audio.sample_rate if audio else None,
        error=error,
        dry_run=dry_run,
    )


def entry_from_json(data: Mapping[str, Any]) -> ManifestEntry:
    return ManifestEntry(
        schema_version=str(data.get("schema_version") or CAPTURE_MANIFEST_SCHEMA_VERSION),
        created_at=str(data.get("created_at") or ""),
        tenant_id=str(data.get("tenant_id") or ""),
        provider=str(data.get("provider") or ""),
        event_key=str(data.get("event_key") or ""),
        provider_call_id=str(data.get("provider_call_id") or ""),
        recording_id=optional_str(data.get("recording_id")),
        started_at=str(data.get("started_at") or ""),
        ended_at=optional_str(data.get("ended_at")),
        direction=str(data.get("direction") or Direction.UNKNOWN.value),
        client_phone=optional_str(data.get("client_phone")),
        manager_ref=optional_str(data.get("manager_ref")),
        status=str(data.get("status") or ""),
        local_audio_path=optional_str(data.get("local_audio_path")),
        canonical_event_key=optional_str(data.get("canonical_event_key")),
        canonical_recording_id=optional_str(data.get("canonical_recording_id")),
        canonical_audio_path=optional_str(data.get("canonical_audio_path")),
        size_bytes=optional_int(data.get("size_bytes")),
        checksum_sha256=optional_str(data.get("checksum_sha256")),
        duration_sec=optional_float(data.get("duration_sec")),
        codec_name=optional_str(data.get("codec_name")),
        channels=optional_int(data.get("channels")),
        sample_rate=optional_int(data.get("sample_rate")),
        error=optional_str(data.get("error")),
        dry_run=bool(data.get("dry_run", False)),
    )


def audit_capture_manifest(manifest_path: Path, recordings_dir: Optional[Path] = None) -> Mapping[str, Any]:
    store = CaptureManifestStore(manifest_path)
    entries = store.read_entries()
    latest_by_event = store.latest_by_event_key()
    status_counts = Counter(entry.status for entry in entries)
    latest_status_counts = Counter(entry.status for entry in latest_by_event.values())

    missing_files = []
    zero_size_files = []
    checksum_missing = []
    duration_missing = []
    duplicate_recordings = Counter(
        entry.recording_id for entry in latest_by_event.values() if entry.recording_id
    )
    duplicate_recording_ids = {
        recording_id: count for recording_id, count in duplicate_recordings.items() if count > 1
    }

    for entry in latest_by_event.values():
        if entry.status != "downloaded":
            continue
        if not entry.local_audio_path:
            missing_files.append({"event_key": entry.event_key, "path": None})
            continue
        path = Path(entry.local_audio_path)
        if not path.exists():
            missing_files.append({"event_key": entry.event_key, "path": entry.local_audio_path})
            continue
        if path.stat().st_size <= 0:
            zero_size_files.append({"event_key": entry.event_key, "path": entry.local_audio_path})
        if not entry.checksum_sha256:
            checksum_missing.append(entry.event_key)
        if entry.duration_sec is None:
            duration_missing.append(entry.event_key)

    mp3_files = []
    if recordings_dir and recordings_dir.exists():
        mp3_files = list(recordings_dir.glob("*.mp3"))
    referenced_audio_paths = {
        str(Path(entry.local_audio_path))
        for entry in latest_by_event.values()
        if entry.local_audio_path
    }
    unreferenced_audio_files = [
        str(path)
        for path in mp3_files
        if str(path) not in referenced_audio_paths
    ]

    return {
        "manifest_path": str(manifest_path),
        "recordings_dir": str(recordings_dir) if recordings_dir else None,
        "manifest_rows": len(entries),
        "latest_unique_events": len(latest_by_event),
        "status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "downloaded_latest_events": latest_status_counts.get("downloaded", 0),
        "duplicate_recording_ids": len(duplicate_recording_ids),
        "missing_files": len(missing_files),
        "zero_size_files": len(zero_size_files),
        "checksum_missing": len(checksum_missing),
        "duration_missing": len(duration_missing),
        "recordings_dir_mp3_files": len(mp3_files),
        "recordings_dir_total_mb": round(sum(path.stat().st_size for path in mp3_files) / 1024 / 1024, 2),
        "unreferenced_audio_files": len(unreferenced_audio_files),
        "samples": {
            "missing_files": missing_files[:20],
            "zero_size_files": zero_size_files[:20],
            "checksum_missing": checksum_missing[:20],
            "duration_missing": duration_missing[:20],
            "unreferenced_audio_files": unreferenced_audio_files[:20],
        },
    }


def validate_audio_file(path: Path) -> AudioValidation:
    if not path.exists():
        raise FileNotFoundError(path)
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"Audio file is empty: {path}")

    from mango_mvp.utils.audio import probe_audio

    meta = probe_audio(path)
    return AudioValidation(
        size_bytes=size_bytes,
        checksum_sha256=file_sha256(path),
        duration_sec=optional_float(meta.get("duration_sec")),
        codec_name=optional_str(meta.get("codec_name")),
        channels=optional_int(meta.get("channels")),
        sample_rate=optional_int(meta.get("sample_rate")),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_capture_audio_filename(event: TelephonyCallEvent, recording_id: str) -> str:
    started = event.started_at.astimezone(DEFAULT_CAPTURE_FILENAME_TZ).strftime("%Y-%m-%d__%H-%M-%S")
    phone = sanitize_filename_part(event.client_phone or "no-phone")
    call_id = sanitize_filename_part(event.provider_call_id)
    return f"{started}__{phone}__mango_{call_id}.mp3"


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9А-Яа-яёЁ+_.=-]+", "_", value.strip())
    return cleaned.strip("._")[:120] or "unknown"


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
