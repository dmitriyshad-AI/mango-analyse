from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.contracts import Direction, TelephonyCallEvent


CAPTURE_MANIFEST_SCHEMA_VERSION = "capture_manifest_v1"
CAPTURE_RECOVERY_SCHEMA_VERSION = "capture_manifest_recovery_v1"
DEFAULT_CAPTURE_FILENAME_TZ = ZoneInfo("Europe/Moscow")
TERMINAL_EVENT_STATUSES = {
    "audio_integrity_quarantined",
    "downloaded",
    "duplicate_recording",
    "recording_retry_expired",
    "multiple_recordings_needs_review",
}
ASSET_STATUSES = {"downloaded", "multiple_recordings_needs_review"}
RECOVERABLE_EOF_JSON_MESSAGES = {
    "Expecting ',' delimiter",
    "Expecting ':' delimiter",
    "Expecting property name enclosed in double quotes",
    "Expecting value",
}
JSON_LITERAL_PREFIXES = frozenset(
    literal[:length]
    for literal in ("false", "null", "true")
    for length in range(1, len(literal))
)
INCOMPLETE_NUMBER_SUFFIX_RE = re.compile(r"(?:\.\d*|[eE][+-]?\d*)\Z")
INCOMPLETE_UNICODE_ESCAPE_RE = re.compile(r"u[0-9a-fA-F]{0,4}\Z")
REQUIRED_MANIFEST_STRING_FIELDS = (
    "created_at",
    "tenant_id",
    "provider",
    "event_key",
    "provider_call_id",
    "started_at",
    "direction",
    "status",
)


def manifest_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular_file(
    path: Path,
    flags: int,
    *,
    mode: int = 0o600,
    label: str,
) -> int:
    safe_flags = (
        flags
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, safe_flags, mode)
    try:
        is_regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError:
        os.close(descriptor)
        raise
    if not is_regular:
        os.close(descriptor)
        raise RuntimeError(f"{label} must be a regular file")
    return descriptor


def _assert_open_path_identity(
    path: Path,
    descriptor: int,
    *,
    label: str,
) -> os.stat_result:
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink < 1:
        raise RuntimeError(f"{label} changed while open")
    path_stat = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_dev != descriptor_stat.st_dev
        or path_stat.st_ino != descriptor_stat.st_ino
    ):
        raise RuntimeError(f"{label} changed while open")
    return descriptor_stat


def atomic_write_private_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    indent: Optional[int] = None,
) -> None:
    if os.path.lexists(path) and path.is_symlink():
        raise RuntimeError("refusing to replace symlink with private JSON")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=indent, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path) and path.is_symlink():
            raise RuntimeError("refusing to replace symlink with private JSON")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_recovery_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f".{manifest_path.name}.recovery.json")


def valid_capture_recovery_fingerprint(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    size_bytes = value.get("size_bytes")
    return bool(
        re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256") or ""))
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("valid_prefix_sha256") or ""))
        and isinstance(size_bytes, int)
        and not isinstance(size_bytes, bool)
        and size_bytes > 0
        and isinstance(value.get("valid_prefix_size_bytes"), int)
        and not isinstance(value.get("valid_prefix_size_bytes"), bool)
        and int(value["valid_prefix_size_bytes"]) >= 0
    )


def capture_recovery_incident_sha256(tails: Sequence[Mapping[str, Any]]) -> str:
    canonical = sorted(
        json.dumps(dict(item), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for item in tails
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def load_capture_recovery(path: Path) -> Mapping[str, Any]:
    if not os.path.lexists(path):
        return {
            "schema_version": CAPTURE_RECOVERY_SCHEMA_VERSION,
            "status": "resolved",
            "unresolved_count": 0,
            "tails": [],
            "incident_sha256": None,
            "acknowledged_incident_sha256": None,
        }
    if path.is_symlink():
        raise RuntimeError("capture recovery ledger must not be a symlink")
    descriptor = _open_regular_file(
        path,
        os.O_RDONLY,
        label="capture recovery ledger",
    )
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError("capture recovery ledger must be a JSON object")
    status = payload.get("status")
    tails = payload.get("tails")
    unresolved_count = payload.get("unresolved_count")
    fingerprint_keys = [
        json.dumps(dict(item), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for item in tails
    ] if isinstance(tails, list) and all(isinstance(item, Mapping) for item in tails) else []
    incident_sha256 = payload.get("incident_sha256")
    acknowledged_sha256 = payload.get("acknowledged_incident_sha256")
    if (
        payload.get("schema_version") != CAPTURE_RECOVERY_SCHEMA_VERSION
        or status not in {"resolved", "unresolved"}
        or not isinstance(tails, list)
        or any(not valid_capture_recovery_fingerprint(item) for item in tails)
        or not isinstance(unresolved_count, int)
        or isinstance(unresolved_count, bool)
        or unresolved_count != (len(tails) if status == "unresolved" else 0)
        or (status == "unresolved" and unresolved_count <= 0)
        or (status == "resolved" and bool(tails))
        or len(fingerprint_keys) != len(set(fingerprint_keys))
        or (
            status == "unresolved"
            and (
                not isinstance(incident_sha256, str)
                or incident_sha256 != capture_recovery_incident_sha256(tails)
            )
        )
        or (status == "resolved" and incident_sha256 is not None)
        or (
            acknowledged_sha256 is not None
            and not re.fullmatch(r"[0-9a-f]{64}", str(acknowledged_sha256))
        )
    ):
        raise RuntimeError("capture recovery ledger is invalid")
    return payload


def record_capture_recovery(path: Path, tail: bytes, valid_prefix: bytes) -> tuple[int, str]:
    if not tail:
        raise RuntimeError("capture recovery tail must not be empty")
    existing = load_capture_recovery(path)
    tails = list(existing.get("tails") or ()) if existing.get("status") == "unresolved" else []
    fingerprint = {
        "sha256": hashlib.sha256(tail).hexdigest(),
        "size_bytes": len(tail),
        "valid_prefix_sha256": hashlib.sha256(valid_prefix).hexdigest(),
        "valid_prefix_size_bytes": len(valid_prefix),
    }
    if fingerprint in tails:
        return len(tails), str(existing["incident_sha256"])
    tails.append(fingerprint)
    incident_sha256 = capture_recovery_incident_sha256(tails)
    atomic_write_private_json(
        path,
        {
            "schema_version": CAPTURE_RECOVERY_SCHEMA_VERSION,
            "status": "unresolved",
            "unresolved_count": len(tails),
            "tails": tails,
            "incident_sha256": incident_sha256,
            "acknowledged_incident_sha256": existing.get("acknowledged_incident_sha256"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return len(tails), incident_sha256


def acknowledge_capture_recovery(
    manifest_path: Path,
    *,
    expected_count: int,
    expected_incident_sha256: str,
) -> int:
    recovery_path = capture_recovery_path(manifest_path)
    if expected_count < 0:
        raise ValueError("expected recovery count must not be negative")
    if not manifest_path.exists() or not recovery_path.exists():
        if expected_count:
            raise RuntimeError("capture recovery ledger is missing")
        return 0
    descriptor = _open_regular_file(
        manifest_path,
        os.O_RDONLY,
        label="capture manifest",
    )
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _assert_open_path_identity(
            manifest_path,
            handle.fileno(),
            label="capture manifest",
        )
        existing = load_capture_recovery(recovery_path)
        count = int(existing.get("unresolved_count") or 0)
        if (
            count != expected_count
            or existing.get("incident_sha256") != expected_incident_sha256
        ):
            raise RuntimeError("capture recovery ledger changed before acknowledgement")
        if count:
            atomic_write_private_json(
                recovery_path,
                {
                    "schema_version": CAPTURE_RECOVERY_SCHEMA_VERSION,
                    "status": "resolved",
                    "unresolved_count": 0,
                    "tails": [],
                    "incident_sha256": None,
                    "acknowledged_incident_sha256": expected_incident_sha256,
                    "recovered_count": count,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        _assert_open_path_identity(
            manifest_path,
            handle.fileno(),
            label="capture manifest",
        )
        return count


def recoverable_json_tail(text: str, error: json.JSONDecodeError) -> bool:
    """Return True only when *text* is demonstrably a prefix of JSON object syntax."""
    if not text.lstrip().startswith("{"):
        return False
    if error.pos == len(text) and error.msg in RECOVERABLE_EOF_JSON_MESSAGES:
        return True
    if error.msg == "Unterminated string starting at":
        return error.pos < len(text) and text[error.pos] == '"'
    suffix = text[error.pos:]
    if error.msg == "Expecting value" and suffix in JSON_LITERAL_PREFIXES | {"-"}:
        return True
    if error.msg == "Expecting ',' delimiter" and INCOMPLETE_NUMBER_SUFFIX_RE.fullmatch(suffix):
        return True
    return bool(
        error.msg == "Invalid \\uXXXX escape"
        and error.pos > 0
        and text[error.pos - 1] == "\\"
        and INCOMPLETE_UNICODE_ESCAPE_RE.fullmatch(suffix)
    )


def recoverable_utf8_tail(line_bytes: bytes, error: UnicodeDecodeError) -> bool:
    if error.reason != "unexpected end of data" or error.end != len(line_bytes):
        return False
    prefix = line_bytes[: error.start].decode("utf-8")
    try:
        json.loads(prefix)
    except json.JSONDecodeError as json_error:
        return recoverable_json_tail(prefix, json_error)
    return False


def _previous_newline_offset(handle: BinaryIO, before: int) -> int:
    position = before
    while position > 0:
        start = max(0, position - 64 * 1024)
        handle.seek(start)
        chunk = handle.read(position - start)
        index = chunk.rfind(b"\n")
        if index >= 0:
            return start + index
        position = start
    return -1


def _last_nonempty_manifest_line(handle: BinaryIO) -> tuple[bytes, bool]:
    handle.seek(0, os.SEEK_END)
    boundary = handle.tell()
    terminated = False
    while True:
        newline = _previous_newline_offset(handle, boundary)
        start = newline + 1
        handle.seek(start)
        line = handle.read(boundary - start)
        if line.strip():
            return line, terminated
        if newline < 0:
            return b"", True
        boundary = newline
        terminated = True


def _capture_manifest_tail_status(handle: BinaryIO) -> str:
    line_bytes, terminated = _last_nonempty_manifest_line(handle)
    if not line_bytes:
        return "clean"
    try:
        line = line_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return (
            "incomplete"
            if not terminated and recoverable_utf8_tail(line_bytes, exc)
            else "invalid"
        )
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return (
            "incomplete"
            if not terminated and recoverable_json_tail(line, exc)
            else "invalid"
        )
    try:
        entry_from_json(payload)
    except (TypeError, ValueError):
        return "invalid"
    return "clean"


def capture_manifest_health(manifest_path: Path) -> Mapping[str, Any]:
    health: dict[str, Any] = {
        "tail_status": "missing",
        "recovery_status": "resolved",
        "recovery_unresolved_count": 0,
    }
    if os.path.lexists(manifest_path) and manifest_path.is_symlink():
        health["tail_status"] = "invalid"
        return health
    if not os.path.lexists(manifest_path):
        try:
            recovery = load_capture_recovery(capture_recovery_path(manifest_path))
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
            health["recovery_status"] = "invalid"
        else:
            health["recovery_status"] = recovery["status"]
            health["recovery_unresolved_count"] = recovery["unresolved_count"]
        return health
    try:
        descriptor = _open_regular_file(
            manifest_path,
            os.O_RDONLY,
            label="capture manifest",
        )
    except (OSError, RuntimeError):
        health["tail_status"] = "invalid"
        return health
    try:
        with os.fdopen(descriptor, "rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            before = manifest_signature(
                _assert_open_path_identity(
                    manifest_path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            try:
                recovery = load_capture_recovery(capture_recovery_path(manifest_path))
            except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
                health["recovery_status"] = "invalid"
            else:
                health["recovery_status"] = recovery["status"]
                health["recovery_unresolved_count"] = recovery["unresolved_count"]
            health["tail_status"] = _capture_manifest_tail_status(handle)
            after = manifest_signature(
                _assert_open_path_identity(
                    manifest_path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if after != before:
                health["tail_status"] = "invalid"
    except (OSError, RuntimeError):
        health["tail_status"] = "invalid"
    return health


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
    recording_ids: tuple[str, ...] = ()
    recording_paths: tuple[str, ...] = ()
    recording_assets: tuple[Mapping[str, Any], ...] = ()
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
    remediation_code: Optional[str] = None
    dry_run: bool = False
    host_id: Optional[str] = None
    recovery_state: Optional[str] = None

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
    needs_review_multiple_recordings: int
    manifest_path: str
    recordings_dir: str
    integrity_quarantined: int = 0
    incomplete_trailing_manifest_records: int = 0
    recovered_trailing_manifest_records: int = 0
    recovery_incident_sha256: Optional[str] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class CaptureManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_info = os.lstat(self.path.parent)
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or self.path.parent.is_symlink()
            or parent_info.st_uid != os.getuid()
        ):
            raise RuntimeError("capture manifest directory is unsafe")
        self.path.parent.chmod(0o700)
        self.recovery_path = capture_recovery_path(path)
        self._manifest_seen = os.path.lexists(path)
        recovery = load_capture_recovery(self.recovery_path)
        self.incomplete_trailing_records = 0
        self.recovered_trailing_records = int(recovery.get("unresolved_count") or 0)
        self.recovery_incident_sha256 = (
            str(recovery.get("incident_sha256"))
            if self.recovered_trailing_records
            else None
        )
        self._valid_prefix_bytes = 0
        self._needs_line_separator = False
        self._validated_signature: Optional[tuple[int, int, int, int, int]] = None
        self._lineage_prefix_bytes = 0
        self._validated_digest: Optional[bytes] = None
        self._validated_hasher: Optional[Any] = None
        self._validated_raw_size_bytes = 0
        self._validated_raw_digest: Optional[bytes] = None
        self._validated_tail_fingerprint: Optional[Mapping[str, Any]] = None
        self._cached_entries: list[ManifestEntry] = []

    def _assert_append_only_lineage(
        self,
        signature: tuple[int, int, int, int, int],
        raw: bytes,
        recovery: Mapping[str, Any],
    ) -> None:
        if self._validated_signature is None:
            return
        old_dev, old_ino, _, _, _ = self._validated_signature
        if signature[0] != old_dev or signature[1] != old_ino:
            raise RuntimeError("capture manifest inode changed during active store")
        if len(raw) < self._lineage_prefix_bytes:
            raise RuntimeError("capture manifest shrank during active store")
        if self._validated_digest is None:
            raise RuntimeError("capture manifest lineage is unavailable")
        if (
            hashlib.sha256(raw[: self._lineage_prefix_bytes]).digest()
            != self._validated_digest
        ):
            raise RuntimeError("capture manifest was rewritten during active store")
        if self._validated_tail_fingerprint is None:
            return
        unchanged_raw = bool(
            len(raw) == self._validated_raw_size_bytes
            and self._validated_raw_digest is not None
            and hashlib.sha256(raw).digest() == self._validated_raw_digest
        )
        recorded_tails = recovery.get("tails") if recovery.get("status") == "unresolved" else ()
        recovery_proves_tail = bool(
            isinstance(recorded_tails, list)
            and self._validated_tail_fingerprint in recorded_tails
        )
        if not unchanged_raw and not recovery_proves_tail:
            raise RuntimeError("capture manifest incomplete tail changed without recovery record")

    def _set_validated_snapshot(
        self,
        signature: tuple[int, int, int, int, int],
        raw: bytes,
    ) -> None:
        prefix = raw[: self._valid_prefix_bytes]
        hasher = hashlib.sha256(prefix)
        self._validated_signature = signature
        self._lineage_prefix_bytes = len(prefix)
        self._validated_digest = hasher.digest()
        self._validated_hasher = hasher
        self._validated_raw_size_bytes = len(raw)
        self._validated_raw_digest = hashlib.sha256(raw).digest()
        if self.incomplete_trailing_records:
            tail = raw[self._valid_prefix_bytes :]
            self._validated_tail_fingerprint = {
                "sha256": hashlib.sha256(tail).hexdigest(),
                "size_bytes": len(tail),
                "valid_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "valid_prefix_size_bytes": len(prefix),
            }
        else:
            self._validated_tail_fingerprint = None

    def _refresh_recovery(self) -> Mapping[str, Any]:
        recovery = load_capture_recovery(self.recovery_path)
        count = int(recovery.get("unresolved_count") or 0)
        incident_sha256 = str(recovery.get("incident_sha256")) if count else None
        if count and self.recovered_trailing_records and count <= self.recovered_trailing_records:
            if incident_sha256 != self.recovery_incident_sha256:
                raise RuntimeError("capture recovery ledger identity changed during active run")
            return recovery
        if count > self.recovered_trailing_records:
            self.recovered_trailing_records = count
            self.recovery_incident_sha256 = incident_sha256
        return recovery

    def ensure_exists(self) -> None:
        current_exists = os.path.lexists(self.path)
        allow_create = not self._manifest_seen and not current_exists
        created = False
        if allow_create:
            if os.path.lexists(self.recovery_path):
                raise RuntimeError("capture manifest is missing after recorded recovery")
            try:
                descriptor = _open_regular_file(
                    self.path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL,
                    mode=0o600,
                    label="capture manifest",
                )
                created = True
            except FileExistsError:
                descriptor = _open_regular_file(
                    self.path,
                    os.O_RDWR,
                    label="capture manifest",
                )
        else:
            descriptor = _open_regular_file(
                self.path,
                os.O_RDWR,
                label="capture manifest",
            )
        self._manifest_seen = True
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            before = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if os.fstat(handle.fileno()).st_mode & 0o777 != 0o600:
                os.fchmod(handle.fileno(), 0o600)
            handle.seek(0)
            raw = handle.read()
            os.fsync(handle.fileno())
            after = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if before[0:3] != after[0:3] or len(raw) != after[2]:
                raise RuntimeError("capture manifest changed while validating")
            recovery = self._refresh_recovery()
            self._assert_append_only_lineage(after, raw, recovery)
            self._parse_raw(raw)
            self._set_validated_snapshot(after, raw)
        if created:
            fsync_directory(self.path.parent)

    def read_entries(self) -> Sequence[ManifestEntry]:
        if not os.path.lexists(self.path):
            if self._manifest_seen or os.path.lexists(self.recovery_path):
                raise RuntimeError("capture manifest disappeared")
            self.incomplete_trailing_records = 0
            self._valid_prefix_bytes = 0
            self._needs_line_separator = False
            self._validated_signature = None
            self._lineage_prefix_bytes = 0
            self._validated_digest = None
            self._validated_hasher = None
            self._validated_raw_size_bytes = 0
            self._validated_raw_digest = None
            self._validated_tail_fingerprint = None
            self._cached_entries = []
            return ()
        descriptor = _open_regular_file(
            self.path,
            os.O_RDONLY,
            label="capture manifest",
        )
        self._manifest_seen = True
        with os.fdopen(descriptor, "rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            before = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            recovery = self._refresh_recovery()
            if before == self._validated_signature:
                cached_after = manifest_signature(
                    _assert_open_path_identity(
                        self.path,
                        handle.fileno(),
                        label="capture manifest",
                    )
                )
                if cached_after != before:
                    raise RuntimeError("capture manifest changed while reading")
                return tuple(self._cached_entries)
            raw = handle.read()
            after = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if before != after or len(raw) != after[2]:
                raise RuntimeError("capture manifest changed while reading")
        self._assert_append_only_lineage(after, raw, recovery)
        self._parse_raw(raw)
        self._set_validated_snapshot(after, raw)
        return tuple(self._cached_entries)

    def _parse_raw(self, raw: bytes) -> None:
        self.incomplete_trailing_records = 0
        self._valid_prefix_bytes = 0
        self._needs_line_separator = False
        self._cached_entries = []
        if not raw:
            return
        has_unterminated_tail = not raw.endswith(b"\n")
        lines = raw.split(b"\n")
        if not has_unterminated_tail:
            lines.pop()
        entries: list[ManifestEntry] = []
        self._valid_prefix_bytes = len(raw)
        offset = 0
        for index, line_bytes in enumerate(lines):
            is_unterminated_final_line = has_unterminated_tail and index == len(lines) - 1
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                if not is_unterminated_final_line or not recoverable_utf8_tail(line_bytes, exc):
                    raise
                self.incomplete_trailing_records = 1
                self._valid_prefix_bytes = offset
                break
            if not line.strip():
                offset += len(line_bytes) + (0 if is_unterminated_final_line else 1)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if not is_unterminated_final_line or not recoverable_json_tail(line, exc):
                    raise
                self.incomplete_trailing_records = 1
                self._valid_prefix_bytes = offset
                break
            entries.append(entry_from_json(payload))
            offset += len(line_bytes) + (0 if is_unterminated_final_line else 1)
        self._needs_line_separator = (
            bool(raw)
            and not raw.endswith(b"\n")
            and not self.incomplete_trailing_records
        )
        self._cached_entries = entries

    def latest_by_event_key(self) -> Mapping[str, ManifestEntry]:
        latest = {}
        for entry in self.read_entries():
            latest[entry.event_key] = entry
        return latest

    def latest_assets_by_recording_id(
        self,
        recordings_dir: Optional[Path] = None,
        *,
        require_integrity_metadata: bool = False,
    ) -> Mapping[str, ManifestEntry]:
        latest: dict[str, ManifestEntry] = {}
        for entry in self.read_entries():
            recording_ids = entry_recording_ids(entry)
            if entry.status not in ASSET_STATUSES or not recording_ids:
                continue
            if not manifest_assets_exist(
                entry,
                recordings_dir,
                require_integrity_metadata=require_integrity_metadata,
            ):
                continue
            for recording_id in recording_ids:
                latest[recording_id] = entry
        return latest

    def recover_incomplete_tail(self) -> int:
        if not self._manifest_seen:
            self.ensure_exists()
        descriptor = _open_regular_file(
            self.path,
            os.O_RDWR,
            label="capture manifest",
        )
        self._manifest_seen = True
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            before = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            recovery = self._refresh_recovery()
            handle.seek(0)
            raw = handle.read()
            stable_signature = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if before != stable_signature or len(raw) != stable_signature[2]:
                raise RuntimeError("capture manifest changed while reading")
            if stable_signature != self._validated_signature:
                self._assert_append_only_lineage(stable_signature, raw, recovery)
                self._parse_raw(raw)
                self._set_validated_snapshot(stable_signature, raw)
            if not self.incomplete_trailing_records:
                return 0
            valid_prefix = raw[: self._valid_prefix_bytes]
            tail = raw[self._valid_prefix_bytes :]
            (
                self.recovered_trailing_records,
                self.recovery_incident_sha256,
            ) = record_capture_recovery(
                self.recovery_path,
                tail,
                valid_prefix,
            )
            handle.seek(self._valid_prefix_bytes)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            after = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    handle.fileno(),
                    label="capture manifest",
                )
            )
            if after[2] != len(valid_prefix):
                raise RuntimeError("capture manifest changed while recovering")
            self._parse_raw(valid_prefix)
            self._set_validated_snapshot(after, valid_prefix)
            return 1

    def append(self, entry: ManifestEntry) -> None:
        encoded = (json.dumps(entry.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if not self._manifest_seen:
            self.ensure_exists()
        descriptor = _open_regular_file(
            self.path,
            os.O_RDWR,
            label="capture manifest",
        )
        self._manifest_seen = True
        with os.fdopen(descriptor, "r+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            _assert_open_path_identity(
                self.path,
                fh.fileno(),
                label="capture manifest",
            )
            recovery = self._refresh_recovery()
            current_stat = os.fstat(fh.fileno())
            if current_stat.st_mode & 0o777 != 0o600:
                os.fchmod(fh.fileno(), 0o600)
                current_stat = os.fstat(fh.fileno())
            signature = manifest_signature(current_stat)
            if signature != self._validated_signature:
                fh.seek(0)
                raw = fh.read()
                stable_signature = manifest_signature(
                    _assert_open_path_identity(
                        self.path,
                        fh.fileno(),
                        label="capture manifest",
                    )
                )
                if signature != stable_signature or len(raw) != stable_signature[2]:
                    raise RuntimeError("capture manifest changed while reading")
                self._assert_append_only_lineage(stable_signature, raw, recovery)
                self._parse_raw(raw)
                self._set_validated_snapshot(stable_signature, raw)
            if self._validated_hasher is None:
                raise RuntimeError("capture manifest lineage is unavailable")
            recovered_prefix: Optional[bytes] = None
            if self.incomplete_trailing_records:
                fh.seek(0)
                valid_prefix = fh.read(self._valid_prefix_bytes)
                recovered_prefix = valid_prefix
                fh.seek(self._valid_prefix_bytes)
                (
                    self.recovered_trailing_records,
                    self.recovery_incident_sha256,
                ) = record_capture_recovery(
                    self.recovery_path,
                    fh.read(),
                    valid_prefix,
                )
                fh.truncate(self._valid_prefix_bytes)
            fh.seek(0, os.SEEK_END)
            appended = b""
            if self._needs_line_separator:
                fh.write(b"\n")
                appended += b"\n"
            fh.write(encoded)
            appended += encoded
            fh.flush()
            os.fsync(fh.fileno())
            self.incomplete_trailing_records = 0
            self._needs_line_separator = False
            self._valid_prefix_bytes = fh.tell()
            self._cached_entries.append(entry)
            hasher = (
                hashlib.sha256(recovered_prefix)
                if recovered_prefix is not None
                else self._validated_hasher.copy()
            )
            hasher.update(appended)
            self._validated_hasher = hasher
            self._validated_digest = hasher.digest()
            self._lineage_prefix_bytes = self._valid_prefix_bytes
            self._validated_raw_size_bytes = self._valid_prefix_bytes
            self._validated_raw_digest = self._validated_digest
            self._validated_tail_fingerprint = None
            self._validated_signature = manifest_signature(
                _assert_open_path_identity(
                    self.path,
                    fh.fileno(),
                    label="capture manifest",
                )
            )
            if self._validated_signature[2] != self._valid_prefix_bytes:
                raise RuntimeError("capture manifest changed while appending")


def _existing_capture_target(path: Path) -> Optional[os.stat_result]:
    if not os.path.lexists(path):
        return None
    info = os.lstat(path)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
    ):
        raise RuntimeError("capture target is not a private regular file")
    return info


def stage_capture_events(
    events: Iterable[TelephonyCallEvent],
    manifest_store: CaptureManifestStore,
    recordings_dir: Path,
    downloader: Optional[RecordingDownloader] = None,
    dry_run: bool = False,
    sleep_sec: float = 0.0,
    validator: Optional[Callable[[Path], AudioValidation]] = None,
    host_id: Optional[str] = None,
    require_integrity_metadata: bool = False,
) -> CaptureStageSummary:
    recordings_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    recordings_info = os.lstat(recordings_dir)
    if (
        not stat.S_ISDIR(recordings_info.st_mode)
        or recordings_dir.is_symlink()
        or recordings_info.st_uid != os.getuid()
    ):
        raise RuntimeError("capture recordings directory is unsafe")
    recordings_dir.chmod(0o700)
    validate = validator or validate_audio_file
    latest_by_event = dict(manifest_store.latest_by_event_key())
    assets_by_recording = dict(
        manifest_store.latest_assets_by_recording_id(
            recordings_dir,
            require_integrity_metadata=require_integrity_metadata,
        )
    )

    counts = Counter()
    total = 0
    for event in events:
        total += 1
        existing = latest_by_event.get(event.event_key)
        recording_ids = merge_recording_ids(entry_recording_ids(existing), event_recording_ids(event))
        if recording_ids != event_recording_ids(event):
            event = replace(event, recording_ref=recording_ids[0] if recording_ids else None, recording_refs=recording_ids)
        recording_id = recording_ids[0] if recording_ids else None
        if (
            existing is not None
            and existing.status in TERMINAL_EVENT_STATUSES
            and existing.status != "recording_retry_expired"
            and manifest_assets_exist(
                existing,
                recordings_dir,
                require_integrity_metadata=require_integrity_metadata,
            )
            and entry_recording_ids(existing) == recording_ids
        ):
            counts["already_manifested"] += 1
            continue
        if existing is not None and existing.status == "skipped_no_recording" and not recording_id:
            counts["already_manifested"] += 1
            continue
        if (
            existing is not None
            and existing.status == "recording_retry_expired"
            and not recording_id
        ):
            # The caller owns the once-per-day retry cadence marker.  Avoid a
            # transient expired -> pending -> expired state and a duplicate
            # append when Mango still has no recording id.
            counts["already_manifested"] += 1
            continue
        if not recording_id:
            entry = manifest_entry_from_event(
                event, status="skipped_no_recording", host_id=host_id
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["skipped_no_recording"] += 1
            continue

        canonical_assets = [assets_by_recording.get(item) for item in recording_ids]
        canonical = (
            canonical_assets[0]
            if canonical_assets
            and all(item is not None for item in canonical_assets)
            and len({item.event_key for item in canonical_assets if item is not None}) == 1
            else None
        )
        if canonical is not None:
            entry = manifest_entry_from_event(
                event,
                status="duplicate_recording",
                canonical_event_key=canonical.event_key,
                canonical_recording_id=canonical.recording_id,
                canonical_audio_path=(
                    canonical.local_audio_path
                    or (canonical.recording_paths[0] if canonical.recording_paths else None)
                ),
                host_id=host_id,
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["duplicate_recording"] += 1
            continue

        target_path = recordings_dir / build_capture_audio_filename(event, recording_id)
        # A sealed asset is append-only evidence.  Missing or changed bytes
        # must never be silently downloaded again and adopted under the same
        # event, for either a single recording or a multi-part recording.  The
        # gate precedes dry-run so a probe cannot mask the sealed evidence.
        if (
            existing is not None
            and existing.recovery_state == "immutable_audio_violation"
        ):
            if existing.status != "audio_integrity_quarantined":
                entry = replace(
                    existing,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    status="audio_integrity_quarantined",
                    error="capture_target_integrity_mismatch",
                    remediation_code="manual_restore_or_quarantine_corrupted_audio",
                )
                manifest_store.append(entry)
                latest_by_event[event.event_key] = entry
            counts["integrity_quarantined"] += 1
            continue

        if (
            existing is not None
            and existing.status in ASSET_STATUSES
            and not manifest_assets_exist(
                existing,
                recordings_dir,
                require_integrity_metadata=require_integrity_metadata,
            )
        ):
            entry = replace(
                existing,
                created_at=datetime.now(timezone.utc).isoformat(),
                status="audio_integrity_quarantined",
                error="capture_target_integrity_mismatch",
                remediation_code="manual_restore_or_quarantine_corrupted_audio",
                recovery_state="immutable_audio_violation",
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["integrity_quarantined"] += 1
            continue

        if dry_run:
            entry = manifest_entry_from_event(
                event,
                status="dry_run_download",
                local_audio_path=str(target_path),
                dry_run=True,
                host_id=host_id,
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            counts["dry_run_download"] += 1
            continue

        if len(recording_ids) > 1:
            part_paths = recording_part_paths(target_path, recording_ids)
            try:
                if downloader is None:
                    raise RuntimeError("downloader is required when recording parts are missing")
                part_assets: list[Mapping[str, Any]] = []
                for part_id, part_path in zip(recording_ids, part_paths):
                    part_info = _existing_capture_target(part_path)
                    if part_info is not None and part_info.st_size <= 0:
                        part_path.unlink()
                        fsync_directory(part_path.parent)
                        part_info = None
                    if part_info is None:
                        downloader.download(recording_id=part_id, target_path=part_path)
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
                        part_info = _existing_capture_target(part_path)
                    if part_info is None or part_info.st_size <= 0:
                        raise RuntimeError("capture recording part is empty")
                    try:
                        part_audio = validate(part_path)
                    except Exception:
                        part_path.unlink(missing_ok=True)
                        raise
                    part_assets.append(
                        {
                            "recording_id": part_id,
                            "path": str(part_path),
                            "size_bytes": part_audio.size_bytes,
                            "checksum_sha256": part_audio.checksum_sha256,
                            "duration_sec": part_audio.duration_sec,
                            "codec_name": part_audio.codec_name,
                            "channels": part_audio.channels,
                            "sample_rate": part_audio.sample_rate,
                        }
                    )
                entry = manifest_entry_from_event(
                    event,
                    status="multiple_recordings_needs_review",
                    recording_paths=tuple(str(path) for path in part_paths),
                    recording_assets=tuple(part_assets),
                    remediation_code="manual_recording_selection",
                    host_id=host_id,
                )
                manifest_store.append(entry)
                latest_by_event[event.event_key] = entry
                for part_id in recording_ids:
                    assets_by_recording[part_id] = entry
                counts["needs_review_multiple_recordings"] += 1
            except Exception as exc:
                entry = manifest_entry_from_event(
                    event,
                    status="failed",
                    recording_paths=tuple(str(path) for path in part_paths if path.is_file()),
                    error=f"{type(exc).__name__}:capture_failed",
                    host_id=host_id,
                )
                manifest_store.append(entry)
                latest_by_event[event.event_key] = entry
                counts["failed"] += 1
            continue

        # A crash can leave a fully fsynced target before the durable manifest
        # append.  It may be adopted only when no prior sealed asset claims the
        # same target.  A changed sealed asset is evidence, not a new truth.
        try:
            target_info = _existing_capture_target(target_path)
            if target_info is not None and target_info.st_size <= 0:
                target_path.unlink()
                fsync_directory(target_path.parent)
                target_info = None
            reused_existing = target_info is not None
            audio: Optional[AudioValidation] = None
            if reused_existing and existing is not None and existing.status == "failed":
                try:
                    audio = validate(target_path)
                except Exception:
                    _existing_capture_target(target_path)
                    target_path.unlink()
                    fsync_directory(target_path.parent)
                    reused_existing = False
            if not reused_existing:
                if downloader is None:
                    raise RuntimeError("downloader is required when target file does not exist")
                downloader.download(recording_id=recording_id, target_path=target_path)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                downloaded_info = _existing_capture_target(target_path)
                if downloaded_info is None or downloaded_info.st_size <= 0:
                    raise RuntimeError("downloaded capture target is empty")

            audio = audio or validate(target_path)
            entry = manifest_entry_from_event(
                event,
                status="downloaded",
                local_audio_path=str(target_path),
                audio=audio,
                host_id=host_id,
                recovery_state=(
                    "recovered_late_recording"
                    if existing is not None
                    and existing.status == "recording_retry_expired"
                    else None
                ),
            )
            manifest_store.append(entry)
            latest_by_event[event.event_key] = entry
            assets_by_recording[recording_id] = entry
            counts["reused_existing_file" if reused_existing else "downloaded"] += 1
        except Exception as exc:
            late_retry = (
                existing is not None
                and existing.status == "recording_retry_expired"
            )
            entry = manifest_entry_from_event(
                event,
                status=("recording_retry_expired" if late_retry else "failed"),
                local_audio_path=str(target_path),
                error=f"{type(exc).__name__}:capture_failed",
                remediation_code=(
                    "manual_review_or_retry_if_recording_appears"
                    if late_retry
                    else None
                ),
                host_id=host_id,
                recovery_state=("late_recording_retry_failed" if late_retry else None),
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
        needs_review_multiple_recordings=counts["needs_review_multiple_recordings"],
        manifest_path=str(manifest_store.path),
        recordings_dir=str(recordings_dir),
        integrity_quarantined=counts["integrity_quarantined"],
        incomplete_trailing_manifest_records=manifest_store.incomplete_trailing_records,
        recovered_trailing_manifest_records=manifest_store.recovered_trailing_records,
        recovery_incident_sha256=manifest_store.recovery_incident_sha256,
    )


def manifest_audio_exists(
    entry: ManifestEntry,
    recordings_dir: Optional[Path] = None,
    *,
    require_integrity_metadata: bool = False,
) -> bool:
    path = Path(entry.local_audio_path) if entry.local_audio_path else None
    try:
        if not path or path.is_symlink() or not path.is_file():
            return False
        if recordings_dir is not None:
            path.resolve().relative_to(recordings_dir.resolve())
        info = path.stat()
        if info.st_size <= 0:
            return False
        if require_integrity_metadata and (
            not entry.size_bytes
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.checksum_sha256 or ""))
        ):
            return False
        if entry.size_bytes is not None and info.st_size != entry.size_bytes:
            return False
        if entry.checksum_sha256 and file_sha256(path) != entry.checksum_sha256:
            return False
        return True
    except ValueError:
        return False
    except OSError:
        return False


def manifest_assets_exist(
    entry: ManifestEntry,
    recordings_dir: Optional[Path] = None,
    *,
    require_integrity_metadata: bool = False,
) -> bool:
    if entry.status == "downloaded":
        return manifest_audio_exists(
            entry,
            recordings_dir,
            require_integrity_metadata=require_integrity_metadata,
        )
    if entry.status != "multiple_recordings_needs_review":
        return True
    if len(entry.recording_paths) != len(entry_recording_ids(entry)):
        return False
    if require_integrity_metadata and len(entry.recording_assets) != len(
        entry.recording_paths
    ):
        return False
    try:
        for index, raw_path in enumerate(entry.recording_paths):
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                return False
            if recordings_dir is not None:
                path.resolve().relative_to(recordings_dir.resolve())
            if require_integrity_metadata:
                asset = entry.recording_assets[index]
                if (
                    str(asset.get("recording_id") or "")
                    != entry_recording_ids(entry)[index]
                    or str(asset.get("path") or "") != raw_path
                    or not isinstance(asset.get("size_bytes"), int)
                    or int(asset["size_bytes"]) != path.stat().st_size
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(asset.get("checksum_sha256") or "")
                    )
                    or file_sha256(path) != asset.get("checksum_sha256")
                ):
                    return False
        return True
    except (OSError, ValueError):
        return False


def event_recording_ids(event: TelephonyCallEvent) -> tuple[str, ...]:
    fallback = event.recording_ref or event.recording_url
    return tuple(event.recording_refs) or ((fallback,) if fallback else ())


def entry_recording_ids(entry: Optional[ManifestEntry]) -> tuple[str, ...]:
    return () if entry is None else entry.recording_ids or ((entry.recording_id,) if entry.recording_id else ())


def merge_recording_ids(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group if item))


def recording_part_paths(target_path: Path, recording_ids: Sequence[str]) -> tuple[Path, ...]:
    paths = [target_path]
    for recording_id in recording_ids[1:]:
        suffix = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:12]
        paths.append(target_path.with_name(f"{target_path.stem}__part-{suffix}{target_path.suffix}"))
    return tuple(paths)


def manifest_entry_from_event(
    event: TelephonyCallEvent,
    status: str,
    local_audio_path: Optional[str] = None,
    canonical_event_key: Optional[str] = None,
    canonical_recording_id: Optional[str] = None,
    canonical_audio_path: Optional[str] = None,
    audio: Optional[AudioValidation] = None,
    recording_paths: Sequence[str] = (),
    recording_assets: Sequence[Mapping[str, Any]] = (),
    error: Optional[str] = None,
    remediation_code: Optional[str] = None,
    dry_run: bool = False,
    host_id: Optional[str] = None,
    recovery_state: Optional[str] = None,
) -> ManifestEntry:
    return ManifestEntry(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        tenant_id=event.tenant.tenant_id,
        provider=event.provider,
        event_key=event.event_key,
        provider_call_id=event.provider_call_id,
        recording_id=(event_recording_ids(event) or (None,))[0],
        recording_ids=event_recording_ids(event),
        recording_paths=tuple(recording_paths),
        recording_assets=tuple(dict(value) for value in recording_assets),
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
        remediation_code=remediation_code,
        dry_run=dry_run,
        host_id=host_id,
        recovery_state=recovery_state,
    )


def entry_from_json(data: Mapping[str, Any]) -> ManifestEntry:
    if not isinstance(data, Mapping):
        raise ValueError("capture manifest entry must be a JSON object")
    invalid_fields = [
        key
        for key in REQUIRED_MANIFEST_STRING_FIELDS
        if not isinstance(data.get(key), str) or not str(data[key]).strip()
    ]
    if invalid_fields:
        raise ValueError(f"capture manifest entry has invalid required fields: {','.join(invalid_fields)}")
    raw_recording_ids = data.get("recording_ids")
    recording_ids = raw_recording_ids if isinstance(raw_recording_ids, (list, tuple)) and raw_recording_ids else [data.get("recording_id")]
    raw_recording_paths = data.get("recording_paths")
    recording_paths = raw_recording_paths if isinstance(raw_recording_paths, (list, tuple)) else ()
    raw_recording_assets = data.get("recording_assets")
    recording_assets = (
        tuple(dict(item) for item in raw_recording_assets if isinstance(item, Mapping))
        if isinstance(raw_recording_assets, (list, tuple))
        else ()
    )
    return ManifestEntry(
        schema_version=str(data.get("schema_version") or CAPTURE_MANIFEST_SCHEMA_VERSION),
        created_at=str(data.get("created_at") or ""),
        tenant_id=str(data.get("tenant_id") or ""),
        provider=str(data.get("provider") or ""),
        event_key=str(data.get("event_key") or ""),
        provider_call_id=str(data.get("provider_call_id") or ""),
        recording_id=optional_str(data.get("recording_id")),
        recording_ids=tuple(str(item).strip() for item in recording_ids if str(item or "").strip()),
        recording_paths=tuple(str(item).strip() for item in recording_paths if str(item or "").strip()),
        recording_assets=recording_assets,
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
        remediation_code=optional_str(data.get("remediation_code")),
        dry_run=bool(data.get("dry_run", False)),
        host_id=optional_str(data.get("host_id")),
        recovery_state=optional_str(data.get("recovery_state")),
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
        str(Path(path))
        for entry in latest_by_event.values()
        for path in ((entry.local_audio_path,) + entry.recording_paths)
        if path
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
        "incomplete_trailing_records": store.incomplete_trailing_records,
        "recovered_trailing_records": store.recovered_trailing_records,
        "recovery_incident_sha256": store.recovery_incident_sha256,
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
