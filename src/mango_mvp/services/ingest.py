from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from mango_mvp.models import CallRecord
from mango_mvp.utils.audio import probe_audio
from mango_mvp.utils.phone import normalize_phone

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
PHONE_RE = re.compile(r"^\+?\d{7,15}$")


def _as_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _pick(row: Dict[str, str], *keys: str) -> Optional[str]:
    lower = {k.strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = lower.get(key.lower())
        if value:
            return value
    return None


def _looks_like_phone(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value or "")
    return bool(PHONE_RE.fullmatch(normalized))


def _split_suffix_id(value: str) -> tuple[str, Optional[str]]:
    cleaned = (value or "").strip()
    if "_" not in cleaned:
        return cleaned, None
    base, tail = cleaned.rsplit("_", 1)
    if tail.isdigit():
        return base.strip(), tail
    return cleaned, None


def _parse_started_at_from_filename(date_part: str, time_part: str) -> Optional[datetime]:
    raw_date = (date_part or "").strip()
    raw_time = (time_part or "").strip()
    if not raw_date or not raw_time:
        return None
    try:
        return datetime.strptime(f"{raw_date} {raw_time}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        return None


def parse_filename_metadata(source_filename: str) -> Dict[str, Any]:
    stem = Path(source_filename).stem
    parts = stem.split("__")
    if len(parts) < 4:
        return {
            "phone": None,
            "manager_name": None,
            "source_call_id": None,
            "started_at": None,
        }

    date_part = parts[0].strip()
    time_part = parts[1].strip()
    left = parts[2].strip()
    right_raw = "__".join(parts[3:]).strip()
    right, suffix_call_id = _split_suffix_id(right_raw)

    left_phone = normalize_phone(left) if _looks_like_phone(left) else None
    right_phone = normalize_phone(right) if _looks_like_phone(right) else None

    if left_phone and right and not _looks_like_phone(right):
        manager_name = right
    elif right_phone and left and not _looks_like_phone(left):
        manager_name = left
    elif left and not _looks_like_phone(left):
        manager_name = left
    elif right and not _looks_like_phone(right):
        manager_name = right
    else:
        manager_name = None

    return {
        "phone": left_phone or right_phone,
        "manager_name": manager_name,
        "source_call_id": suffix_call_id,
        "started_at": _parse_started_at_from_filename(date_part, time_part),
    }


def load_metadata_index(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if csv_path is None:
        return {}
    index: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = _pick(row, "filename", "file_name", "recording", "record", "audio_file")
            if not filename:
                continue
            index[Path(filename).name] = row
    return index


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path


def ingest_from_directory(
    session: Session,
    recordings_dir: Path,
    metadata_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    metadata = load_metadata_index(metadata_csv)
    processed = 0
    inserted = 0
    skipped = 0

    files = iter_audio_files(recordings_dir)
    for file_path in files:
        if limit is not None and processed >= limit:
            break
        processed += 1
        abs_path = str(file_path.resolve())
        exists = session.scalar(select(CallRecord.id).where(CallRecord.source_file == abs_path))
        if exists:
            skipped += 1
            continue

        row = metadata.get(file_path.name, {})
        filename_meta = parse_filename_metadata(file_path.name)
        audio_meta = probe_audio(file_path)
        phone = normalize_phone(
            _pick(row, "phone", "client_phone", "abonent_number", "contact_phone")
        ) or filename_meta.get("phone")
        manager_name = (
            _pick(row, "manager", "manager_name", "operator")
            or filename_meta.get("manager_name")
        )
        source_call_id = _pick(row, "call_id", "id", "record_id") or filename_meta.get(
            "source_call_id"
        )
        started_at = _as_datetime(_pick(row, "started_at", "start_time", "date_time")) or filename_meta.get(
            "started_at"
        )
        call = CallRecord(
            source_file=abs_path,
            source_filename=file_path.name,
            source_call_id=source_call_id,
            audio_codec=audio_meta.get("codec_name"),
            sample_rate=audio_meta.get("sample_rate"),  # type: ignore[arg-type]
            channels=audio_meta.get("channels"),  # type: ignore[arg-type]
            duration_sec=audio_meta.get("duration_sec"),  # type: ignore[arg-type]
            phone=phone,
            manager_name=manager_name,
            direction=_pick(row, "direction", "call_direction"),
            started_at=started_at,
        )
        session.add(call)
        inserted += 1
        if inserted % 200 == 0:
            session.commit()

    session.commit()
    return {"processed": processed, "inserted": inserted, "skipped": skipped}
