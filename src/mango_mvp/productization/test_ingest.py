from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.config import Settings, get_settings
from mango_mvp.db import build_session_factory
from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, ingest_from_directory


TEST_INGEST_SCHEMA_VERSION = "quarantine_test_ingest_v1"
RUNTIME_DB_FILENAMES = {"mango_mvp.db", "ai_office.db"}
SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm")


@dataclass(frozen=True)
class QuarantineTestIngestSummary:
    schema_version: str
    db_path: str
    audio_dir: str
    metadata_csv_path: str
    replaced_existing_db: bool
    metadata_rows: int
    audio_files: int
    ingest_processed: int
    ingest_inserted: int
    ingest_skipped: int
    db_call_records: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def run_quarantine_test_ingest(
    audio_dir: Path,
    metadata_csv_path: Path,
    db_path: Path,
    out_allowed_root: Path,
    replace_existing: bool = False,
    allow_existing: bool = False,
    limit: Optional[int] = None,
    base_settings: Optional[Settings] = None,
) -> Mapping[str, Any]:
    audio_dir = audio_dir.resolve(strict=False)
    metadata_csv_path = metadata_csv_path.resolve(strict=False)
    db_path = db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)

    guard_test_ingest_paths(
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv_path,
        db_path=db_path,
        out_allowed_root=out_allowed_root,
    )

    metadata_rows = read_metadata_rows(metadata_csv_path)
    audio_files = list_audio_files(audio_dir)
    if replace_existing and allow_existing:
        raise ValueError("replace_existing and allow_existing are mutually exclusive")
    replaced_existing_db = False
    if replace_existing:
        replaced_existing_db = remove_disposable_sqlite_db(db_path, out_allowed_root=out_allowed_root)
    elif db_path.exists() and not allow_existing:
        raise FileExistsError(f"test ingest DB already exists: {db_path}")

    settings = dataclass_replace(
        base_settings or get_settings(),
        database_url=f"sqlite:///{db_path}",
        sqlite_wal_enabled=False,
    )
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        ingest_result = ingest_from_directory(
            session=session,
            recordings_dir=audio_dir,
            metadata_csv=metadata_csv_path,
            limit=limit,
        )

    audit = audit_quarantine_test_ingest_db(
        db_path=db_path,
        audio_dir=audio_dir,
        metadata_rows=metadata_rows,
        audio_files=audio_files,
    )
    blocked = int(audit["blocked"])
    warnings = int(audit["warnings"])
    validation_ok = (
        blocked == 0
        and int(audit["db_call_records"]) == len(metadata_rows)
        and int(audit["audio_files"]) == len(audio_files)
    )
    summary = QuarantineTestIngestSummary(
        schema_version=TEST_INGEST_SCHEMA_VERSION,
        db_path=str(db_path),
        audio_dir=str(audio_dir),
        metadata_csv_path=str(metadata_csv_path),
        replaced_existing_db=replaced_existing_db,
        metadata_rows=len(metadata_rows),
        audio_files=len(audio_files),
        ingest_processed=int(ingest_result["processed"]),
        ingest_inserted=int(ingest_result["inserted"]),
        ingest_skipped=int(ingest_result["skipped"]),
        db_call_records=int(audit["db_call_records"]),
        validation_ok=validation_ok,
        blocked=blocked,
        warnings=warnings,
    )
    return {
        "summary": summary.to_json_dict(),
        "ingest_result": ingest_result,
        "audit": audit,
    }


def audit_quarantine_test_ingest_db(
    db_path: Path,
    audio_dir: Path,
    metadata_rows: Sequence[Mapping[str, str]],
    audio_files: Sequence[Path],
) -> Mapping[str, Any]:
    db_rows = read_call_records(db_path)
    db_by_filename: dict[str, list[Mapping[str, Any]]] = {}
    for row in db_rows:
        db_by_filename.setdefault(str(row.get("source_filename") or ""), []).append(row)

    metadata_filenames = [Path(str(row.get("filename") or "")).name for row in metadata_rows if row.get("filename")]
    metadata_filename_counts = Counter(metadata_filenames)
    db_filename_counts = Counter(str(row.get("source_filename") or "") for row in db_rows)
    audio_names = {path.name for path in audio_files}

    missing_audio_for_metadata = sorted(name for name in metadata_filenames if name not in audio_names)
    missing_db_for_metadata = sorted(name for name in metadata_filenames if name not in db_by_filename)
    extra_db_not_in_metadata = sorted(name for name in db_filename_counts if name not in metadata_filename_counts)
    duplicate_db_source_filenames = {
        name: count for name, count in sorted(db_filename_counts.items()) if name and count > 1
    }
    duplicate_metadata_filenames = {
        name: count for name, count in sorted(metadata_filename_counts.items()) if name and count > 1
    }

    path_outside_audio_dir = []
    source_file_missing = []
    for row in db_rows:
        source_path = Path(str(row.get("source_file") or ""))
        if not path_is_relative_to(source_path, audio_dir):
            path_outside_audio_dir.append(str(source_path))
        if not source_path.exists():
            source_file_missing.append(str(source_path))

    mismatches = metadata_mismatches(metadata_rows=metadata_rows, db_by_filename=db_by_filename)
    status_counts = {
        "transcription_status": count_field(db_rows, "transcription_status"),
        "resolve_status": count_field(db_rows, "resolve_status"),
        "analysis_status": count_field(db_rows, "analysis_status"),
        "sync_status": count_field(db_rows, "sync_status"),
    }
    blocked_reasons = {
        "missing_audio_for_metadata": len(missing_audio_for_metadata),
        "missing_db_for_metadata": len(missing_db_for_metadata),
        "extra_db_not_in_metadata": len(extra_db_not_in_metadata),
        "duplicate_db_source_filenames": len(duplicate_db_source_filenames),
        "duplicate_metadata_filenames": len(duplicate_metadata_filenames),
        "path_outside_audio_dir": len(path_outside_audio_dir),
        "source_file_missing": len(source_file_missing),
        "source_call_id_mismatches": len(mismatches["source_call_id_mismatches"]),
        "phone_mismatches": len(mismatches["phone_mismatches"]),
        "direction_mismatches": len(mismatches["direction_mismatches"]),
    }
    blocked = sum(blocked_reasons.values())
    warning_reasons = {
        "phone_missing": count_missing(db_rows, "phone"),
        "manager_name_missing": count_missing(db_rows, "manager_name"),
        "started_at_missing": count_missing(db_rows, "started_at"),
        "duration_sec_missing": count_missing(db_rows, "duration_sec"),
        "audio_codec_missing": count_missing(db_rows, "audio_codec"),
        "sample_rate_missing": count_missing(db_rows, "sample_rate"),
        "channels_missing": count_missing(db_rows, "channels"),
    }
    warnings = sum(warning_reasons.values())
    return {
        "db_path": str(db_path),
        "audio_dir": str(audio_dir),
        "db_call_records": len(db_rows),
        "metadata_rows": len(metadata_rows),
        "audio_files": len(audio_files),
        "blocked": blocked,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "warning_reasons": warning_reasons,
        "status_counts": status_counts,
        "direction_counts": count_field(db_rows, "direction"),
        "manager_name_counts_top20": top_counts(db_rows, "manager_name", limit=20),
        "source_filename_unique": len(db_filename_counts),
        "source_file_unique": len({str(row.get("source_file") or "") for row in db_rows}),
        "metadata_filename_unique": len(metadata_filename_counts),
        "missing_audio_for_metadata": missing_audio_for_metadata[:100],
        "missing_db_for_metadata": missing_db_for_metadata[:100],
        "extra_db_not_in_metadata": extra_db_not_in_metadata[:100],
        "duplicate_db_source_filenames": duplicate_db_source_filenames,
        "duplicate_metadata_filenames": duplicate_metadata_filenames,
        "path_outside_audio_dir": path_outside_audio_dir[:100],
        "source_file_missing": source_file_missing[:100],
        "metadata_mismatches": mismatches,
        "current_call_records_model_gaps": [
            "recording_id is available in metadata.csv but is not stored in call_records",
            "event_key is available in metadata.csv but is not stored in call_records",
            "checksum_sha256 is available in metadata.csv but is not stored in call_records",
            "tenant_id/provider are available in metadata.csv but are not stored in call_records",
            "source_size_bytes is available in metadata.csv but is not stored in call_records",
        ],
        "samples": {
            "db_rows": db_rows[:20],
            "metadata_rows": [dict(row) for row in metadata_rows[:20]],
        },
    }


def guard_test_ingest_paths(
    audio_dir: Path,
    metadata_csv_path: Path,
    db_path: Path,
    out_allowed_root: Path,
) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing to use runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts:
        raise ValueError("refusing to write disposable DB under stable_runtime")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"test ingest DB must stay under allowed root: {out_allowed_root}")
    if db_path.exists() and db_path.is_dir():
        raise ValueError(f"test ingest DB path is a directory: {db_path}")
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")
    if not metadata_csv_path.exists() or not metadata_csv_path.is_file():
        raise FileNotFoundError(f"metadata csv not found: {metadata_csv_path}")


def remove_disposable_sqlite_db(db_path: Path, out_allowed_root: Path) -> bool:
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"refusing to remove DB outside allowed root: {out_allowed_root}")
    removed = False
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        path = Path(str(db_path) + suffix)
        if path.exists():
            if path.is_dir():
                raise ValueError(f"refusing to remove directory SQLite sidecar: {path}")
            path.unlink()
            removed = True
    return removed


def read_metadata_rows(metadata_csv_path: Path) -> list[Mapping[str, str]]:
    with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def list_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_call_records(db_path: Path) -> list[Mapping[str, Any]]:
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            select
              id,
              source_file,
              source_filename,
              source_call_id,
              audio_codec,
              sample_rate,
              channels,
              duration_sec,
              phone,
              manager_name,
              direction,
              started_at,
              transcription_status,
              resolve_status,
              analysis_status,
              sync_status,
              transcribe_attempts,
              resolve_attempts,
              analyze_attempts,
              sync_attempts
            from call_records
            order by id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def metadata_mismatches(
    metadata_rows: Sequence[Mapping[str, str]],
    db_by_filename: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    source_call_id_mismatches = []
    phone_mismatches = []
    direction_mismatches = []
    for row in metadata_rows:
        filename = Path(str(row.get("filename") or "")).name
        matches = db_by_filename.get(filename) or []
        if len(matches) != 1:
            continue
        db_row = matches[0]
        expected_call_id = clean(row.get("call_id"))
        if expected_call_id and clean(db_row.get("source_call_id")) != expected_call_id:
            source_call_id_mismatches.append(
                {
                    "filename": filename,
                    "expected": expected_call_id,
                    "actual": clean(db_row.get("source_call_id")),
                }
            )
        expected_phone = clean(row.get("phone")) or clean(row.get("client_phone"))
        if expected_phone and clean(db_row.get("phone")) != expected_phone:
            phone_mismatches.append(
                {
                    "filename": filename,
                    "expected": expected_phone,
                    "actual": clean(db_row.get("phone")),
                }
            )
        expected_direction = clean(row.get("direction"))
        if expected_direction and clean(db_row.get("direction")) != expected_direction:
            direction_mismatches.append(
                {
                    "filename": filename,
                    "expected": expected_direction,
                    "actual": clean(db_row.get("direction")),
                }
            )
    return {
        "source_call_id_mismatches": source_call_id_mismatches[:100],
        "phone_mismatches": phone_mismatches[:100],
        "direction_mismatches": direction_mismatches[:100],
    }


def count_field(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(row.get(field)) or "empty" for row in rows).items()))


def top_counts(rows: Sequence[Mapping[str, Any]], field: str, limit: int) -> Mapping[str, int]:
    return dict(Counter(clean(row.get(field)) or "empty" for row in rows).most_common(limit))


def count_missing(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(1 for row in rows if not clean(row.get(field)))


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
