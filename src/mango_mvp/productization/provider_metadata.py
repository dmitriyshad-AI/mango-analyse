from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.productization.test_ingest import (
    RUNTIME_DB_FILENAMES,
    clean,
    path_is_relative_to,
    read_metadata_rows,
)


PROVIDER_METADATA_SCHEMA_VERSION = "provider_call_metadata_v1"
PROVIDER_METADATA_TABLE = "provider_call_metadata"
PROVIDER_METADATA_REQUIRED_FIELDS = (
    "filename",
    "tenant_id",
    "provider",
    "provider_call_id",
    "recording_id",
    "event_key",
    "checksum_sha256",
    "source_size_bytes",
    "manager",
)


@dataclass(frozen=True)
class ProviderMetadataSummary:
    schema_version: str
    db_path: str
    metadata_csv_path: str
    table_name: str
    replaced_existing_table: bool
    metadata_rows: int
    call_records: int
    sidecar_rows: int
    inserted: int
    updated: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def install_provider_metadata_sidecar(
    db_path: Path,
    metadata_csv_path: Path,
    out_allowed_root: Path,
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    metadata_csv_path = metadata_csv_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    guard_provider_metadata_paths(
        db_path=db_path,
        metadata_csv_path=metadata_csv_path,
        out_allowed_root=out_allowed_root,
    )

    metadata_rows = read_metadata_rows(metadata_csv_path)
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        call_records = read_call_record_index(con)
        preflight = preflight_provider_metadata(metadata_rows=metadata_rows, call_records=call_records)
        if int(preflight["blocked"]) > 0:
            summary = ProviderMetadataSummary(
                schema_version=PROVIDER_METADATA_SCHEMA_VERSION,
                db_path=str(db_path),
                metadata_csv_path=str(metadata_csv_path),
                table_name=PROVIDER_METADATA_TABLE,
                replaced_existing_table=False,
                metadata_rows=len(metadata_rows),
                call_records=len(call_records),
                sidecar_rows=provider_metadata_row_count(con),
                inserted=0,
                updated=0,
                validation_ok=False,
                blocked=int(preflight["blocked"]),
                warnings=int(preflight["warnings"]),
            )
            return {
                "summary": summary.to_json_dict(),
                "audit": preflight,
                "items": [],
            }

        replaced_existing_table = create_provider_metadata_schema(con, replace_existing=replace_existing)
        existing_filenames = existing_provider_metadata_filenames(con)
        items = build_provider_metadata_items(metadata_rows=metadata_rows, call_records=call_records)
        inserted = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()
        for item in items:
            if item["source_filename"] in existing_filenames:
                update_provider_metadata_item(con, item=item, updated_at=now)
                updated += 1
            else:
                insert_provider_metadata_item(con, item=item, created_at=now, updated_at=now)
                inserted += 1
        con.commit()
        audit = audit_provider_metadata_sidecar(
            con=con,
            metadata_rows=metadata_rows,
            call_records=call_records,
        )
        blocked = int(audit["blocked"])
        warnings = int(audit["warnings"])
        validation_ok = blocked == 0 and int(audit["sidecar_rows"]) == len(metadata_rows)
        summary = ProviderMetadataSummary(
            schema_version=PROVIDER_METADATA_SCHEMA_VERSION,
            db_path=str(db_path),
            metadata_csv_path=str(metadata_csv_path),
            table_name=PROVIDER_METADATA_TABLE,
            replaced_existing_table=replaced_existing_table,
            metadata_rows=len(metadata_rows),
            call_records=len(call_records),
            sidecar_rows=int(audit["sidecar_rows"]),
            inserted=inserted,
            updated=updated,
            validation_ok=validation_ok,
            blocked=blocked,
            warnings=warnings,
        )
        return {
            "summary": summary.to_json_dict(),
            "audit": audit,
            "items": items[:100],
        }


def guard_provider_metadata_paths(
    db_path: Path,
    metadata_csv_path: Path,
    out_allowed_root: Path,
) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing to use runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts:
        raise ValueError("refusing to write provider metadata under stable_runtime")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"provider metadata DB must stay under allowed root: {out_allowed_root}")
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"disposable DB not found: {db_path}")
    if not metadata_csv_path.exists() or not metadata_csv_path.is_file():
        raise FileNotFoundError(f"metadata csv not found: {metadata_csv_path}")


def create_provider_metadata_schema(con: sqlite3.Connection, replace_existing: bool = False) -> bool:
    if replace_existing:
        con.execute(f"DROP TABLE IF EXISTS {PROVIDER_METADATA_TABLE}")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PROVIDER_METADATA_TABLE} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          call_record_id INTEGER NOT NULL,
          source_filename TEXT NOT NULL,
          source_file TEXT NOT NULL,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          provider_call_id TEXT NOT NULL,
          recording_id TEXT NOT NULL,
          event_key TEXT NOT NULL,
          checksum_sha256 TEXT NOT NULL,
          source_size_bytes INTEGER NOT NULL,
          manager_extension TEXT NOT NULL,
          raw_payload_ref TEXT,
          target_audio_path TEXT,
          source_audio_path TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(source_filename),
          UNIQUE(tenant_id, provider, provider_call_id),
          UNIQUE(tenant_id, provider, recording_id),
          UNIQUE(tenant_id, provider, event_key),
          FOREIGN KEY(call_record_id) REFERENCES call_records(id)
        )
        """
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{PROVIDER_METADATA_TABLE}_call_record_id "
        f"ON {PROVIDER_METADATA_TABLE} (call_record_id)"
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{PROVIDER_METADATA_TABLE}_manager_extension "
        f"ON {PROVIDER_METADATA_TABLE} (manager_extension)"
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{PROVIDER_METADATA_TABLE}_tenant_provider "
        f"ON {PROVIDER_METADATA_TABLE} (tenant_id, provider)"
    )
    return replace_existing


def preflight_provider_metadata(
    metadata_rows: Sequence[Mapping[str, str]],
    call_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    filenames = [filename_from_metadata(row) for row in metadata_rows]
    required_gaps = required_metadata_gaps(metadata_rows)
    missing_call_records = sorted(name for name in filenames if name and name not in call_records)
    duplicate_source_filenames = duplicate_counts(filenames)
    duplicate_provider_call_keys = duplicate_counts(provider_key(row, "provider_call_id") for row in metadata_rows)
    duplicate_recording_keys = duplicate_counts(provider_key(row, "recording_id") for row in metadata_rows)
    duplicate_event_keys = duplicate_counts(provider_key(row, "event_key") for row in metadata_rows)
    invalid_source_size = [
        filename_from_metadata(row)
        for row in metadata_rows
        if clean(row.get("source_size_bytes")) and optional_int(row.get("source_size_bytes")) is None
    ]
    blocked_reasons = {
        "required_metadata_gaps": len(required_gaps),
        "missing_call_records": len(missing_call_records),
        "duplicate_source_filenames": len(duplicate_source_filenames),
        "duplicate_provider_call_keys": len(duplicate_provider_call_keys),
        "duplicate_recording_keys": len(duplicate_recording_keys),
        "duplicate_event_keys": len(duplicate_event_keys),
        "invalid_source_size": len(invalid_source_size),
    }
    raw_payload_ref_missing = sum(1 for row in metadata_rows if not clean(row.get("raw_payload_ref")))
    return {
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "warnings": raw_payload_ref_missing,
        "warning_reasons": {
            "raw_payload_ref_missing": raw_payload_ref_missing,
        },
        "metadata_rows": len(metadata_rows),
        "call_records": len(call_records),
        "required_metadata_gaps": required_gaps[:100],
        "missing_call_records": missing_call_records[:100],
        "duplicate_source_filenames": duplicate_source_filenames,
        "duplicate_provider_call_keys": duplicate_provider_call_keys,
        "duplicate_recording_keys": duplicate_recording_keys,
        "duplicate_event_keys": duplicate_event_keys,
        "invalid_source_size": invalid_source_size[:100],
    }


def audit_provider_metadata_sidecar(
    con: sqlite3.Connection,
    metadata_rows: Sequence[Mapping[str, str]],
    call_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    sidecar_rows = read_provider_metadata_rows(con)
    sidecar_by_filename = {clean(row.get("source_filename")): row for row in sidecar_rows}
    metadata_filenames = [filename_from_metadata(row) for row in metadata_rows]
    sidecar_filenames = [clean(row.get("source_filename")) for row in sidecar_rows]
    missing_sidecar_for_metadata = sorted(name for name in metadata_filenames if name and name not in sidecar_by_filename)
    extra_sidecar_not_in_metadata = sorted(name for name in sidecar_filenames if name not in set(metadata_filenames))
    orphan_sidecar_rows = [
        clean(row.get("source_filename"))
        for row in sidecar_rows
        if clean(row.get("source_filename")) not in call_records
    ]
    mismatches = sidecar_mismatches(metadata_rows=metadata_rows, sidecar_by_filename=sidecar_by_filename)
    blocked_reasons = {
        "missing_sidecar_for_metadata": len(missing_sidecar_for_metadata),
        "extra_sidecar_not_in_metadata": len(extra_sidecar_not_in_metadata),
        "orphan_sidecar_rows": len(orphan_sidecar_rows),
        "provider_call_id_mismatches": len(mismatches["provider_call_id_mismatches"]),
        "recording_id_mismatches": len(mismatches["recording_id_mismatches"]),
        "event_key_mismatches": len(mismatches["event_key_mismatches"]),
        "checksum_mismatches": len(mismatches["checksum_mismatches"]),
        "source_size_mismatches": len(mismatches["source_size_mismatches"]),
        "manager_extension_mismatches": len(mismatches["manager_extension_mismatches"]),
    }
    raw_payload_ref_missing = count_missing(sidecar_rows, "raw_payload_ref")
    known_gaps = [
        "manager_extension is not mapped to a human CRM/telephony user yet",
    ]
    if raw_payload_ref_missing:
        known_gaps.insert(0, "raw Mango stats payload is not archived per row yet")
    return {
        "table_name": PROVIDER_METADATA_TABLE,
        "metadata_rows": len(metadata_rows),
        "call_records": len(call_records),
        "sidecar_rows": len(sidecar_rows),
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "warnings": raw_payload_ref_missing,
        "warning_reasons": {
            "raw_payload_ref_missing": raw_payload_ref_missing,
        },
        "tenant_provider_counts": tenant_provider_counts(sidecar_rows),
        "manager_extension_counts": dict(Counter(clean(row.get("manager_extension")) for row in sidecar_rows).most_common()),
        "unique_provider_call_keys": len({provider_sidecar_key(row, "provider_call_id") for row in sidecar_rows}),
        "unique_recording_keys": len({provider_sidecar_key(row, "recording_id") for row in sidecar_rows}),
        "unique_event_keys": len({provider_sidecar_key(row, "event_key") for row in sidecar_rows}),
        "missing_sidecar_for_metadata": missing_sidecar_for_metadata[:100],
        "extra_sidecar_not_in_metadata": extra_sidecar_not_in_metadata[:100],
        "orphan_sidecar_rows": orphan_sidecar_rows[:100],
        "mismatches": mismatches,
        "known_gaps": known_gaps,
        "samples": {
            "sidecar_rows": sidecar_rows[:20],
        },
    }


def build_provider_metadata_items(
    metadata_rows: Sequence[Mapping[str, str]],
    call_records: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for row in metadata_rows:
        filename = filename_from_metadata(row)
        call_record = call_records[filename]
        items.append(
            {
                "call_record_id": int(call_record["id"]),
                "source_filename": filename,
                "source_file": clean(call_record["source_file"]),
                "tenant_id": clean(row.get("tenant_id")),
                "provider": clean(row.get("provider")) or provider_from_source(row),
                "provider_call_id": clean(row.get("provider_call_id")) or clean(row.get("call_id")),
                "recording_id": clean(row.get("recording_id")) or clean(row.get("record_id")),
                "event_key": clean(row.get("event_key")),
                "checksum_sha256": clean(row.get("checksum_sha256")),
                "source_size_bytes": optional_int(row.get("source_size_bytes")) or 0,
                "manager_extension": clean(row.get("manager")) or manager_extension_from_name(row.get("manager_name")),
                "raw_payload_ref": clean(row.get("raw_payload_ref")) or None,
                "target_audio_path": clean(row.get("target_audio_path")) or None,
                "source_audio_path": clean(row.get("source_audio_path")) or None,
            }
        )
    return items


def insert_provider_metadata_item(
    con: sqlite3.Connection,
    item: Mapping[str, Any],
    created_at: str,
    updated_at: str,
) -> None:
    con.execute(
        f"""
        INSERT INTO {PROVIDER_METADATA_TABLE} (
          call_record_id, source_filename, source_file, tenant_id, provider,
          provider_call_id, recording_id, event_key, checksum_sha256,
          source_size_bytes, manager_extension, raw_payload_ref,
          target_audio_path, source_audio_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        item_values(item, created_at=created_at, updated_at=updated_at),
    )


def update_provider_metadata_item(
    con: sqlite3.Connection,
    item: Mapping[str, Any],
    updated_at: str,
) -> None:
    con.execute(
        f"""
        UPDATE {PROVIDER_METADATA_TABLE}
           SET call_record_id = ?,
               source_file = ?,
               tenant_id = ?,
               provider = ?,
               provider_call_id = ?,
               recording_id = ?,
               event_key = ?,
               checksum_sha256 = ?,
               source_size_bytes = ?,
               manager_extension = ?,
               raw_payload_ref = COALESCE(?, raw_payload_ref),
               target_audio_path = ?,
               source_audio_path = ?,
               updated_at = ?
         WHERE source_filename = ?
        """,
        (
            int(item["call_record_id"]),
            clean(item.get("source_file")),
            clean(item.get("tenant_id")),
            clean(item.get("provider")),
            clean(item.get("provider_call_id")),
            clean(item.get("recording_id")),
            clean(item.get("event_key")),
            clean(item.get("checksum_sha256")),
            int(item.get("source_size_bytes") or 0),
            clean(item.get("manager_extension")),
            item.get("raw_payload_ref"),
            item.get("target_audio_path"),
            item.get("source_audio_path"),
            updated_at,
            clean(item.get("source_filename")),
        ),
    )


def item_values(item: Mapping[str, Any], created_at: str, updated_at: str) -> tuple[Any, ...]:
    return (
        int(item["call_record_id"]),
        clean(item.get("source_filename")),
        clean(item.get("source_file")),
        clean(item.get("tenant_id")),
        clean(item.get("provider")),
        clean(item.get("provider_call_id")),
        clean(item.get("recording_id")),
        clean(item.get("event_key")),
        clean(item.get("checksum_sha256")),
        int(item.get("source_size_bytes") or 0),
        clean(item.get("manager_extension")),
        item.get("raw_payload_ref"),
        item.get("target_audio_path"),
        item.get("source_audio_path"),
        created_at,
        updated_at,
    )


def read_call_record_index(con: sqlite3.Connection) -> Mapping[str, Mapping[str, Any]]:
    rows = con.execute("select id, source_filename, source_file from call_records order by id").fetchall()
    return {clean(row["source_filename"]): dict(row) for row in rows}


def read_provider_metadata_rows(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    if not table_exists(con, PROVIDER_METADATA_TABLE):
        return []
    rows = con.execute(
        f"""
        SELECT
          id,
          call_record_id,
          source_filename,
          source_file,
          tenant_id,
          provider,
          provider_call_id,
          recording_id,
          event_key,
          checksum_sha256,
          source_size_bytes,
          manager_extension,
          raw_payload_ref,
          target_audio_path,
          source_audio_path,
          created_at,
          updated_at
        FROM {PROVIDER_METADATA_TABLE}
        ORDER BY id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def existing_provider_metadata_filenames(con: sqlite3.Connection) -> set[str]:
    if not table_exists(con, PROVIDER_METADATA_TABLE):
        return set()
    rows = con.execute(f"select source_filename from {PROVIDER_METADATA_TABLE}").fetchall()
    return {clean(row["source_filename"]) for row in rows}


def provider_metadata_row_count(con: sqlite3.Connection) -> int:
    if not table_exists(con, PROVIDER_METADATA_TABLE):
        return 0
    return int(con.execute(f"select count(*) from {PROVIDER_METADATA_TABLE}").fetchone()[0])


def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def required_metadata_gaps(metadata_rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, Any]]:
    gaps = []
    for index, row in enumerate(metadata_rows, start=1):
        missing = [field for field in PROVIDER_METADATA_REQUIRED_FIELDS if not required_value(row, field)]
        if missing:
            gaps.append(
                {
                    "row": index,
                    "filename": filename_from_metadata(row),
                    "missing": missing,
                }
            )
    return gaps


def required_value(row: Mapping[str, str], field: str) -> str:
    if field == "provider":
        return clean(row.get("provider")) or provider_from_source(row)
    if field == "provider_call_id":
        return clean(row.get("provider_call_id")) or clean(row.get("call_id"))
    if field == "recording_id":
        return clean(row.get("recording_id")) or clean(row.get("record_id"))
    if field == "manager":
        return clean(row.get("manager")) or manager_extension_from_name(row.get("manager_name"))
    return clean(row.get(field))


def sidecar_mismatches(
    metadata_rows: Sequence[Mapping[str, str]],
    sidecar_by_filename: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    provider_call_id_mismatches = []
    recording_id_mismatches = []
    event_key_mismatches = []
    checksum_mismatches = []
    source_size_mismatches = []
    manager_extension_mismatches = []
    for row in metadata_rows:
        filename = filename_from_metadata(row)
        sidecar = sidecar_by_filename.get(filename)
        if not sidecar:
            continue
        compare_text(row, sidecar, "provider_call_id", provider_call_id_mismatches, fallback="call_id")
        compare_text(row, sidecar, "recording_id", recording_id_mismatches, fallback="record_id")
        compare_text(row, sidecar, "event_key", event_key_mismatches)
        compare_text(row, sidecar, "checksum_sha256", checksum_mismatches)
        expected_size = optional_int(row.get("source_size_bytes"))
        if expected_size is not None and int(sidecar.get("source_size_bytes") or 0) != expected_size:
            source_size_mismatches.append(
                {
                    "filename": filename,
                    "expected": expected_size,
                    "actual": int(sidecar.get("source_size_bytes") or 0),
                }
            )
        expected_manager = clean(row.get("manager")) or manager_extension_from_name(row.get("manager_name"))
        if expected_manager and clean(sidecar.get("manager_extension")) != expected_manager:
            manager_extension_mismatches.append(
                {
                    "filename": filename,
                    "expected": expected_manager,
                    "actual": clean(sidecar.get("manager_extension")),
                }
            )
    return {
        "provider_call_id_mismatches": provider_call_id_mismatches[:100],
        "recording_id_mismatches": recording_id_mismatches[:100],
        "event_key_mismatches": event_key_mismatches[:100],
        "checksum_mismatches": checksum_mismatches[:100],
        "source_size_mismatches": source_size_mismatches[:100],
        "manager_extension_mismatches": manager_extension_mismatches[:100],
    }


def compare_text(
    metadata_row: Mapping[str, str],
    sidecar_row: Mapping[str, Any],
    field: str,
    out: list[Mapping[str, Any]],
    fallback: str | None = None,
) -> None:
    expected = clean(metadata_row.get(field)) or (clean(metadata_row.get(fallback)) if fallback else "")
    actual = clean(sidecar_row.get(field))
    if expected and actual != expected:
        out.append(
            {
                "filename": filename_from_metadata(metadata_row),
                "field": field,
                "expected": expected,
                "actual": actual,
            }
        )


def filename_from_metadata(row: Mapping[str, str]) -> str:
    return Path(clean(row.get("filename"))).name


def provider_key(row: Mapping[str, str], field: str) -> tuple[str, str, str]:
    provider = clean(row.get("provider")) or provider_from_source(row)
    value = clean(row.get(field))
    if field == "provider_call_id":
        value = value or clean(row.get("call_id"))
    elif field == "recording_id":
        value = value or clean(row.get("record_id"))
    return (clean(row.get("tenant_id")), provider, value)


def provider_sidecar_key(row: Mapping[str, Any], field: str) -> tuple[str, str, str]:
    return (clean(row.get("tenant_id")), clean(row.get("provider")), clean(row.get(field)))


def duplicate_counts(values: Sequence[Any]) -> Mapping[str, int]:
    counts = Counter(value for value in values if value and not empty_key(value))
    return {stringify_key(key): count for key, count in sorted(counts.items()) if count > 1}


def empty_key(value: Any) -> bool:
    if isinstance(value, tuple):
        return any(not clean(item) for item in value)
    return not clean(value)


def stringify_key(value: Any) -> str:
    if isinstance(value, tuple):
        return "|".join(clean(part) for part in value)
    return clean(value)


def provider_from_source(row: Mapping[str, str]) -> str:
    source = clean(row.get("source")).lower()
    if source.startswith("mango"):
        return "mango"
    return source or "unknown"


def manager_extension_from_name(value: Any) -> str:
    text = clean(value)
    if text.startswith("mango_"):
        return text.removeprefix("mango_")
    return text


def optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def tenant_provider_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = Counter(f"{clean(row.get('tenant_id'))}|{clean(row.get('provider'))}" for row in rows)
    return dict(sorted(counts.items()))


def count_missing(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(1 for row in rows if not clean(row.get(field)))
