from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.productization.provider_metadata import PROVIDER_METADATA_TABLE
from mango_mvp.productization.test_ingest import (
    RUNTIME_DB_FILENAMES,
    clean,
    path_is_relative_to,
    read_metadata_rows,
)


PAYLOAD_ARCHIVE_SCHEMA_VERSION = "mango_payload_archive_v1"
SHADOW_POLL_RAW_PAYLOAD_SCHEMA_VERSION = "mango_shadow_poll_raw_payload_v1"


@dataclass(frozen=True)
class PayloadArchiveSummary:
    schema_version: str
    db_path: str
    metadata_csv_path: str
    source_payload_path: str
    archive_root: str
    metadata_rows: int
    source_payload_rows: int
    archived_entries: int
    archive_files: int
    sidecar_rows: int
    sidecar_refs_present: int
    sidecar_refs_updated: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def archive_mango_payloads_and_update_sidecar(
    db_path: Path,
    metadata_csv_path: Path,
    source_payload_path: Path,
    archive_root: Path,
    out_allowed_root: Path,
    replace_existing: bool = False,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    metadata_csv_path = metadata_csv_path.resolve(strict=False)
    source_payload_path = source_payload_path.resolve(strict=False)
    archive_root = archive_root.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    guard_payload_archive_paths(
        db_path=db_path,
        metadata_csv_path=metadata_csv_path,
        source_payload_path=source_payload_path,
        archive_root=archive_root,
        out_allowed_root=out_allowed_root,
    )

    metadata_rows = read_metadata_rows(metadata_csv_path)
    source_rows = load_source_payload_rows(source_payload_path)
    source_index = index_source_payload_rows(source_rows)
    archive_entries, build_audit = build_payload_archive_entries(
        metadata_rows=metadata_rows,
        source_index=source_index,
        archive_root=archive_root,
        out_allowed_root=out_allowed_root,
    )
    if replace_existing and archive_root.exists():
        shutil.rmtree(archive_root)
    archive_files = write_payload_archive_entries(archive_entries)

    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        sidecar_refs_updated = update_sidecar_raw_payload_refs(con, archive_entries)
        con.commit()
        audit = audit_payload_archive(
            con=con,
            archive_entries=archive_entries,
            archive_root=archive_root,
            out_allowed_root=out_allowed_root,
            build_audit=build_audit,
        )

    blocked = int(audit["blocked"])
    warnings = int(audit["warnings"])
    validation_ok = (
        blocked == 0
        and int(audit["archived_entries"]) == len(metadata_rows)
        and int(audit["sidecar_refs_present"]) == int(audit["sidecar_rows"])
    )
    summary = PayloadArchiveSummary(
        schema_version=PAYLOAD_ARCHIVE_SCHEMA_VERSION,
        db_path=str(db_path),
        metadata_csv_path=str(metadata_csv_path),
        source_payload_path=str(source_payload_path),
        archive_root=str(archive_root),
        metadata_rows=len(metadata_rows),
        source_payload_rows=len(source_rows),
        archived_entries=len(archive_entries),
        archive_files=archive_files,
        sidecar_rows=int(audit["sidecar_rows"]),
        sidecar_refs_present=int(audit["sidecar_refs_present"]),
        sidecar_refs_updated=sidecar_refs_updated,
        validation_ok=validation_ok,
        blocked=blocked,
        warnings=warnings,
    )
    return {
        "summary": summary.to_json_dict(),
        "audit": audit,
        "items": [sample_archive_entry(entry) for entry in archive_entries[:100]],
    }


def guard_payload_archive_paths(
    db_path: Path,
    metadata_csv_path: Path,
    source_payload_path: Path,
    archive_root: Path,
    out_allowed_root: Path,
) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing to use runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts or "stable_runtime" in archive_root.parts:
        raise ValueError("refusing to write payload archive under stable_runtime")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"DB must stay under allowed root: {out_allowed_root}")
    if not path_is_relative_to(archive_root, out_allowed_root):
        raise ValueError(f"archive root must stay under allowed root: {out_allowed_root}")
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"disposable DB not found: {db_path}")
    if not metadata_csv_path.exists() or not metadata_csv_path.is_file():
        raise FileNotFoundError(f"metadata csv not found: {metadata_csv_path}")
    if not source_payload_path.exists() or not source_payload_path.is_file():
        raise FileNotFoundError(f"source payload file not found: {source_payload_path}")


def load_source_payload_rows(source_payload_path: Path) -> list[Mapping[str, Any]]:
    if source_payload_path.suffix.lower() == ".jsonl":
        return load_source_payload_jsonl(source_payload_path)
    payload = json.loads(source_payload_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = []
        for key in ("missing", "matched", "rows", "items", "decisions"):
            values = payload.get(key)
            if isinstance(values, list):
                for index, row in enumerate(values, start=1):
                    raw_payload = row.get("raw_payload") if isinstance(row, Mapping) else None
                    rows.append(
                        source_payload_record(
                            raw_payload=raw_payload if isinstance(raw_payload, Mapping) else row,
                            source_payload_path=source_payload_path,
                            source_line=None,
                            source_index=index,
                            source_kind=f"json_report:{key}",
                            source_entry=row if isinstance(row, Mapping) else {},
                        )
                    )
        return rows
    if isinstance(payload, list):
        return [
            source_payload_record(
                raw_payload=row,
                source_payload_path=source_payload_path,
                source_line=None,
                source_index=index,
                source_kind="json_list",
                source_entry=row if isinstance(row, Mapping) else {},
            )
            for index, row in enumerate(payload, start=1)
            if isinstance(row, Mapping)
        ]
    raise ValueError(f"unsupported source payload structure: {source_payload_path}")


def load_source_payload_jsonl(source_payload_path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with source_payload_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            entry = json.loads(text)
            if not isinstance(entry, Mapping):
                continue
            raw_payload = entry.get("raw_payload")
            if not isinstance(raw_payload, Mapping):
                raw_payload = entry
            rows.append(
                source_payload_record(
                    raw_payload=raw_payload,
                    source_payload_path=source_payload_path,
                    source_line=line_number,
                    source_index=line_number,
                    source_kind=clean(entry.get("schema_version")) or "jsonl",
                    source_entry=entry,
                )
            )
    return rows


def source_payload_record(
    raw_payload: Mapping[str, Any],
    source_payload_path: Path,
    source_line: int | None,
    source_index: int,
    source_kind: str,
    source_entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    provider_call_id = source_provider_call_id(raw_payload, source_entry)
    event_key = clean(source_entry.get("event_key")) or clean(raw_payload.get("event_key"))
    return {
        "provider_call_id": provider_call_id,
        "event_key": event_key,
        "recording_id": source_recording_id(raw_payload, source_entry),
        "source_payload_ref": source_payload_ref(source_payload_path, source_line, source_index),
        "source_kind": source_kind,
        "raw_payload": dict(raw_payload),
        "source_entry": dict(source_entry),
    }


def index_source_payload_rows(source_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        for key in (
            clean(row.get("event_key")),
            clean(row.get("provider_call_id")),
            clean(row.get("recording_id")),
        ):
            if key and key not in index:
                index[key] = row
    return index


def build_payload_archive_entries(
    metadata_rows: Sequence[Mapping[str, str]],
    source_index: Mapping[str, Mapping[str, Any]],
    archive_root: Path,
    out_allowed_root: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    entries = []
    missing_source_payloads = []
    for row in metadata_rows:
        source = match_source_payload(row, source_index)
        if not source:
            missing_source_payloads.append(metadata_identity(row))
            continue
        entry = build_payload_archive_entry(
            metadata_row=row,
            source_payload=source,
            archive_root=archive_root,
            out_allowed_root=out_allowed_root,
        )
        entries.append(entry)
    return entries, {
        "missing_source_payloads": missing_source_payloads[:100],
        "missing_source_payload_count": len(missing_source_payloads),
    }


def build_payload_archive_entry(
    metadata_row: Mapping[str, str],
    source_payload: Mapping[str, Any],
    archive_root: Path,
    out_allowed_root: Path,
) -> Mapping[str, Any]:
    tenant_id = clean(metadata_row.get("tenant_id"))
    provider = clean(metadata_row.get("provider"))
    started_at = clean(metadata_row.get("started_at")) or clean(metadata_row.get("start_time"))
    archive_date = started_at[:10] if started_at else "unknown"
    archive_file = archive_root / safe_part(f"tenant={tenant_id}") / safe_part(f"provider={provider}") / safe_part(f"date={archive_date}") / "payloads.jsonl"
    entry_hash = payload_entry_hash(metadata_row)
    raw_payload_ref = f"{relative_ref(archive_file, out_allowed_root)}#entry={entry_hash}"
    return {
        "schema_version": PAYLOAD_ARCHIVE_SCHEMA_VERSION,
        "entry_hash": entry_hash,
        "raw_payload_ref": raw_payload_ref,
        "archive_file": str(archive_file),
        "tenant_id": tenant_id,
        "provider": provider,
        "provider_call_id": clean(metadata_row.get("provider_call_id")) or clean(metadata_row.get("call_id")),
        "recording_id": clean(metadata_row.get("recording_id")) or clean(metadata_row.get("record_id")),
        "event_key": clean(metadata_row.get("event_key")),
        "source_filename": Path(clean(metadata_row.get("filename"))).name,
        "source_payload_ref": source_payload.get("source_payload_ref"),
        "source_kind": source_payload.get("source_kind"),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "raw_payload": source_payload.get("raw_payload") or {},
        "metadata_row": dict(metadata_row),
    }


def write_payload_archive_entries(entries: Sequence[Mapping[str, Any]]) -> int:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["archive_file"])].append(entry)
    for archive_file, group in grouped.items():
        path = Path(archive_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for entry in sorted(group, key=lambda item: clean(item.get("event_key"))):
                fh.write(json.dumps(strip_internal_archive_fields(entry), ensure_ascii=False, sort_keys=True) + "\n")
    return len(grouped)


def strip_internal_archive_fields(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in entry.items() if key != "archive_file"}


def update_sidecar_raw_payload_refs(
    con: sqlite3.Connection,
    entries: Sequence[Mapping[str, Any]],
) -> int:
    updated = 0
    for entry in entries:
        cursor = con.execute(
            f"""
            UPDATE {PROVIDER_METADATA_TABLE}
               SET raw_payload_ref = ?,
                   updated_at = ?
             WHERE tenant_id = ?
               AND provider = ?
               AND event_key = ?
            """,
            (
                clean(entry.get("raw_payload_ref")),
                datetime.now(timezone.utc).isoformat(),
                clean(entry.get("tenant_id")),
                clean(entry.get("provider")),
                clean(entry.get("event_key")),
            ),
        )
        updated += cursor.rowcount
    return updated


def audit_payload_archive(
    con: sqlite3.Connection,
    archive_entries: Sequence[Mapping[str, Any]],
    archive_root: Path,
    out_allowed_root: Path,
    build_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    sidecar_rows = read_sidecar_rows(con)
    refs = [clean(row.get("raw_payload_ref")) for row in sidecar_rows if clean(row.get("raw_payload_ref"))]
    entry_refs = {clean(entry.get("raw_payload_ref")) for entry in archive_entries}
    sidecar_refs_missing = [clean(row.get("source_filename")) for row in sidecar_rows if not clean(row.get("raw_payload_ref"))]
    sidecar_refs_not_archived = sorted(ref for ref in refs if ref not in entry_refs)
    archive_refs_not_in_sidecar = sorted(ref for ref in entry_refs if ref not in set(refs))
    missing_archive_files = sorted(ref for ref in entry_refs if not raw_payload_ref_file_exists(ref, out_allowed_root))
    archive_files = sorted(archive_root.rglob("*.jsonl")) if archive_root.exists() else []
    archive_file_rows = sum(1 for path in archive_files for _line in path.open("r", encoding="utf-8"))
    blocked_reasons = {
        "missing_source_payloads": int(build_audit.get("missing_source_payload_count") or 0),
        "sidecar_refs_missing": len(sidecar_refs_missing),
        "sidecar_refs_not_archived": len(sidecar_refs_not_archived),
        "archive_refs_not_in_sidecar": len(archive_refs_not_in_sidecar),
        "missing_archive_files": len(missing_archive_files),
        "archive_file_row_mismatch": 0 if archive_file_rows == len(archive_entries) else 1,
    }
    source_kind_counts = Counter(clean(entry.get("source_kind")) for entry in archive_entries)
    return {
        "archive_root": str(archive_root),
        "archived_entries": len(archive_entries),
        "archive_files": len(archive_files),
        "archive_file_rows": archive_file_rows,
        "sidecar_rows": len(sidecar_rows),
        "sidecar_refs_present": len(refs),
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "warnings": 0,
        "warning_reasons": {},
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "tenant_provider_counts": dict(sorted(Counter(f"{clean(entry.get('tenant_id'))}|{clean(entry.get('provider'))}" for entry in archive_entries).items())),
        "missing_source_payloads": build_audit.get("missing_source_payloads", []),
        "sidecar_refs_missing": sidecar_refs_missing[:100],
        "sidecar_refs_not_archived": sidecar_refs_not_archived[:100],
        "archive_refs_not_in_sidecar": archive_refs_not_in_sidecar[:100],
        "missing_archive_files": missing_archive_files[:100],
        "samples": {
            "archive_entries": [sample_archive_entry(entry) for entry in archive_entries[:20]],
        },
    }


def read_sidecar_rows(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    rows = con.execute(
        f"""
        SELECT source_filename, tenant_id, provider, provider_call_id,
               recording_id, event_key, raw_payload_ref
          FROM {PROVIDER_METADATA_TABLE}
         ORDER BY id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def write_shadow_poll_raw_payload_jsonl(
    rows: Iterable[Mapping[str, Any]],
    out_path: Path,
    tenant_id: str,
    provider: str,
    base_url: str,
    since: str,
    until: str,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, start=1):
            entry = {
                "schema_version": SHADOW_POLL_RAW_PAYLOAD_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "provider": provider,
                "base_url": base_url,
                "window": {"since": since, "until": until},
                "row_index": index,
                "provider_call_id": source_provider_call_id(row, {}),
                "recording_id": source_recording_id(row, {}),
                "event_key": None,
                "raw_payload": dict(row),
                "archived_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def match_source_payload(
    metadata_row: Mapping[str, str],
    source_index: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for key in (
        clean(metadata_row.get("event_key")),
        clean(metadata_row.get("provider_call_id")) or clean(metadata_row.get("call_id")),
        clean(metadata_row.get("recording_id")) or clean(metadata_row.get("record_id")),
    ):
        if key and key in source_index:
            return source_index[key]
    return None


def source_provider_call_id(raw_payload: Mapping[str, Any], source_entry: Mapping[str, Any]) -> str:
    return (
        clean(source_entry.get("provider_call_id"))
        or clean(raw_payload.get("provider_call_id"))
        or clean(raw_payload.get("entry_id"))
        or clean(raw_payload.get("call_id"))
        or clean(raw_payload.get("id"))
    )


def source_recording_id(raw_payload: Mapping[str, Any], source_entry: Mapping[str, Any]) -> str:
    return (
        clean(source_entry.get("recording_id"))
        or clean(source_entry.get("recording_ref"))
        or clean(raw_payload.get("recording_id"))
        or clean(raw_payload.get("record_id"))
        or clean(raw_payload.get("recording_ref"))
        or first_recording_from_records(raw_payload.get("records"))
    )


def first_recording_from_records(value: Any) -> str:
    if isinstance(value, list):
        return clean(value[0]) if value else ""
    text = clean(value)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
        return clean(text.split(",", 1)[0])
    return text


def source_payload_ref(source_payload_path: Path, source_line: int | None, source_index: int) -> str:
    if source_line is not None:
        return f"{source_payload_path}#line={source_line}"
    return f"{source_payload_path}#index={source_index}"


def payload_entry_hash(metadata_row: Mapping[str, str]) -> str:
    material = "|".join(
        (
            clean(metadata_row.get("tenant_id")),
            clean(metadata_row.get("provider")),
            clean(metadata_row.get("provider_call_id")) or clean(metadata_row.get("call_id")),
            clean(metadata_row.get("recording_id")) or clean(metadata_row.get("record_id")),
            clean(metadata_row.get("event_key")),
            Path(clean(metadata_row.get("filename"))).name,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def metadata_identity(row: Mapping[str, str]) -> Mapping[str, str]:
    return {
        "filename": Path(clean(row.get("filename"))).name,
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")) or clean(row.get("call_id")),
        "recording_id": clean(row.get("recording_id")) or clean(row.get("record_id")),
    }


def sample_archive_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "raw_payload_ref": entry.get("raw_payload_ref"),
        "source_payload_ref": entry.get("source_payload_ref"),
        "source_kind": entry.get("source_kind"),
        "tenant_id": entry.get("tenant_id"),
        "provider": entry.get("provider"),
        "provider_call_id": entry.get("provider_call_id"),
        "recording_id": entry.get("recording_id"),
        "event_key": entry.get("event_key"),
        "source_filename": entry.get("source_filename"),
    }


def raw_payload_ref_file_exists(raw_payload_ref: str, out_allowed_root: Path) -> bool:
    path_part = raw_payload_ref.split("#", 1)[0]
    path = Path(path_part)
    if not path.is_absolute():
        path = out_allowed_root / path
    return path.exists() and path.is_file()


def relative_ref(path: Path, out_allowed_root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(out_allowed_root.resolve(strict=False)))
    except ValueError:
        return str(path)


def safe_part(value: str) -> str:
    text = clean(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "=", "."} else "_" for ch in text) or "unknown"
