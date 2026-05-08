from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


RECORDING_ASSET_INGEST_SCHEMA_VERSION = "recording_asset_ingest_v1"
RECORDING_ASSET_INGEST_MIGRATION_ID = "20260507_001_recording_asset_ingest"
SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm")
SAFE_DB_SUFFIXES = {".sqlite", ".db"}
READY_STATUS = "quarantined_ready"


@dataclass(frozen=True)
class RecordingAssetIngestSummary:
    schema_version: str
    product_root: str
    package_root: str
    package_ref: str
    db_path: str
    metadata_csv_path: str
    audio_dir: str
    dry_run: bool
    replaced_existing_db: bool
    metadata_rows: int
    planned_assets: int
    inserted: int
    already_present: int
    updated: int
    blocked: int
    warnings: int
    db_assets_for_package: int
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def run_recording_asset_ingest(
    package_root: Path,
    audio_dir: Path,
    metadata_csv_path: Path,
    db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
    package_ref: Optional[str] = None,
    replace_existing_db: bool = False,
    allow_existing_db: bool = False,
    dry_run: bool = False,
    verify_checksum: bool = True,
    limit: Optional[int] = None,
) -> Mapping[str, Any]:
    paths = resolve_ingest_paths(
        product_root=product_root,
        package_root=package_root,
        audio_dir=audio_dir,
        metadata_csv_path=metadata_csv_path,
        db_path=db_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    package_root = paths["package_root"]
    audio_dir = paths["audio_dir"]
    metadata_csv_path = paths["metadata_csv_path"]
    db_path = paths["db_path"]
    out_path = paths.get("out_path")
    package_ref = clean(package_ref) or package_root.name

    if replace_existing_db and allow_existing_db:
        raise ValueError("replace_existing_db and allow_existing_db are mutually exclusive")

    metadata_rows = read_metadata_rows(metadata_csv_path, limit=limit)
    items = build_ingest_items(
        metadata_rows=metadata_rows,
        product_root=product_root,
        audio_dir=audio_dir,
        verify_checksum=verify_checksum,
    )
    items = apply_duplicate_blocks(items)
    planned_assets = sum(1 for item in items if item["action"] == "PLAN_RECORDING_ASSET_INGEST")
    blocked_before_db = sum(1 for item in items if item["action"].startswith("BLOCK_"))

    replaced = False
    db_audit: Mapping[str, Any] = {
        "db_path": str(db_path),
        "assets_for_package": 0,
        "status_counts": {},
        "blocked": 0,
        "blocked_reasons": {},
    }
    if not dry_run:
        if replace_existing_db:
            replaced = remove_sqlite_db(db_path, product_root)
        elif db_path.exists() and not allow_existing_db:
            raise FileExistsError(f"recording asset ingest DB already exists: {db_path}")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())
        now = now_utc()
        with sqlite3.connect(str(db_path)) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            create_recording_asset_ingest_schema(con)
            apply_recording_asset_ingest_migration(con)
            upsert_import_package(
                con,
                package_ref=package_ref,
                package_root=package_root,
                metadata_csv_path=metadata_csv_path,
                audio_dir=audio_dir,
                metadata_rows=len(metadata_rows),
                package_hash=sha256_file(metadata_csv_path),
                run_id=run_id,
                now=now,
            )
            items = upsert_recording_assets(
                con,
                items=items,
                package_ref=package_ref,
                run_id=run_id,
                now=now,
            )
            counts = action_counts(items)
            blocked_after_db = sum(1 for item in items if item["action"].startswith("BLOCK_"))
            insert_ingest_run(
                con,
                run_id=run_id,
                package_ref=package_ref,
                started_at=now,
                finished_at=now_utc(),
                metadata_rows=len(metadata_rows),
                planned_assets=planned_assets,
                inserted=int(counts.get("INGEST_RECORDING_ASSET") or 0),
                already_present=int(counts.get("SKIP_ALREADY_INGESTED") or 0),
                updated=int(counts.get("UPDATE_RECORDING_ASSET") or 0),
                blocked=blocked_after_db,
                warnings=count_item_warnings(items),
                dry_run=False,
                audit_json="{}",
            )
            db_audit = audit_recording_asset_ingest_db(
                con,
                product_root=product_root,
                audio_dir=audio_dir,
                package_ref=package_ref,
                expected_items=items,
            )
            con.execute(
                """
                UPDATE recording_asset_ingest_runs
                   SET audit_json = ?
                 WHERE run_id = ?
                """,
                (json.dumps({"action_counts": counts, "db_audit": db_audit}, ensure_ascii=False), run_id),
            )
            con.commit()

    counts = action_counts(items)
    blocked = sum(1 for item in items if item["action"].startswith("BLOCK_"))
    warnings = count_item_warnings(items)
    db_assets_for_package = int(db_audit.get("assets_for_package") or 0)
    validation_ok = blocked == 0 and (dry_run or db_assets_for_package == planned_assets)
    summary = RecordingAssetIngestSummary(
        schema_version=RECORDING_ASSET_INGEST_SCHEMA_VERSION,
        product_root=str(product_root),
        package_root=str(package_root),
        package_ref=package_ref,
        db_path=str(db_path),
        metadata_csv_path=str(metadata_csv_path),
        audio_dir=str(audio_dir),
        dry_run=dry_run,
        replaced_existing_db=replaced,
        metadata_rows=len(metadata_rows),
        planned_assets=planned_assets,
        inserted=int(counts.get("INGEST_RECORDING_ASSET") or 0),
        already_present=int(counts.get("SKIP_ALREADY_INGESTED") or 0),
        updated=int(counts.get("UPDATE_RECORDING_ASSET") or 0),
        blocked=blocked,
        warnings=warnings,
        db_assets_for_package=db_assets_for_package,
        validation_ok=validation_ok,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": counts,
        "db_audit": db_audit,
        "items": items,
        "safety": {
            "isolated_productization_db_writes": bool(not dry_run),
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "downloads_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_tallanto": False,
        },
        "known_scope": {
            "processing_queue_created": False,
            "legacy_call_records_used": False,
            "product_calls_updated": False,
        },
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_ingest_items(
    metadata_rows: Sequence[Mapping[str, str]],
    product_root: Path,
    audio_dir: Path,
    verify_checksum: bool,
) -> list[dict[str, Any]]:
    return [
        build_ingest_item(row, row_number=index, product_root=product_root, audio_dir=audio_dir, verify_checksum=verify_checksum)
        for index, row in enumerate(metadata_rows, start=2)
    ]


def build_ingest_item(
    row: Mapping[str, str],
    row_number: int,
    product_root: Path,
    audio_dir: Path,
    verify_checksum: bool,
) -> dict[str, Any]:
    filename = Path(clean(row.get("filename"))).name
    target_audio_path = Path(clean(row.get("target_audio_path")) or str(audio_dir / filename)).resolve(strict=False)
    source_audio_path = Path(clean(row.get("source_audio_path"))).resolve(strict=False) if clean(row.get("source_audio_path")) else None
    checksum = clean(row.get("checksum_sha256")).lower()
    item: dict[str, Any] = {
        "action": "PLAN_RECORDING_ASSET_INGEST",
        "reason": "ready_for_isolated_recording_asset_ingest",
        "row_number": row_number,
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")) or "mango",
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")) or clean(row.get("call_id")),
        "recording_id": clean(row.get("recording_id")) or clean(row.get("record_id")),
        "filename": filename,
        "target_audio_path": str(target_audio_path),
        "source_audio_path": str(source_audio_path) if source_audio_path else None,
        "checksum_sha256": checksum,
        "size_bytes": optional_int(row.get("source_size_bytes")),
        "duration_sec": optional_float(row.get("duration_sec")),
        "started_at": clean(row.get("started_at")) or clean(row.get("start_time")) or None,
        "direction": clean(row.get("direction")) or None,
        "client_phone": clean(row.get("client_phone")) or clean(row.get("phone")) or None,
        "manager_ref": clean(row.get("manager")) or None,
        "manager_name": clean(row.get("manager_name")) or None,
        "source": clean(row.get("source")) or "mango_api_capture",
        "metadata": dict(row),
        "warnings": [],
        "blocked_reasons": [],
    }

    required_fields = ("tenant_id", "provider", "event_key", "provider_call_id", "recording_id", "filename")
    for field in required_fields:
        if not clean(item.get(field)):
            item["blocked_reasons"].append(f"missing_{field}")
    if not checksum:
        item["blocked_reasons"].append("missing_checksum_sha256")
    if item["duration_sec"] is None:
        item["warnings"].append("duration_sec_missing")
    if not item["manager_ref"]:
        item["warnings"].append("manager_ref_missing")

    path_block_reasons = validate_audio_paths(
        target_audio_path=target_audio_path,
        source_audio_path=source_audio_path,
        product_root=product_root,
        audio_dir=audio_dir,
    )
    item["blocked_reasons"].extend(path_block_reasons)

    if not path_block_reasons and target_audio_path.exists() and target_audio_path.is_file():
        actual_size = target_audio_path.stat().st_size
        item["actual_size_bytes"] = actual_size
        if actual_size <= 0:
            item["blocked_reasons"].append("zero_size_audio")
        expected_size = item["size_bytes"]
        if expected_size is not None and expected_size != actual_size:
            item["blocked_reasons"].append("size_bytes_mismatch")
        if verify_checksum:
            actual_checksum = sha256_file(target_audio_path)
            item["actual_checksum_sha256"] = actual_checksum
            if checksum and checksum != actual_checksum:
                item["blocked_reasons"].append("checksum_sha256_mismatch")
    else:
        item["actual_size_bytes"] = None

    if item["blocked_reasons"]:
        item["action"] = "BLOCK_RECORDING_ASSET_INGEST"
        item["reason"] = ",".join(sorted(set(item["blocked_reasons"])))
    return item


def validate_audio_paths(
    target_audio_path: Path,
    source_audio_path: Optional[Path],
    product_root: Path,
    audio_dir: Path,
) -> list[str]:
    blocked: list[str] = []
    if "stable_runtime" in target_audio_path.parts:
        blocked.append("target_audio_under_stable_runtime")
    if not path_is_relative_to(target_audio_path, product_root):
        blocked.append("target_audio_outside_product_root")
    if not path_is_relative_to(target_audio_path, audio_dir):
        blocked.append("target_audio_outside_audio_dir")
    if target_audio_path.suffix.lower() != ".mp3":
        blocked.append("unsupported_audio_extension")
    if not target_audio_path.exists():
        blocked.append("target_audio_missing")
    elif not target_audio_path.is_file():
        blocked.append("target_audio_not_file")

    if source_audio_path:
        if "stable_runtime" in source_audio_path.parts:
            blocked.append("source_audio_under_stable_runtime")
        if not path_is_relative_to(source_audio_path, product_root):
            blocked.append("source_audio_outside_product_root")
    return blocked


def apply_duplicate_blocks(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    event_counts = Counter(clean(item.get("event_key")) for item in items if clean(item.get("event_key")))
    recording_counts = Counter(
        f"{clean(item.get('tenant_id'))}:{clean(item.get('provider'))}:{clean(item.get('recording_id'))}"
        for item in items
        if clean(item.get("tenant_id")) and clean(item.get("provider")) and clean(item.get("recording_id"))
    )
    result: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if event_counts.get(clean(item.get("event_key")), 0) > 1:
            blocked.append("duplicate_event_key_in_metadata")
        recording_key = f"{clean(item.get('tenant_id'))}:{clean(item.get('provider'))}:{clean(item.get('recording_id'))}"
        if recording_counts.get(recording_key, 0) > 1:
            blocked.append("duplicate_recording_id_in_metadata")
        if blocked:
            item["blocked_reasons"] = sorted(set(blocked))
            item["action"] = "BLOCK_RECORDING_ASSET_INGEST"
            item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def upsert_recording_assets(
    con: sqlite3.Connection,
    items: Sequence[Mapping[str, Any]],
    package_ref: str,
    run_id: str,
    now: str,
) -> list[dict[str, Any]]:
    updated_items: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item["action"].startswith("BLOCK_"):
            updated_items.append(item)
            continue
        conflict = find_asset_conflict(con, item)
        if conflict:
            item["action"] = "BLOCK_RECORDING_ASSET_INGEST"
            item["reason"] = "recording_asset_conflicts_with_existing_row"
            item["blocked_reasons"] = ["recording_asset_conflicts_with_existing_row"]
            item["conflict"] = conflict
            updated_items.append(item)
            continue

        existing = find_existing_asset(con, item)
        if existing is None:
            insert_recording_asset(con, item, package_ref=package_ref, run_id=run_id, now=now)
            item["action"] = "INGEST_RECORDING_ASSET"
            item["reason"] = "inserted_recording_asset"
        elif asset_matches(existing, item, package_ref):
            item["action"] = "SKIP_ALREADY_INGESTED"
            item["reason"] = "recording_asset_already_present"
            item["asset_id"] = int(existing["id"])
        else:
            update_recording_asset(con, item, asset_id=int(existing["id"]), package_ref=package_ref, run_id=run_id, now=now)
            item["action"] = "UPDATE_RECORDING_ASSET"
            item["reason"] = "updated_recording_asset_metadata"
            item["asset_id"] = int(existing["id"])
        updated_items.append(item)
    return updated_items


def create_recording_asset_ingest_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS recording_asset_schema_migrations (
          migration_id TEXT PRIMARY KEY,
          schema_version TEXT NOT NULL,
          applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS recording_import_packages (
          package_ref TEXT PRIMARY KEY,
          package_root TEXT NOT NULL,
          metadata_csv_path TEXT NOT NULL,
          audio_dir TEXT NOT NULL,
          metadata_rows INTEGER NOT NULL,
          package_hash TEXT NOT NULL,
          first_ingested_at TEXT NOT NULL,
          last_ingested_at TEXT NOT NULL,
          last_run_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS captured_recording_assets (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          event_key TEXT NOT NULL,
          provider_call_id TEXT NOT NULL,
          recording_id TEXT NOT NULL,
          source TEXT,
          source_audio_path TEXT,
          audio_path TEXT NOT NULL,
          source_filename TEXT NOT NULL,
          checksum_sha256 TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          duration_sec REAL,
          started_at TEXT,
          direction TEXT,
          client_phone TEXT,
          manager_ref TEXT,
          manager_name TEXT,
          status TEXT NOT NULL,
          package_ref TEXT NOT NULL,
          last_run_id TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          first_ingested_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(tenant_id, provider, event_key),
          UNIQUE(tenant_id, provider, recording_id),
          FOREIGN KEY(package_ref) REFERENCES recording_import_packages(package_ref)
        );

        CREATE TABLE IF NOT EXISTS recording_asset_ingest_runs (
          run_id TEXT PRIMARY KEY,
          package_ref TEXT NOT NULL,
          started_at TEXT NOT NULL,
          finished_at TEXT NOT NULL,
          metadata_rows INTEGER NOT NULL,
          planned_assets INTEGER NOT NULL,
          inserted INTEGER NOT NULL,
          already_present INTEGER NOT NULL,
          updated INTEGER NOT NULL,
          blocked INTEGER NOT NULL,
          warnings INTEGER NOT NULL,
          dry_run INTEGER NOT NULL,
          audit_json TEXT,
          FOREIGN KEY(package_ref) REFERENCES recording_import_packages(package_ref)
        );

        CREATE INDEX IF NOT EXISTS ix_captured_recording_assets_status
          ON captured_recording_assets(status, started_at);
        CREATE INDEX IF NOT EXISTS ix_captured_recording_assets_manager
          ON captured_recording_assets(tenant_id, provider, manager_ref, started_at);
        CREATE INDEX IF NOT EXISTS ix_captured_recording_assets_package
          ON captured_recording_assets(package_ref);
        """
    )


def apply_recording_asset_ingest_migration(con: sqlite3.Connection) -> int:
    row = con.execute(
        "SELECT 1 FROM recording_asset_schema_migrations WHERE migration_id = ?",
        (RECORDING_ASSET_INGEST_MIGRATION_ID,),
    ).fetchone()
    if row:
        return 0
    con.execute(
        """
        INSERT INTO recording_asset_schema_migrations (migration_id, schema_version, applied_at)
        VALUES (?, ?, ?)
        """,
        (RECORDING_ASSET_INGEST_MIGRATION_ID, RECORDING_ASSET_INGEST_SCHEMA_VERSION, now_utc()),
    )
    return 1


def upsert_import_package(
    con: sqlite3.Connection,
    package_ref: str,
    package_root: Path,
    metadata_csv_path: Path,
    audio_dir: Path,
    metadata_rows: int,
    package_hash: str,
    run_id: str,
    now: str,
) -> None:
    con.execute(
        """
        INSERT INTO recording_import_packages (
          package_ref, package_root, metadata_csv_path, audio_dir, metadata_rows,
          package_hash, first_ingested_at, last_ingested_at, last_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(package_ref) DO UPDATE SET
          package_root = excluded.package_root,
          metadata_csv_path = excluded.metadata_csv_path,
          audio_dir = excluded.audio_dir,
          metadata_rows = excluded.metadata_rows,
          package_hash = excluded.package_hash,
          last_ingested_at = excluded.last_ingested_at,
          last_run_id = excluded.last_run_id
        """,
        (
            package_ref,
            str(package_root),
            str(metadata_csv_path),
            str(audio_dir),
            metadata_rows,
            package_hash,
            now,
            now,
            run_id,
        ),
    )


def insert_recording_asset(con: sqlite3.Connection, item: Mapping[str, Any], package_ref: str, run_id: str, now: str) -> None:
    con.execute(
        """
        INSERT INTO captured_recording_assets (
          tenant_id, provider, event_key, provider_call_id, recording_id,
          source, source_audio_path, audio_path, source_filename, checksum_sha256,
          size_bytes, duration_sec, started_at, direction, client_phone, manager_ref,
          manager_name, status, package_ref, last_run_id, metadata_json, first_ingested_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        asset_params(item, package_ref=package_ref, run_id=run_id, now=now, include_first_ingested=True),
    )


def update_recording_asset(
    con: sqlite3.Connection,
    item: Mapping[str, Any],
    asset_id: int,
    package_ref: str,
    run_id: str,
    now: str,
) -> None:
    params = asset_params(item, package_ref=package_ref, run_id=run_id, now=now, include_first_ingested=False)
    con.execute(
        """
        UPDATE captured_recording_assets
           SET tenant_id = ?,
               provider = ?,
               event_key = ?,
               provider_call_id = ?,
               recording_id = ?,
               source = ?,
               source_audio_path = ?,
               audio_path = ?,
               source_filename = ?,
               checksum_sha256 = ?,
               size_bytes = ?,
               duration_sec = ?,
               started_at = ?,
               direction = ?,
               client_phone = ?,
               manager_ref = ?,
               manager_name = ?,
               status = ?,
               package_ref = ?,
               last_run_id = ?,
               metadata_json = ?,
               updated_at = ?
         WHERE id = ?
        """,
        params + (asset_id,),
    )


def asset_params(
    item: Mapping[str, Any],
    package_ref: str,
    run_id: str,
    now: str,
    include_first_ingested: bool,
) -> tuple[Any, ...]:
    base = (
        clean(item.get("tenant_id")),
        clean(item.get("provider")),
        clean(item.get("event_key")),
        clean(item.get("provider_call_id")),
        clean(item.get("recording_id")),
        clean(item.get("source")) or None,
        clean(item.get("source_audio_path")) or None,
        clean(item.get("target_audio_path")),
        clean(item.get("filename")),
        clean(item.get("checksum_sha256")),
        int(item.get("actual_size_bytes") or item.get("size_bytes") or 0),
        optional_float(item.get("duration_sec")),
        clean(item.get("started_at")) or None,
        clean(item.get("direction")) or None,
        clean(item.get("client_phone")) or None,
        clean(item.get("manager_ref")) or None,
        clean(item.get("manager_name")) or None,
        READY_STATUS,
        package_ref,
        run_id,
        json.dumps(item.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
    )
    if include_first_ingested:
        return base + (now, now)
    return base + (now,)


def find_existing_asset(con: sqlite3.Connection, item: Mapping[str, Any]) -> Optional[sqlite3.Row]:
    return con.execute(
        """
        SELECT *
          FROM captured_recording_assets
         WHERE tenant_id = ?
           AND provider = ?
           AND event_key = ?
        """,
        (clean(item.get("tenant_id")), clean(item.get("provider")), clean(item.get("event_key"))),
    ).fetchone()


def find_asset_conflict(con: sqlite3.Connection, item: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    row = con.execute(
        """
        SELECT id, event_key, recording_id
          FROM captured_recording_assets
         WHERE tenant_id = ?
           AND provider = ?
           AND recording_id = ?
           AND event_key != ?
        """,
        (
            clean(item.get("tenant_id")),
            clean(item.get("provider")),
            clean(item.get("recording_id")),
            clean(item.get("event_key")),
        ),
    ).fetchone()
    return dict(row) if row else None


def asset_matches(existing: sqlite3.Row, item: Mapping[str, Any], package_ref: str) -> bool:
    fields = {
        "provider_call_id": clean(item.get("provider_call_id")),
        "recording_id": clean(item.get("recording_id")),
        "audio_path": clean(item.get("target_audio_path")),
        "source_filename": clean(item.get("filename")),
        "checksum_sha256": clean(item.get("checksum_sha256")),
        "size_bytes": str(int(item.get("actual_size_bytes") or item.get("size_bytes") or 0)),
        "status": READY_STATUS,
        "package_ref": package_ref,
    }
    for field, expected in fields.items():
        if clean(existing[field]) != expected:
            return False
    return True


def insert_ingest_run(
    con: sqlite3.Connection,
    run_id: str,
    package_ref: str,
    started_at: str,
    finished_at: str,
    metadata_rows: int,
    planned_assets: int,
    inserted: int,
    already_present: int,
    updated: int,
    blocked: int,
    warnings: int,
    dry_run: bool,
    audit_json: str,
) -> None:
    con.execute(
        """
        INSERT INTO recording_asset_ingest_runs (
          run_id, package_ref, started_at, finished_at, metadata_rows, planned_assets,
          inserted, already_present, updated, blocked, warnings, dry_run, audit_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            package_ref,
            started_at,
            finished_at,
            metadata_rows,
            planned_assets,
            inserted,
            already_present,
            updated,
            blocked,
            warnings,
            1 if dry_run else 0,
            audit_json,
        ),
    )


def audit_recording_asset_ingest_db(
    con: sqlite3.Connection,
    product_root: Path,
    audio_dir: Path,
    package_ref: str,
    expected_items: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    rows = con.execute(
        """
        SELECT *
          FROM captured_recording_assets
         WHERE package_ref = ?
         ORDER BY started_at, id
        """,
        (package_ref,),
    ).fetchall()
    expected_events = {clean(item.get("event_key")) for item in expected_items if item["action"] != "BLOCK_RECORDING_ASSET_INGEST"}
    db_events = {clean(row["event_key"]) for row in rows}
    missing_db_for_metadata = sorted(expected_events - db_events)
    extra_db_not_in_metadata = sorted(db_events - expected_events)
    missing_audio = []
    checksum_mismatch = []
    path_outside_product_root = []
    path_outside_audio_dir = []
    for row in rows:
        audio_path = Path(clean(row["audio_path"])).resolve(strict=False)
        if not path_is_relative_to(audio_path, product_root):
            path_outside_product_root.append(str(audio_path))
        if not path_is_relative_to(audio_path, audio_dir):
            path_outside_audio_dir.append(str(audio_path))
        if not audio_path.exists():
            missing_audio.append(str(audio_path))
            continue
        if clean(row["checksum_sha256"]) != sha256_file(audio_path):
            checksum_mismatch.append(str(audio_path))
    blocked_reasons = {
        "missing_db_for_metadata": len(missing_db_for_metadata),
        "extra_db_not_in_metadata": len(extra_db_not_in_metadata),
        "missing_audio": len(missing_audio),
        "checksum_mismatch": len(checksum_mismatch),
        "path_outside_product_root": len(path_outside_product_root),
        "path_outside_audio_dir": len(path_outside_audio_dir),
    }
    return {
        "schema_migrations": relation_count(con, "recording_asset_schema_migrations"),
        "import_packages": relation_count(con, "recording_import_packages"),
        "ingest_runs": relation_count(con, "recording_asset_ingest_runs"),
        "assets_total": relation_count(con, "captured_recording_assets"),
        "assets_for_package": len(rows),
        "status_counts": count_query(con, "captured_recording_assets", "status", "package_ref = ?", (package_ref,)),
        "manager_counts": count_query(con, "captured_recording_assets", "manager_ref", "package_ref = ?", (package_ref,)),
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "missing_db_for_metadata": missing_db_for_metadata[:100],
        "extra_db_not_in_metadata": extra_db_not_in_metadata[:100],
        "samples": {"assets": [dict(row) for row in rows[:20]]},
    }


def read_metadata_rows(metadata_csv_path: Path, limit: Optional[int] = None) -> list[Mapping[str, str]]:
    with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    return rows[:limit] if limit is not None else rows


def resolve_ingest_paths(
    product_root: Path,
    package_root: Path,
    audio_dir: Path,
    metadata_csv_path: Path,
    db_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "package_root": package_root.resolve(strict=False),
        "audio_dir": audio_dir.resolve(strict=False),
        "metadata_csv_path": metadata_csv_path.resolve(strict=False),
        "db_path": db_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_ingest_paths(**paths)
    return paths


def guard_ingest_paths(
    product_root: Path,
    package_root: Path,
    audio_dir: Path,
    metadata_csv_path: Path,
    db_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("package root", package_root),
        ("audio dir", audio_dir),
        ("metadata csv", metadata_csv_path),
        ("recording asset ingest DB", db_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {db_path.name}")
    if db_path.suffix.lower() not in SAFE_DB_SUFFIXES:
        raise ValueError(f"recording asset ingest DB suffix must be one of: {sorted(SAFE_DB_SUFFIXES)}")
    if db_path.exists() and db_path.is_dir():
        raise ValueError(f"recording asset ingest DB path is a directory: {db_path}")
    if not package_root.exists() or not package_root.is_dir():
        raise FileNotFoundError(f"package root not found: {package_root}")
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")
    if not metadata_csv_path.exists() or not metadata_csv_path.is_file():
        raise FileNotFoundError(f"metadata csv not found: {metadata_csv_path}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing audit output under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"audit output must stay under product root: {product_root}")


def remove_sqlite_db(db_path: Path, product_root: Path) -> bool:
    removed = False
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        path = Path(f"{db_path}{suffix}")
        if not path.exists():
            continue
        if not path_is_relative_to(path.resolve(strict=False), product_root):
            raise ValueError(f"refusing to remove SQLite sidecar outside product root: {path}")
        if path.is_dir():
            raise ValueError(f"refusing to remove directory SQLite sidecar: {path}")
        path.unlink()
        removed = True
    return removed


def action_counts(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))


def count_item_warnings(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(item.get("warnings") or []) for item in items)


def relation_count(con: sqlite3.Connection, table: str) -> int:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    if not row:
        return 0
    result = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(result[0] or 0) if result else 0


def count_query(
    con: sqlite3.Connection,
    table: str,
    field: str,
    where_sql: str,
    params: Sequence[Any],
) -> Mapping[str, int]:
    rows = con.execute(
        f"""
        SELECT coalesce(nullif({field}, ''), 'empty') AS value, count(*) AS n
          FROM {table}
         WHERE {where_sql}
         GROUP BY value
         ORDER BY value
        """,
        tuple(params),
    ).fetchall()
    return {clean(row["value"]): int(row["n"] or 0) for row in rows}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
