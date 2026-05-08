from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.provider_metadata import PROVIDER_METADATA_TABLE
from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


MANAGER_IDENTITY_SCHEMA_VERSION = "manager_identity_map_v1"
MANAGER_IDENTITY_TABLE = "manager_identity_map"
MANAGER_IDENTITY_VIEW = "provider_call_metadata_with_manager"


@dataclass(frozen=True)
class ManagerIdentitySummary:
    schema_version: str
    db_path: str
    mango_users_path: str
    amo_users_path: Optional[str]
    table_name: str
    view_name: str
    replaced_existing_table: bool
    manager_extensions: int
    sidecar_rows: int
    view_rows: int
    mapped_mango_users: int
    missing_mango_users: int
    crm_owner_matched: int
    crm_owner_unmatched: int
    calls_with_mango_user: int
    calls_with_crm_owner: int
    blocked: int
    warnings: int
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def install_manager_identity_map(
    db_path: Path,
    mango_users_path: Path,
    out_allowed_root: Path,
    amo_users_path: Optional[Path] = None,
    replace_existing: bool = False,
    csv_out: Optional[Path] = None,
) -> Mapping[str, Any]:
    db_path = db_path.resolve(strict=False)
    mango_users_path = mango_users_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    amo_users_path = amo_users_path.resolve(strict=False) if amo_users_path else None
    csv_out = csv_out.resolve(strict=False) if csv_out else None
    guard_manager_identity_paths(
        db_path=db_path,
        mango_users_path=mango_users_path,
        amo_users_path=amo_users_path,
        out_allowed_root=out_allowed_root,
        csv_out=csv_out,
    )

    mango_users = parse_mango_users(mango_users_path)
    amo_users = parse_amo_users(amo_users_path) if amo_users_path else []
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        extension_stats = read_manager_extension_stats(con)
        replaced_existing_table = create_manager_identity_schema(con, replace_existing=replace_existing)
        rows = build_manager_identity_rows(
            extension_stats=extension_stats,
            mango_users=mango_users,
            amo_users=amo_users,
            mango_users_path=mango_users_path,
        )
        upsert_manager_identity_rows(con, rows)
        con.commit()
        audit = audit_manager_identity(con, rows=rows)

    if csv_out:
        write_manager_identity_csv(rows=rows, csv_out=csv_out)

    blocked = int(audit["blocked"])
    warnings = int(audit["warnings"])
    summary = ManagerIdentitySummary(
        schema_version=MANAGER_IDENTITY_SCHEMA_VERSION,
        db_path=str(db_path),
        mango_users_path=str(mango_users_path),
        amo_users_path=str(amo_users_path) if amo_users_path else None,
        table_name=MANAGER_IDENTITY_TABLE,
        view_name=MANAGER_IDENTITY_VIEW,
        replaced_existing_table=replaced_existing_table,
        manager_extensions=len(rows),
        sidecar_rows=int(audit["sidecar_rows"]),
        view_rows=int(audit["view_rows"]),
        mapped_mango_users=int(audit["mapped_mango_users"]),
        missing_mango_users=int(audit["missing_mango_users"]),
        crm_owner_matched=int(audit["crm_owner_matched"]),
        crm_owner_unmatched=int(audit["crm_owner_unmatched"]),
        calls_with_mango_user=int(audit["calls_with_mango_user"]),
        calls_with_crm_owner=int(audit["calls_with_crm_owner"]),
        blocked=blocked,
        warnings=warnings,
        validation_ok=blocked == 0,
    )
    return {
        "summary": summary.to_json_dict(),
        "audit": audit,
        "items": rows,
    }


def guard_manager_identity_paths(
    db_path: Path,
    mango_users_path: Path,
    amo_users_path: Optional[Path],
    out_allowed_root: Path,
    csv_out: Optional[Path],
) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing to use runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts:
        raise ValueError("refusing to write manager identity under stable_runtime")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"manager identity DB must stay under allowed root: {out_allowed_root}")
    if csv_out and not path_is_relative_to(csv_out, out_allowed_root):
        raise ValueError(f"manager identity CSV must stay under allowed root: {out_allowed_root}")
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"disposable DB not found: {db_path}")
    if not mango_users_path.exists() or not mango_users_path.is_file():
        raise FileNotFoundError(f"Mango users JSON not found: {mango_users_path}")
    if amo_users_path and (not amo_users_path.exists() or not amo_users_path.is_file()):
        raise FileNotFoundError(f"AMO users JSON not found: {amo_users_path}")


def create_manager_identity_schema(con: sqlite3.Connection, replace_existing: bool = False) -> bool:
    con.execute(f"DROP VIEW IF EXISTS {MANAGER_IDENTITY_VIEW}")
    if replace_existing:
        con.execute(f"DROP TABLE IF EXISTS {MANAGER_IDENTITY_TABLE}")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MANAGER_IDENTITY_TABLE} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id TEXT NOT NULL,
          provider TEXT NOT NULL,
          manager_extension TEXT NOT NULL,
          call_count INTEGER NOT NULL,
          first_call_started_at TEXT,
          last_call_started_at TEXT,
          mango_name TEXT,
          mango_email TEXT,
          mango_department TEXT,
          mango_position TEXT,
          mango_user_source_ref TEXT,
          crm_owner_id INTEGER,
          crm_owner_name TEXT,
          crm_owner_email TEXT,
          crm_match_status TEXT NOT NULL,
          mapping_status TEXT NOT NULL,
          notes TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(tenant_id, provider, manager_extension)
        )
        """
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{MANAGER_IDENTITY_TABLE}_tenant_provider "
        f"ON {MANAGER_IDENTITY_TABLE} (tenant_id, provider)"
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{MANAGER_IDENTITY_TABLE}_crm_owner_id "
        f"ON {MANAGER_IDENTITY_TABLE} (crm_owner_id)"
    )
    con.execute(
        f"""
        CREATE VIEW {MANAGER_IDENTITY_VIEW} AS
        SELECT
          pcm.*,
          mim.mango_name AS manager_display_name,
          mim.mango_email AS manager_email,
          mim.crm_owner_id AS manager_crm_owner_id,
          mim.crm_owner_name AS manager_crm_owner_name,
          mim.crm_match_status AS manager_crm_match_status,
          mim.mapping_status AS manager_mapping_status
        FROM {PROVIDER_METADATA_TABLE} pcm
        LEFT JOIN {MANAGER_IDENTITY_TABLE} mim
          ON mim.tenant_id = pcm.tenant_id
         AND mim.provider = pcm.provider
         AND mim.manager_extension = pcm.manager_extension
        """
    )
    return replace_existing


def read_manager_extension_stats(con: sqlite3.Connection) -> list[Mapping[str, Any]]:
    rows = con.execute(
        f"""
        SELECT
          pcm.tenant_id,
          pcm.provider,
          pcm.manager_extension,
          count(*) AS call_count,
          min(cr.started_at) AS first_call_started_at,
          max(cr.started_at) AS last_call_started_at
        FROM {PROVIDER_METADATA_TABLE} pcm
        LEFT JOIN call_records cr ON cr.id = pcm.call_record_id
        GROUP BY pcm.tenant_id, pcm.provider, pcm.manager_extension
        ORDER BY call_count DESC, pcm.manager_extension
        """
    ).fetchall()
    return [dict(row) for row in rows]


def parse_mango_users(path: Path) -> Mapping[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    users = payload.get("users") if isinstance(payload, Mapping) else payload
    result: dict[str, Mapping[str, Any]] = {}
    if not isinstance(users, list):
        return result
    for index, user in enumerate(users, start=1):
        if not isinstance(user, Mapping):
            continue
        general = user.get("general") if isinstance(user.get("general"), Mapping) else {}
        telephony = user.get("telephony") if isinstance(user.get("telephony"), Mapping) else {}
        extension = clean(telephony.get("extension"))
        if not extension:
            continue
        result[extension] = {
            "extension": extension,
            "name": clean(general.get("name")),
            "email": clean(general.get("email")),
            "department": clean(general.get("department")),
            "position": clean(general.get("position")),
            "source_index": index,
        }
    return result


def parse_amo_users(path: Optional[Path]) -> list[Mapping[str, Any]]:
    if not path:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    users = payload.get("users") if isinstance(payload, Mapping) else payload
    if not isinstance(users, list):
        return []
    result = []
    for user in users:
        if not isinstance(user, Mapping):
            continue
        result.append(
            {
                "id": optional_int(user.get("id")),
                "name": clean(user.get("name")),
                "email": clean(user.get("email")),
                "is_active": bool(user.get("is_active", True)),
            }
        )
    return result


def build_manager_identity_rows(
    extension_stats: Sequence[Mapping[str, Any]],
    mango_users: Mapping[str, Mapping[str, Any]],
    amo_users: Sequence[Mapping[str, Any]],
    mango_users_path: Path,
) -> list[Mapping[str, Any]]:
    amo_by_email = {normalize_key(user.get("email")): user for user in amo_users if clean(user.get("email"))}
    amo_by_name = {normalize_key(user.get("name")): user for user in amo_users if clean(user.get("name"))}
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for stat in extension_stats:
        extension = clean(stat.get("manager_extension"))
        mango_user = mango_users.get(extension, {})
        crm_owner, crm_status = match_crm_owner(mango_user=mango_user, amo_by_email=amo_by_email, amo_by_name=amo_by_name)
        mapping_status = "mapped_mango_user" if mango_user else "missing_mango_user"
        rows.append(
            {
                "tenant_id": clean(stat.get("tenant_id")),
                "provider": clean(stat.get("provider")),
                "manager_extension": extension,
                "call_count": int(stat.get("call_count") or 0),
                "first_call_started_at": clean(stat.get("first_call_started_at")) or None,
                "last_call_started_at": clean(stat.get("last_call_started_at")) or None,
                "mango_name": clean(mango_user.get("name")) or None,
                "mango_email": clean(mango_user.get("email")) or None,
                "mango_department": clean(mango_user.get("department")) or None,
                "mango_position": clean(mango_user.get("position")) or None,
                "mango_user_source_ref": f"{mango_users_path}#extension={extension}" if mango_user else None,
                "crm_owner_id": crm_owner.get("id") if crm_owner else None,
                "crm_owner_name": clean(crm_owner.get("name")) if crm_owner else None,
                "crm_owner_email": clean(crm_owner.get("email")) if crm_owner else None,
                "crm_match_status": crm_status,
                "mapping_status": mapping_status,
                "notes": None if mango_user else "Mango user not found for extension",
                "created_at": now,
                "updated_at": now,
            }
        )
    return rows


def match_crm_owner(
    mango_user: Mapping[str, Any],
    amo_by_email: Mapping[str, Mapping[str, Any]],
    amo_by_name: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str]:
    if not mango_user:
        return {}, "missing_mango_user"
    email_key = normalize_key(mango_user.get("email"))
    if email_key and email_key in amo_by_email:
        return amo_by_email[email_key], "matched_email"
    name_key = normalize_key(mango_user.get("name"))
    if name_key and name_key in amo_by_name:
        return amo_by_name[name_key], "matched_name"
    return {}, "unmatched"


def upsert_manager_identity_rows(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        con.execute(
            f"""
            INSERT INTO {MANAGER_IDENTITY_TABLE} (
              tenant_id, provider, manager_extension, call_count,
              first_call_started_at, last_call_started_at,
              mango_name, mango_email, mango_department, mango_position,
              mango_user_source_ref, crm_owner_id, crm_owner_name, crm_owner_email,
              crm_match_status, mapping_status, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, provider, manager_extension) DO UPDATE SET
              call_count = excluded.call_count,
              first_call_started_at = excluded.first_call_started_at,
              last_call_started_at = excluded.last_call_started_at,
              mango_name = excluded.mango_name,
              mango_email = excluded.mango_email,
              mango_department = excluded.mango_department,
              mango_position = excluded.mango_position,
              mango_user_source_ref = excluded.mango_user_source_ref,
              crm_owner_id = excluded.crm_owner_id,
              crm_owner_name = excluded.crm_owner_name,
              crm_owner_email = excluded.crm_owner_email,
              crm_match_status = excluded.crm_match_status,
              mapping_status = excluded.mapping_status,
              notes = excluded.notes,
              updated_at = excluded.updated_at
            """,
            manager_identity_values(row),
        )


def manager_identity_values(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        clean(row.get("tenant_id")),
        clean(row.get("provider")),
        clean(row.get("manager_extension")),
        int(row.get("call_count") or 0),
        row.get("first_call_started_at"),
        row.get("last_call_started_at"),
        row.get("mango_name"),
        row.get("mango_email"),
        row.get("mango_department"),
        row.get("mango_position"),
        row.get("mango_user_source_ref"),
        row.get("crm_owner_id"),
        row.get("crm_owner_name"),
        row.get("crm_owner_email"),
        clean(row.get("crm_match_status")),
        clean(row.get("mapping_status")),
        row.get("notes"),
        clean(row.get("created_at")),
        clean(row.get("updated_at")),
    )


def audit_manager_identity(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    sidecar_rows = int(con.execute(f"select count(*) from {PROVIDER_METADATA_TABLE}").fetchone()[0])
    table_rows = con.execute(f"select * from {MANAGER_IDENTITY_TABLE} order by call_count desc, manager_extension").fetchall()
    table = [dict(row) for row in table_rows]
    view_rows = int(con.execute(f"select count(*) from {MANAGER_IDENTITY_VIEW}").fetchone()[0])
    mapped = [row for row in table if clean(row.get("mapping_status")) == "mapped_mango_user"]
    missing = [row for row in table if clean(row.get("mapping_status")) != "mapped_mango_user"]
    crm_matched = [row for row in table if clean(row.get("crm_match_status")).startswith("matched_")]
    crm_unmatched = [row for row in table if clean(row.get("crm_match_status")) == "unmatched"]
    calls_with_mango_user = sum(int(row.get("call_count") or 0) for row in mapped)
    calls_with_crm_owner = sum(int(row.get("call_count") or 0) for row in crm_matched)
    crm_owner_unmatched_call_count = sum(int(row.get("call_count") or 0) for row in crm_unmatched)
    missing_mango_user_call_count = sum(int(row.get("call_count") or 0) for row in missing)
    blocked_reasons = {
        "view_row_mismatch": 0 if view_rows == sidecar_rows else 1,
        "missing_mango_users": len(missing),
    }
    warning_reasons = {
        "crm_owner_unmatched": len(crm_unmatched),
    }
    crm_match_call_counts = Counter()
    mapping_status_call_counts = Counter()
    for row in table:
        call_count = int(row.get("call_count") or 0)
        crm_match_call_counts[clean(row.get("crm_match_status"))] += call_count
        mapping_status_call_counts[clean(row.get("mapping_status"))] += call_count
    return {
        "table_name": MANAGER_IDENTITY_TABLE,
        "view_name": MANAGER_IDENTITY_VIEW,
        "manager_extensions": len(table),
        "sidecar_rows": sidecar_rows,
        "view_rows": view_rows,
        "mapped_mango_users": len(mapped),
        "missing_mango_users": len(missing),
        "crm_owner_matched": len(crm_matched),
        "crm_owner_unmatched": len(crm_unmatched),
        "calls_with_mango_user": calls_with_mango_user,
        "calls_with_crm_owner": calls_with_crm_owner,
        "crm_owner_unmatched_call_count": crm_owner_unmatched_call_count,
        "missing_mango_user_call_count": missing_mango_user_call_count,
        "mapping_status_counts": dict(sorted(Counter(clean(row.get("mapping_status")) for row in table).items())),
        "crm_match_status_counts": dict(sorted(Counter(clean(row.get("crm_match_status")) for row in table).items())),
        "mapping_status_call_counts": dict(sorted(mapping_status_call_counts.items())),
        "crm_match_status_call_counts": dict(sorted(crm_match_call_counts.items())),
        "manager_call_counts": {
            clean(row.get("manager_extension")): int(row.get("call_count") or 0) for row in table
        },
        "blocked": sum(blocked_reasons.values()),
        "blocked_reasons": blocked_reasons,
        "warnings": sum(warning_reasons.values()),
        "warning_reasons": warning_reasons,
        "manual_review_items": manual_review_items(table),
        "samples": {
            "manager_identity_rows": table[:20],
        },
    }


def manual_review_items(table: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    items = []
    for row in table:
        mapping_status = clean(row.get("mapping_status"))
        crm_match_status = clean(row.get("crm_match_status"))
        if mapping_status == "mapped_mango_user" and crm_match_status != "unmatched":
            continue
        reason = "missing_mango_user" if mapping_status != "mapped_mango_user" else "crm_owner_unmatched"
        items.append(
            {
                "tenant_id": clean(row.get("tenant_id")),
                "provider": clean(row.get("provider")),
                "manager_extension": clean(row.get("manager_extension")),
                "call_count": int(row.get("call_count") or 0),
                "mango_name": clean(row.get("mango_name")) or None,
                "mango_email": clean(row.get("mango_email")) or None,
                "crm_match_status": crm_match_status,
                "mapping_status": mapping_status,
                "reason": reason,
                "notes": clean(row.get("notes")) or None,
            }
        )
    return items


def write_manager_identity_csv(rows: Sequence[Mapping[str, Any]], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tenant_id",
        "provider",
        "manager_extension",
        "call_count",
        "first_call_started_at",
        "last_call_started_at",
        "mango_name",
        "mango_email",
        "mango_department",
        "mango_position",
        "crm_owner_id",
        "crm_owner_name",
        "crm_owner_email",
        "crm_match_status",
        "mapping_status",
        "notes",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_key(value: Any) -> str:
    return " ".join(clean(value).casefold().split())


def optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
