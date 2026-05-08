from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.manager_identity import MANAGER_IDENTITY_TABLE, MANAGER_IDENTITY_VIEW
from mango_mvp.productization.provider_metadata import PROVIDER_METADATA_TABLE
from mango_mvp.productization.test_ingest import RUNTIME_DB_FILENAMES, clean, path_is_relative_to


PRODUCT_REPOSITORY_SCHEMA_VERSION = "product_repository_readonly_v1"


@dataclass(frozen=True)
class ProductRepositorySummary:
    schema_version: str
    db_path: str
    call_records: int
    provider_metadata_rows: int
    enriched_view_rows: int
    manager_extensions: int
    calls_with_manager_identity: int
    calls_with_crm_owner: int
    manual_owner_review_items: int
    raw_payload_refs_present: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductCallRecord:
    call_record_id: int
    source_filename: str
    started_at: Optional[str]
    duration_sec: Optional[float]
    tenant_id: str
    provider: str
    provider_call_id: str
    recording_id: str
    event_key: str
    manager_extension: str
    manager_display_name: Optional[str]
    manager_email: Optional[str]
    manager_crm_owner_id: Optional[int]
    manager_crm_owner_name: Optional[str]
    manager_crm_match_status: Optional[str]
    manager_mapping_status: Optional[str]
    raw_payload_ref: Optional[str]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagerRollupItem:
    tenant_id: str
    provider: str
    manager_extension: str
    call_count: int
    mango_name: Optional[str]
    mango_email: Optional[str]
    crm_owner_id: Optional[int]
    crm_owner_name: Optional[str]
    crm_owner_email: Optional[str]
    crm_match_status: str
    mapping_status: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class ProductRepository:
    """Read-only repository over a productization SQLite DB.

    The repository is intentionally scoped to disposable/productization DBs. It
    refuses runtime-looking DB names and `stable_runtime` paths.
    """

    def __init__(self, db_path: Path, out_allowed_root: Path) -> None:
        self.db_path = db_path.resolve(strict=False)
        self.out_allowed_root = out_allowed_root.resolve(strict=False)
        guard_repository_paths(self.db_path, self.out_allowed_root)

    def summary(self) -> ProductRepositorySummary:
        with self.connect() as con:
            call_records = count_table(con, "call_records")
            provider_rows = count_table(con, PROVIDER_METADATA_TABLE)
            enriched_rows = count_table(con, MANAGER_IDENTITY_VIEW, is_view=True)
            manager_extensions = count_table(con, MANAGER_IDENTITY_TABLE)
            calls_with_manager = scalar_int(
                con,
                f"select count(*) from {MANAGER_IDENTITY_VIEW} where manager_mapping_status = 'mapped_mango_user'",
            )
            calls_with_crm_owner = scalar_int(
                con,
                f"select count(*) from {MANAGER_IDENTITY_VIEW} where manager_crm_owner_id is not null",
            )
            manual_review_items = scalar_int(
                con,
                f"""
                select count(*) from {MANAGER_IDENTITY_TABLE}
                 where mapping_status != 'mapped_mango_user'
                    or crm_owner_id is null
                """,
            )
            raw_payload_refs = scalar_int(
                con,
                f"select count(*) from {PROVIDER_METADATA_TABLE} where raw_payload_ref is not null and raw_payload_ref != ''",
            )
        blocked = 0 if enriched_rows == provider_rows else 1
        warnings = manual_review_items
        return ProductRepositorySummary(
            schema_version=PRODUCT_REPOSITORY_SCHEMA_VERSION,
            db_path=str(self.db_path),
            call_records=call_records,
            provider_metadata_rows=provider_rows,
            enriched_view_rows=enriched_rows,
            manager_extensions=manager_extensions,
            calls_with_manager_identity=calls_with_manager,
            calls_with_crm_owner=calls_with_crm_owner,
            manual_owner_review_items=manual_review_items,
            raw_payload_refs_present=raw_payload_refs,
            validation_ok=blocked == 0,
            blocked=blocked,
            warnings=warnings,
        )

    def list_calls(
        self,
        limit: int = 100,
        offset: int = 0,
        manager_extension: Optional[str] = None,
        crm_owner_status: str = "all",
    ) -> Sequence[ProductCallRecord]:
        if limit < 1:
            return ()
        where = []
        params: list[Any] = []
        if manager_extension:
            where.append("pcm.manager_extension = ?")
            params.append(clean(manager_extension))
        if crm_owner_status == "present":
            where.append("pcm.manager_crm_owner_id is not null")
        elif crm_owner_status == "missing":
            where.append("pcm.manager_crm_owner_id is null")
        elif crm_owner_status != "all":
            raise ValueError("crm_owner_status must be one of: all, present, missing")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.extend([int(limit), int(offset)])
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT
                  pcm.call_record_id,
                  cr.source_filename,
                  cr.started_at,
                  cr.duration_sec,
                  pcm.tenant_id,
                  pcm.provider,
                  pcm.provider_call_id,
                  pcm.recording_id,
                  pcm.event_key,
                  pcm.manager_extension,
                  pcm.manager_display_name,
                  pcm.manager_email,
                  pcm.manager_crm_owner_id,
                  pcm.manager_crm_owner_name,
                  pcm.manager_crm_match_status,
                  pcm.manager_mapping_status,
                  pcm.raw_payload_ref
                FROM {MANAGER_IDENTITY_VIEW} pcm
                LEFT JOIN call_records cr ON cr.id = pcm.call_record_id
                {where_sql}
                ORDER BY cr.started_at DESC, pcm.call_record_id DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return tuple(product_call_from_row(row) for row in rows)

    def manager_rollup(self) -> Sequence[ManagerRollupItem]:
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT
                  tenant_id,
                  provider,
                  manager_extension,
                  call_count,
                  mango_name,
                  mango_email,
                  crm_owner_id,
                  crm_owner_name,
                  crm_owner_email,
                  crm_match_status,
                  mapping_status
                FROM {MANAGER_IDENTITY_TABLE}
                ORDER BY call_count DESC, manager_extension
                """
            ).fetchall()
        return tuple(manager_rollup_from_row(row) for row in rows)

    def manual_owner_review_queue(self) -> Sequence[ManagerRollupItem]:
        return tuple(
            row
            for row in self.manager_rollup()
            if row.mapping_status != "mapped_mango_user" or row.crm_owner_id is None
        )

    def connect(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(self.db_path), safe='/:')}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        return con


def guard_repository_paths(db_path: Path, out_allowed_root: Path) -> None:
    if db_path.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing to open runtime-looking DB filename: {db_path.name}")
    if "stable_runtime" in db_path.parts:
        raise ValueError("refusing to open stable_runtime DB through product repository")
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"product repository DB must stay under allowed root: {out_allowed_root}")
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"product repository DB not found: {db_path}")


def product_call_from_row(row: sqlite3.Row) -> ProductCallRecord:
    return ProductCallRecord(
        call_record_id=int(row["call_record_id"]),
        source_filename=clean(row["source_filename"]),
        started_at=clean(row["started_at"]) or None,
        duration_sec=optional_float(row["duration_sec"]),
        tenant_id=clean(row["tenant_id"]),
        provider=clean(row["provider"]),
        provider_call_id=clean(row["provider_call_id"]),
        recording_id=clean(row["recording_id"]),
        event_key=clean(row["event_key"]),
        manager_extension=clean(row["manager_extension"]),
        manager_display_name=clean(row["manager_display_name"]) or None,
        manager_email=clean(row["manager_email"]) or None,
        manager_crm_owner_id=optional_int(row["manager_crm_owner_id"]),
        manager_crm_owner_name=clean(row["manager_crm_owner_name"]) or None,
        manager_crm_match_status=clean(row["manager_crm_match_status"]) or None,
        manager_mapping_status=clean(row["manager_mapping_status"]) or None,
        raw_payload_ref=clean(row["raw_payload_ref"]) or None,
    )


def manager_rollup_from_row(row: sqlite3.Row) -> ManagerRollupItem:
    return ManagerRollupItem(
        tenant_id=clean(row["tenant_id"]),
        provider=clean(row["provider"]),
        manager_extension=clean(row["manager_extension"]),
        call_count=int(row["call_count"] or 0),
        mango_name=clean(row["mango_name"]) or None,
        mango_email=clean(row["mango_email"]) or None,
        crm_owner_id=optional_int(row["crm_owner_id"]),
        crm_owner_name=clean(row["crm_owner_name"]) or None,
        crm_owner_email=clean(row["crm_owner_email"]) or None,
        crm_match_status=clean(row["crm_match_status"]),
        mapping_status=clean(row["mapping_status"]),
    )


def count_table(con: sqlite3.Connection, name: str, is_view: bool = False) -> int:
    if not relation_exists(con, name, is_view=is_view):
        return 0
    return scalar_int(con, f"select count(*) from {name}")


def relation_exists(con: sqlite3.Connection, name: str, is_view: bool = False) -> bool:
    rel_type = "view" if is_view else "table"
    row = con.execute("select 1 from sqlite_master where type = ? and name = ?", (rel_type, name)).fetchone()
    return row is not None


def scalar_int(con: sqlite3.Connection, sql: str) -> int:
    try:
        row = con.execute(sql).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row else 0


def optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def optional_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
