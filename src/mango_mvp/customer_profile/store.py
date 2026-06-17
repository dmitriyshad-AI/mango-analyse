from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_profile.contracts import ProfileFieldCandidate, ProfileSnapshot
from mango_mvp.utils.phone import normalize_phone


CUSTOMER_PROFILE_SQLITE_SCHEMA_VERSION = "customer_profile_sqlite_v1"


def _phone_index_enabled() -> bool:
    return os.getenv("PROFILE_PHONE_INDEX", "1") == "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CustomerProfileSQLiteStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.db_path)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        self.bootstrap()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "CustomerProfileSQLiteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def bootstrap(self) -> None:
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS customer_profiles (
              profile_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              primary_phone TEXT,
              display_name TEXT,
              built_at TEXT NOT NULL,
              build_id TEXT NOT NULL,
              source_event_count INTEGER NOT NULL,
              last_event_at TEXT
            );
            CREATE TABLE IF NOT EXISTS profile_fields (
              field_id TEXT PRIMARY KEY,
              profile_id TEXT NOT NULL REFERENCES customer_profiles(profile_id),
              field TEXT NOT NULL,
              value TEXT NOT NULL,
              child_key TEXT NOT NULL DEFAULT '',
              brand TEXT NOT NULL DEFAULT 'unknown',
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              event_at TEXT NOT NULL,
              quote TEXT NOT NULL DEFAULT '',
              superseded_by TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_profile_fields_lookup
              ON profile_fields(profile_id, field, child_key, superseded_by);
            CREATE TABLE IF NOT EXISTS profile_builds (
              build_id TEXT PRIMARY KEY,
              started_at TEXT,
              finished_at TEXT,
              timeline_db_path TEXT,
              timeline_db_sha256 TEXT,
              profiles_built INTEGER,
              notes TEXT
            );
            """
        )
        if _phone_index_enabled():
            self._ensure_phone_index()
        self._con.commit()

    def _ensure_phone_index(self) -> None:
        columns = {
            str(row["name"])
            for row in self._con.execute("PRAGMA table_info(customer_profiles)").fetchall()
        }
        if "primary_phone_norm" not in columns:
            self._con.execute("ALTER TABLE customer_profiles ADD COLUMN primary_phone_norm TEXT")
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_customer_profiles_phone_norm ON customer_profiles(primary_phone_norm)"
        )

    def replace_profiles(
        self,
        *,
        build_id: str,
        built_at: datetime,
        timeline_db_path: Path,
        timeline_db_sha256: str,
        profiles: Sequence[ProfileSnapshot],
        fields: Sequence[ProfileFieldCandidate],
        notes: str = "",
    ) -> Mapping[str, Any]:
        if built_at.tzinfo is None or built_at.utcoffset() is None:
            raise ValueError("built_at must be timezone-aware")
        profile_ids = [profile.profile_id for profile in profiles]
        with self._con:
            for profile_id in profile_ids:
                self._con.execute("DELETE FROM profile_fields WHERE profile_id = ?", (profile_id,))
                self._con.execute("DELETE FROM customer_profiles WHERE profile_id = ?", (profile_id,))
            for profile in profiles:
                base_values = (
                    profile.profile_id,
                    profile.tenant_id,
                    profile.primary_phone,
                    profile.display_name,
                    built_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    build_id,
                    int(profile.source_event_count),
                    profile.last_event_at.astimezone(timezone.utc).isoformat(timespec="seconds")
                    if profile.last_event_at
                    else None,
                )
                if _phone_index_enabled():
                    self._ensure_phone_index()
                    self._con.execute(
                        """
                        INSERT INTO customer_profiles (
                          profile_id, tenant_id, primary_phone, display_name, built_at, build_id,
                          source_event_count, last_event_at, primary_phone_norm
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (*base_values, normalize_phone(profile.primary_phone) or ""),
                    )
                else:
                    self._con.execute(
                        """
                        INSERT INTO customer_profiles (
                          profile_id, tenant_id, primary_phone, display_name, built_at, build_id,
                          source_event_count, last_event_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        base_values,
                    )
            for field in fields:
                self._con.execute(
                    """
                    INSERT INTO profile_fields (
                      field_id, profile_id, field, value, child_key, brand, source_system,
                      source_ref, event_at, quote, superseded_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        field.field_id,
                        field.profile_id,
                        field.field,
                        field.value,
                        field.child_key,
                        field.brand,
                        field.source_system,
                        field.source_ref,
                        field.event_at.isoformat(timespec="seconds"),
                        field.quote,
                        field.superseded_by,
                    ),
                )
            self._con.execute(
                """
                INSERT OR REPLACE INTO profile_builds (
                  build_id, started_at, finished_at, timeline_db_path, timeline_db_sha256,
                  profiles_built, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    built_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    str(timeline_db_path),
                    timeline_db_sha256,
                    len(profiles),
                    notes,
                ),
            )
        return {
            "profiles_built": len(profiles),
            "fields_written": len(fields),
            "superseded_fields": sum(1 for field in fields if field.superseded_by),
            "build_id": build_id,
        }

    def active_fields(self, profile_id: str) -> list[Mapping[str, Any]]:
        rows = self._con.execute(
            """
            SELECT * FROM profile_fields
            WHERE profile_id = ? AND superseded_by = ''
            ORDER BY field, child_key, event_at, source_ref
            """,
            (profile_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> Mapping[str, Any]:
        counts = {}
        for table in ("customer_profiles", "profile_fields", "profile_builds"):
            counts[table] = int(self._con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        superseded = int(self._con.execute("SELECT count(*) FROM profile_fields WHERE superseded_by <> ''").fetchone()[0])
        return {
            "schema_version": CUSTOMER_PROFILE_SQLITE_SCHEMA_VERSION,
            "db_path": str(self.db_path),
            "counts": counts,
            "superseded_fields": superseded,
        }
