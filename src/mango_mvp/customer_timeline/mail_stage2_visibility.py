from __future__ import annotations

import sqlite3
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    json_dumps,
    json_loads,
    scrub_timeline_persisted_json,
)


MAIL_STAGE2_VISIBILITY_HARDENING_SCHEMA_VERSION = "mail_stage2_visibility_hardening_v1"
MAIL_STAGE2_SOURCE_SYSTEM = "mail_archive_stage2"


def mail_stage2_visibility_gate(
    db_path: Path | str,
    *,
    allowed_root: Path | str,
    source_system: str = MAIL_STAGE2_SOURCE_SYSTEM,
    allow_test_paths: bool = False,
) -> Mapping[str, Any]:
    _require_mail_stage2_source(source_system)
    db = _guard_staging_db(db_path, allowed_root=allowed_root, allow_test_paths=allow_test_paths)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        return _visibility_counts(con, source_system=source_system)


def harden_mail_stage2_bot_visibility(
    db_path: Path | str,
    *,
    allowed_root: Path | str,
    source_system: str = MAIL_STAGE2_SOURCE_SYSTEM,
    apply: bool = False,
    allow_test_paths: bool = False,
) -> Mapping[str, Any]:
    _require_mail_stage2_source(source_system)
    db = _guard_staging_db(db_path, allowed_root=allowed_root, allow_test_paths=allow_test_paths)
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    updated = 0
    diagnostic_candidates = 0
    if not apply:
        with sqlite3.connect(db) as con:
            con.row_factory = sqlite3.Row
            before = _visibility_counts(con, source_system=source_system)
            diagnostic_candidates = _diagnostic_candidates(con, source_system=source_system)
            after = before
    else:
        with CustomerTimelineSQLiteStore(db, allowed_root=Path(allowed_root).expanduser()) as store:
            con = store._con  # noqa: SLF001 - staging hardening keeps table columns and JSON in sync.
            before = _visibility_counts(con, source_system=source_system)
            diagnostic_candidates = _diagnostic_candidates(con, source_system=source_system)
            rows = con.execute(
                """
                SELECT chunk_id, record_json
                FROM bot_context_chunks
                WHERE source_system = ?
                  AND COALESCE(superseded_by, '') = ''
                  AND (allowed_for_bot != 0 OR requires_manager_review != 1)
                ORDER BY event_at, chunk_id
                """,
                (source_system,),
            ).fetchall()
            for row in rows:
                payload = json_loads(row["record_json"])
                payload["allowed_for_bot"] = False
                payload["requires_manager_review"] = True
                record_hash = stable_digest(scrub_timeline_persisted_json(payload))
                con.execute(
                    """
                    UPDATE bot_context_chunks
                    SET allowed_for_bot = 0,
                        requires_manager_review = 1,
                        record_json = ?,
                        record_hash = ?
                    WHERE chunk_id = ?
                    """,
                    (json_dumps(payload), record_hash, row["chunk_id"]),
                )
                updated += 1
            if updated:
                store._rebuild_fts_indexes()  # noqa: SLF001 - record_hash participates in chunk FTS summary.
            con.commit()
            after = _visibility_counts(con, source_system=source_system)
    return {
        "schema_version": MAIL_STAGE2_VISIBILITY_HARDENING_SCHEMA_VERSION,
        "db_path": str(db),
        "source_system": source_system,
        "apply": bool(apply),
        "before": before,
        "after": after,
        "updated_chunks": updated,
        "diagnostic_bot_eligible_candidates": diagnostic_candidates,
        "gate_passed": int(after["unsafe_active_chunks"]) == 0,
    }


def _guard_staging_db(path: Path | str, *, allowed_root: Path | str, allow_test_paths: bool) -> Path:
    db = guard_customer_timeline_output_path(Path(path).expanduser(), Path(allowed_root).expanduser())
    if "customer_timeline_prod_" in str(db):
        raise ValueError(f"refusing to harden prod timeline path: {db}")
    if not allow_test_paths:
        parts = tuple(part.casefold() for part in db.parts)
        if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
            raise ValueError(f"mail stage2 visibility hardening is restricted to .codex_local/staging paths: {db}")
    if not db.exists():
        raise FileNotFoundError(f"timeline DB does not exist: {db}")
    return db


def _require_mail_stage2_source(source_system: str) -> None:
    if source_system != MAIL_STAGE2_SOURCE_SYSTEM:
        raise ValueError(f"mail stage2 visibility hardening only supports {MAIL_STAGE2_SOURCE_SYSTEM}")


def _visibility_counts(con: sqlite3.Connection, *, source_system: str) -> Mapping[str, Any]:
    row = con.execute(
        """
        SELECT
          COUNT(*) AS total_chunks,
          SUM(CASE WHEN COALESCE(superseded_by, '') = '' THEN 1 ELSE 0 END) AS active_chunks,
          SUM(CASE WHEN COALESCE(superseded_by, '') = '' AND allowed_for_bot = 1 THEN 1 ELSE 0 END) AS active_allowed_for_bot,
          SUM(CASE WHEN COALESCE(superseded_by, '') = '' AND requires_manager_review = 1 THEN 1 ELSE 0 END) AS active_requires_manager_review,
          SUM(CASE WHEN COALESCE(superseded_by, '') = '' AND (allowed_for_bot != 0 OR requires_manager_review != 1) THEN 1 ELSE 0 END) AS unsafe_active_chunks
        FROM bot_context_chunks
        WHERE source_system = ?
        """,
        (source_system,),
    ).fetchone()
    flag_rows = con.execute(
        """
        SELECT allowed_for_bot, requires_manager_review, COUNT(*) AS count
        FROM bot_context_chunks
        WHERE source_system = ?
          AND COALESCE(superseded_by, '') = ''
        GROUP BY allowed_for_bot, requires_manager_review
        ORDER BY allowed_for_bot, requires_manager_review
        """,
        (source_system,),
    ).fetchall()
    flag_counts = {
        f"allowed_{int(row['allowed_for_bot'])}_review_{int(row['requires_manager_review'])}": int(row["count"])
        for row in flag_rows
    }
    return {
        "total_chunks": int(row["total_chunks"] or 0),
        "active_chunks": int(row["active_chunks"] or 0),
        "active_allowed_for_bot": int(row["active_allowed_for_bot"] or 0),
        "active_requires_manager_review": int(row["active_requires_manager_review"] or 0),
        "unsafe_active_chunks": int(row["unsafe_active_chunks"] or 0),
        "flag_counts": flag_counts,
    }


def _diagnostic_candidates(con: sqlite3.Connection, *, source_system: str) -> int:
    count = 0
    for row in con.execute(
        """
        SELECT record_json
        FROM bot_context_chunks
        WHERE source_system = ?
          AND COALESCE(superseded_by, '') = ''
        """,
        (source_system,),
    ):
        payload = json_loads(row["record_json"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), MappingABC) else {}
        if metadata.get("bot_eligible_candidate") is True:
            count += 1
    return count


def assert_mail_stage2_visibility_gate(
    db_path: Path | str,
    *,
    allowed_root: Path | str,
    source_system: str = MAIL_STAGE2_SOURCE_SYSTEM,
    allow_test_paths: bool = False,
) -> None:
    counts = mail_stage2_visibility_gate(
        db_path,
        allowed_root=allowed_root,
        source_system=source_system,
        allow_test_paths=allow_test_paths,
    )
    if int(counts["unsafe_active_chunks"]) != 0:
        raise ValueError(
            f"unsafe active {source_system} chunks: {counts['unsafe_active_chunks']} "
            "(expected allowed_for_bot=0 and requires_manager_review=1)"
        )
