#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.store import (  # noqa: E402
    CustomerTimelineSQLiteStore,
    customer_timeline_integrity_report,
    customer_timeline_integrity_report_ok,
    customer_timeline_run_lock,
    customer_timeline_writer_lock,
)
from scripts.publish_snapshot.common import (  # noqa: E402
    COMPACT_READER_MAX_BYTES,
    PublishSnapshotError,
    add_common_args,
    classify_publish_worktree_status,
    finish_cli,
    foreign_key_check,
    git_head,
    git_status_short,
    integrity_check,
    live_worktree_untracked,
    load_config,
    report_base,
    sha256_file,
    table_counts,
    user_version,
    wal_checkpoint_truncate,
    write_json,
)
from scripts.publish_snapshot.preflight import nightly_manifest_report  # noqa: E402
from scripts.publish_snapshot.reader_smoke import smoke as reader_smoke  # noqa: E402


_FTS_TABLE_PREFIXES = ("timeline_event_fts", "bot_context_chunk_fts")
_READER_ROW_TABLES = frozenset(
    {
        "a2v3_customer_brand_profiles",
        "a2v3_mail_event_facts",
        "bot_context_chunks",
        "customer_id_mappings",
        "customer_identities",
        "customer_objection_summary_v1",
        "customer_objections_v1",
        "customer_opportunities",
        "customer_purchases_v1",
        "derived_signals",
        "event_artifacts",
        "event_child_attribution_v1",
        "family_links_v1",
        "family_members_v1",
        "identity_links",
        "ingestion_cursors",
        # Canonical manager freshness checks read completed import lineage.
        "ingestion_runs",
        "opportunity_child_attribution_v1",
        "schema_migrations",
        "timeline_conflicts",
        "timeline_events",
    }
)
_SCHEMA_ONLY_TECHNICAL_TABLES = {
    "audit_log": "forensic_audit_rows_not_required_by_reader",
    "customer_objection_extraction_runs_v1": "offline_extraction_run_bookkeeping",
    "email_summary_cache_v1": "offline_mail_summary_cache",
    "family_graph_runs_v1": "offline_family_graph_run_bookkeeping",
}
_EMPTY_READER_TABLES = frozenset(_SCHEMA_ONLY_TECHNICAL_TABLES)
_OMITTED_READER_INDEXES = frozenset(
    {
        "ix_bot_context_chunks_active_customer_time",
        "ix_timeline_events_active_customer_time",
    }
)
_OMITTED_READER_INDEX_EVIDENCE = {
    "ix_bot_context_chunks_active_customer_time": {
        "retained_prefix_index": "ix_chunks_customer_event_time",
        "retained_prefix": ["tenant_id", "customer_id", "event_at"],
        "reason": "reader_active_filter_keeps_customer_scoped_prefix",
    },
    "ix_timeline_events_active_customer_time": {
        "retained_prefix_index": "ix_timeline_events_customer_time",
        "retained_prefix": ["tenant_id", "customer_id", "event_at"],
        "reason": "reader_active_filter_keeps_customer_scoped_prefix",
    },
}


def _excluded_from_reader(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + "_") for prefix in _FTS_TABLE_PREFIXES)


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _remove_temporary_sqlite(path: Path) -> None:
    for candidate in (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    ):
        candidate.unlink(missing_ok=True)


def build_compact_reader(source_db: Path, snapshot_db: Path) -> dict[str, object]:
    """Copy all business rows once, omitting forensic audit rows and FTS."""

    if snapshot_db.exists():
        raise PublishSnapshotError(f"snapshot DB already exists: {snapshot_db}")
    snapshot_db.parent.mkdir(parents=True, exist_ok=True)
    source_uri = source_db.resolve(strict=True).as_uri() + "?mode=ro"
    con = sqlite3.connect(str(snapshot_db), timeout=30, uri=True)
    con.row_factory = sqlite3.Row
    omitted_index_sizes: dict[str, dict[str, int]] = {}
    try:
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        con.execute("PRAGMA foreign_keys = OFF")
        con.execute("ATTACH DATABASE ? AS source", (source_uri,))
        placeholders = ",".join("?" for _ in _OMITTED_READER_INDEXES)
        omitted_index_sizes = {
            str(row["name"]): {"bytes": int(row["bytes"]), "pages": int(row["pages"])}
            for row in con.execute(
                "SELECT name,SUM(pgsize) AS bytes,COUNT(*) AS pages FROM dbstat('source') "
                f"WHERE name IN ({placeholders}) GROUP BY name ORDER BY name",
                tuple(sorted(_OMITTED_READER_INDEXES)),
            )
        }
        page_size = int(con.execute("PRAGMA source.page_size").fetchone()[0])
        user_version_value = int(con.execute("PRAGMA source.user_version").fetchone()[0])
        application_id = int(con.execute("PRAGMA source.application_id").fetchone()[0])
        con.execute(f"PRAGMA page_size = {page_size}")
        con.execute("BEGIN IMMEDIATE")

        schema_rows = con.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM source.sqlite_master
            WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1
                       WHEN 'view' THEN 2 WHEN 'trigger' THEN 3 ELSE 4 END,
                     name
            """
        ).fetchall()
        source_tables = {
            str(row["name"])
            for row in schema_rows
            if row["type"] == "table" and not _excluded_from_reader(str(row["name"]))
        }
        known_tables = _READER_ROW_TABLES | _EMPTY_READER_TABLES
        unknown_tables = sorted(source_tables - known_tables)
        missing_policy_tables = sorted(known_tables - source_tables)
        if unknown_tables:
            raise PublishSnapshotError(
                "compact reader table policy rejects unknown tables: "
                + ", ".join(unknown_tables)
            )
        table_rows = [
            row
            for row in schema_rows
            if row["type"] == "table" and not _excluded_from_reader(str(row["name"]))
        ]
        copied_counts: dict[str, dict[str, int]] = {}
        for row in table_rows:
            con.execute(str(row["sql"]))
        for row in table_rows:
            name = str(row["name"])
            source_rows = int(
                con.execute(f"SELECT COUNT(*) FROM source.{_quote_ident(name)}").fetchone()[0]
            )
            if name in _EMPTY_READER_TABLES:
                copied_rows = 0
            else:
                con.execute(
                    f"INSERT INTO main.{_quote_ident(name)} "
                    f"SELECT * FROM source.{_quote_ident(name)}"
                )
                # ``changes()`` is the exact count for the INSERT above.  Reading the
                # destination table again made every large retained table pay for a
                # third full pass (source COUNT, copy, destination COUNT).
                copied_rows = int(con.execute("SELECT changes()").fetchone()[0])
            copied_counts[name] = {"source": source_rows, "snapshot": copied_rows}

        copied_tables = set(copied_counts)
        for row in schema_rows:
            if row["type"] == "table":
                continue
            if _excluded_from_reader(str(row["name"])) or _excluded_from_reader(str(row["tbl_name"])):
                continue
            if row["type"] == "index" and str(row["name"]) in _OMITTED_READER_INDEXES:
                continue
            if row["type"] in {"index", "trigger"} and str(row["tbl_name"]) not in copied_tables:
                continue
            con.execute(str(row["sql"]))
        con.execute(f"PRAGMA user_version = {user_version_value}")
        con.execute(f"PRAGMA application_id = {application_id}")
        con.commit()
        con.execute("DETACH DATABASE source")
    except Exception:
        con.rollback()
        con.close()
        _remove_temporary_sqlite(snapshot_db)
        raise
    else:
        con.close()
    snapshot_db.chmod(0o600)
    source_size = source_db.stat().st_size
    snapshot_size = snapshot_db.stat().st_size
    business_counts_match = all(
        counts["source"] == counts["snapshot"]
        for table, counts in copied_counts.items()
        if table not in _EMPTY_READER_TABLES
    )
    return {
        "profile": "compact_fallback_reader_v3",
        "capabilities": {
            "customer_scoped_search": True,
            "global_search": True,
            "global_search_performance_slo": False,
            "explicit_fts_search": False,
        },
        "business_counts_match": business_counts_match,
        "table_counts": copied_counts,
        "audit_log_source_rows": copied_counts.get("audit_log", {}).get("source", 0),
        "audit_log_snapshot_rows": copied_counts.get("audit_log", {}).get("snapshot", 0),
        "fts_objects_omitted": list(_FTS_TABLE_PREFIXES),
        "table_policy": {
            "mode": "explicit_allowlist_fail_closed",
            "rows_retained": sorted(_READER_ROW_TABLES & source_tables),
            "schema_only_rows_omitted": {
                table: reason
                for table, reason in sorted(_SCHEMA_ONLY_TECHNICAL_TABLES.items())
                if table in source_tables
            },
            "fts_table_prefixes_omitted": {
                prefix: "fts_unavailable_in_compact_fallback_reader"
                for prefix in _FTS_TABLE_PREFIXES
            },
            "required_policy_tables_missing_from_source": missing_policy_tables,
            "unknown_source_tables": unknown_tables,
            "lineage_tables_retained": ["ingestion_cursors", "ingestion_runs"],
        },
        "indexes_omitted": sorted(_OMITTED_READER_INDEXES),
        "indexes_omitted_source_sizes": omitted_index_sizes,
        "indexes_omitted_source_bytes": sum(
            item["bytes"] for item in omitted_index_sizes.values()
        ),
        "index_omission_evidence": _OMITTED_READER_INDEX_EVIDENCE,
        "source_size_bytes": source_size,
        "snapshot_size_bytes": snapshot_size,
        "saved_bytes": source_size - snapshot_size,
        "size_ratio": round(snapshot_size / source_size, 6) if source_size else None,
        "max_size_bytes": COMPACT_READER_MAX_BYTES,
        "within_size_limit": snapshot_size <= COMPACT_READER_MAX_BYTES,
    }


def fallback_search_check(snapshot_db: Path, *, tenant_id: str) -> dict[str, object]:
    with sqlite3.connect(f"file:{snapshot_db}?mode=ro", uri=True, timeout=30) as con:
        row = con.execute(
            """
            SELECT customer_id,
                   COALESCE(NULLIF(subject, ''), NULLIF(text_preview, ''), NULLIF(summary, ''))
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id IS NOT NULL
              AND COALESCE(NULLIF(subject, ''), NULLIF(text_preview, ''), NULLIF(summary, '')) IS NOT NULL
              AND (superseded_by IS NULL OR superseded_by = '')
            ORDER BY event_at DESC, event_id DESC
            LIMIT 1
            """,
            (tenant_id,),
        ).fetchone()
    if row is None:
        return {"ok": False, "reason": "no_timeline_event_search_probe"}
    customer_id, query = str(row[0]), str(row[1])[:96]
    with CustomerTimelineSQLiteStore.open_read_only(snapshot_db, allowed_root=snapshot_db.parent) as store:
        scoped = store.search_timeline(
            tenant_id,
            query,
            customer_id=customer_id,
            scopes=("events",),
            mode="auto",
            include_highlights=False,
            limit=1,
        )
        global_result = store.search_timeline(
            tenant_id,
            query,
            scopes=("events",),
            mode="auto",
            include_highlights=False,
            limit=1,
        )
        fts_enabled = store.open_result.fts_enabled
    backend = str(scoped.get("backend") or "")
    global_backend = str(global_result.get("backend") or "")
    result_count = len(scoped.get("items") or ())
    global_result_count = len(global_result.get("items") or ())
    return {
        "ok": (
            not fts_enabled
            and backend == "fallback_like"
            and result_count == 1
            and global_backend == "fallback_like"
            and global_result_count >= 1
        ),
        "scope": "customer_and_global",
        "backend": backend,
        "fts_enabled": fts_enabled,
        "result_count": result_count,
        "global_backend": global_backend,
        "global_result_count": global_result_count,
        "probe_customer_sha256": hashlib.sha256(customer_id.encode("utf-8")).hexdigest(),
    }


def _domain_integrity(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as con:
        return dict(customer_timeline_integrity_report(con))


def snapshot_table_counts(
    snapshot_db: Path,
    count_tables: tuple[str, ...],
    compaction: Mapping[str, object],
) -> dict[str, int]:
    """Reuse exact copy balances; query only configured objects outside that policy."""

    raw_counts = compaction.get("table_counts")
    copied_counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    counts: dict[str, int] = {}
    unresolved: list[str] = []
    for table in count_tables:
        balance = copied_counts.get(table)
        if isinstance(balance, Mapping) and isinstance(balance.get("snapshot"), int):
            counts[table] = int(balance["snapshot"])
        else:
            unresolved.append(table)
    if unresolved:
        counts.update(table_counts(snapshot_db, tuple(unresolved)))
    return counts


def build_snapshot(
    config_path: Path,
    *,
    execute: bool,
    snapshot_name: str | None = None,
) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "build_snapshot")
    snapshot_name = snapshot_name or "prod_" + report["generated_at"].replace(":", "").replace("+", "Z")
    snapshot_dir = cfg.snapshot_root / snapshot_name
    snapshot_db = snapshot_dir / "customer_timeline.sqlite"
    temporary_db = snapshot_dir / ".customer_timeline.tmp.sqlite"
    report.update({"snapshot_dir": str(snapshot_dir), "snapshot_db": str(snapshot_db), "execute": execute})
    if not execute:
        report["status"] = "dry_run"
        return report, True
    if not cfg.staging_db.is_file():
        report.update({"status": "failed", "error": "staging_database_missing"})
        return report, False
    if snapshot_db.exists() or temporary_db.exists():
        report.update({"status": "failed", "error": "snapshot_destination_exists"})
        return report, False

    writer_head_start = git_head(ROOT)
    writer_status_start_raw = git_status_short(ROOT)
    writer_status_start = classify_publish_worktree_status(writer_status_start_raw)
    if not writer_head_start or writer_status_start["clean_for_publish"] is not True:
        report.update({"status": "failed", "writer_worktree": writer_status_start})
        return report, False

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    lock_timeout = float(cfg.raw.get("lock_timeout_seconds") or 30.0)
    published_snapshot = False
    try:
        with customer_timeline_run_lock(cfg.staging_db, timeout_seconds=lock_timeout) as run_lock:
            with customer_timeline_writer_lock(cfg.staging_db, timeout_seconds=lock_timeout) as writer_lock:
                checkpoint = wal_checkpoint_truncate(cfg.staging_db)
                nightly_gate = nightly_manifest_report(cfg)
                if nightly_gate.get("ok") is not True:
                    report.update({"status": "failed", "nightly_manifest": nightly_gate})
                    return report, False
                source_stat = cfg.staging_db.stat()
                source_identity = {
                    "sha256": nightly_gate.get("staging_sha256"),
                    "size_bytes": nightly_gate.get("staging_size_bytes"),
                    "mtime_ns": source_stat.st_mtime_ns,
                    "user_version": user_version(cfg.staging_db),
                    "run_lock": dict(run_lock),
                    "writer_lock": dict(writer_lock),
                }
                compaction = build_compact_reader(cfg.staging_db, temporary_db)
                source_after = cfg.staging_db.stat()
                source_unchanged = (
                    source_after.st_size == source_stat.st_size
                    and source_after.st_mtime_ns == source_stat.st_mtime_ns
                )

        domain_integrity = _domain_integrity(temporary_db)
        fallback_search = fallback_search_check(temporary_db, tenant_id=cfg.tenant_id)
        reader_smoke_report, reader_smoke_ok = reader_smoke(
            config_path,
            snapshot_db=temporary_db,
        )
        snapshot_quick_check = str(reader_smoke_report.get("quick_check") or "")
        writer_head_end = git_head(ROOT)
        writer_status_end_raw = git_status_short(ROOT)
        writer_status_end = classify_publish_worktree_status(writer_status_end_raw)
        writer_identity_stable = bool(
            writer_head_end == writer_head_start
            and writer_status_end_raw == writer_status_start_raw
            and writer_status_end["clean_for_publish"] is True
        )
        manifest = {
            "schema_version": "customer_timeline_snapshot_build_manifest_v3",
            "built_at": report["generated_at"],
            "package_name": cfg.package_name,
            "writer_git_head": writer_head_start,
            "writer_git_head_end": writer_head_end,
            "writer_worktree": writer_status_start,
            "writer_identity_stable": writer_identity_stable,
            "source_staging_db": str(cfg.staging_db),
            "source_identity": source_identity,
            "source_unchanged_during_copy": source_unchanged,
            "nightly_manifest": nightly_gate,
            "cutoff": nightly_gate.get("published_at"),
            "source_freshness": {
                "source_counts": list(nightly_gate.get("source_counts") or ()),
                "ingestion_cursors": list(nightly_gate.get("ingestion_cursors") or ()),
                "required_sources_check": dict(
                    nightly_gate.get("required_sources_check") or {}
                ),
                "source_degradation": dict(nightly_gate.get("source_degradation") or {}),
            },
            "snapshot_db": str(snapshot_db),
            "sha256": sha256_file(temporary_db),
            "size_bytes": temporary_db.stat().st_size,
            "integrity_check": integrity_check(temporary_db),
            # reader_smoke owns the one quick_check for this build; the formal
            # manifest gate reuses that exact result instead of rescanning 5+ GB.
            "quick_check": snapshot_quick_check,
            "foreign_key_check_rows": len(foreign_key_check(temporary_db)),
            "domain_integrity": domain_integrity,
            "user_version": user_version(temporary_db),
            "counts": snapshot_table_counts(
                temporary_db,
                cfg.count_tables,
                compaction,
            ),
            "compaction": compaction,
            "fallback_search": fallback_search,
            "reader_smoke": reader_smoke_report,
            "bot_visible_stored": (
                reader_smoke_report.get("bot_visibility", {}).get("bot_visible_stored")
            ),
            "bot_visible_after_reader_policy": (
                reader_smoke_report.get("bot_visibility", {}).get(
                    "bot_visible_after_reader_policy"
                )
            ),
            "control_customers": list(cfg.control_customers),
            "live_worktree_untracked": live_worktree_untracked(cfg.readers),
            "wal_checkpoint": checkpoint,
        }
        ok = bool(
            manifest["integrity_check"] == "ok"
            and manifest["quick_check"] == "ok"
            and manifest["foreign_key_check_rows"] == 0
            and customer_timeline_integrity_report_ok(domain_integrity)
            and compaction["business_counts_match"] is True
            and compaction["audit_log_snapshot_rows"] == 0
            and compaction["within_size_limit"] is True
            and manifest["user_version"] == source_identity["user_version"]
            and fallback_search["ok"] is True
            and reader_smoke_ok
            and source_unchanged
            and writer_identity_stable
        )
        if not ok:
            _remove_temporary_sqlite(temporary_db)
            report.update({"status": "failed", "manifest": manifest})
            return report, False
        os.replace(temporary_db, snapshot_db)
        published_snapshot = True
        snapshot_db.chmod(0o600)
        write_json(snapshot_dir / "manifest.json", manifest)
        report.update({"status": "ok", "manifest": manifest})
        return report, True
    except Exception as exc:  # fail closed; only the file created by this build is removed.
        _remove_temporary_sqlite(temporary_db)
        if published_snapshot:
            _remove_temporary_sqlite(snapshot_db)
        report.update({"status": "failed", "error": type(exc).__name__})
        return report, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable compact Customer Timeline snapshot.")
    add_common_args(parser)
    parser.add_argument("--execute", action="store_true", help="Build and verify one compact snapshot.")
    parser.add_argument("--snapshot-name")
    args = parser.parse_args()
    report, ok = build_snapshot(args.config, execute=args.execute, snapshot_name=args.snapshot_name)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
