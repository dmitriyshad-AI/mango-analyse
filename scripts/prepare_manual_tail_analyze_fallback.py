#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now_sql() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def _copy_schema(source_db: Path, target_db: Path) -> list[str]:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()
    with _connect_ro(source_db) as src, sqlite3.connect(target_db) as dst:
        rows = src.execute(
            """
            select type, name, sql
              from sqlite_master
             where tbl_name = 'call_records'
               and sql is not null
               and type in ('table', 'index')
             order by case type when 'table' then 0 else 1 end, name
            """
        ).fetchall()
        if not rows:
            raise RuntimeError(f"Source DB has no call_records schema: {source_db}")
        for _, _, sql in rows:
            dst.execute(sql)
        dst.commit()
        return [row[1] for row in dst.execute("pragma table_info(call_records)")]


def _source_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("pragma table_info(call_records)")}


def _resolve_fallback_payload(row: sqlite3.Row, source_db: Path) -> str:
    original_payload: dict[str, Any] = {}
    raw = str(row["resolve_json"] or "").strip() if "resolve_json" in row.keys() else ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                original_payload = parsed
        except json.JSONDecodeError:
            original_payload = {"raw_unparsed": raw[:2000]}
    payload = {
        "version": "v1",
        "decision": "manual_resolve_fallback_analyze_raw_transcript",
        "reason": (
            "Resolve could not confidently accept speaker merge; keeping terminal "
            "resolve_status='skipped' so Analyze can process raw transcript_text."
        ),
        "source_db": str(source_db),
        "source_id": int(row["id"]),
        "source_resolve_status": row["resolve_status"],
        "source_analysis_status": row["analysis_status"],
        "source_resolve_quality_score": row["resolve_quality_score"],
        "original_resolve_decision": original_payload.get("decision"),
        "ts_utc": _iso_now(),
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a safe Analyze fallback DB for calls stuck in resolve_status=manual. "
            "Rows are copied from the source DB; original DB is not modified."
        )
    )
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--manual-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_db = Path(args.source_db).expanduser().resolve()
    manual_list = Path(args.manual_list).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    target_db = out_root / f"{out_root.name}.db"
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty, use --force: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    names = _read_names(manual_list)
    if not names:
        raise SystemExit(f"Manual list is empty: {manual_list}")

    columns = _copy_schema(source_db, target_db)
    insert_columns = [column for column in columns if column != "id"]
    placeholders = ", ".join("?" for _ in insert_columns)
    insert_sql = f"insert into call_records ({', '.join(insert_columns)}) values ({placeholders})"
    now = _iso_now_sql()

    rows_out: list[dict[str, Any]] = []
    with _connect_ro(source_db) as src, sqlite3.connect(target_db) as dst:
        src.row_factory = sqlite3.Row
        src_cols = _source_columns(src)
        select_cols = [column for column in columns if column in src_cols]
        select_sql = (
            f"select {', '.join(select_cols)} from call_records "
            f"where source_filename in ({', '.join('?' for _ in names)}) "
            "and transcription_status='done' and resolve_status='manual' "
            "and analysis_status='pending' order by source_filename asc, id asc"
        )
        seen: set[str] = set()
        for row in src.execute(select_sql, names):
            source_filename = str(row["source_filename"] or "").strip()
            if not source_filename or source_filename in seen:
                continue
            seen.add(source_filename)
            values = {column: row[column] if column in row.keys() else None for column in insert_columns}
            values.update(
                {
                    "transcription_status": "done",
                    "resolve_status": "skipped",
                    "analysis_status": "pending",
                    "sync_status": "pending",
                    "resolve_attempts": int(row["resolve_attempts"] or 0),
                    "analyze_attempts": 0,
                    "sync_attempts": 0,
                    "pipeline_stage": None,
                    "pipeline_worker_id": None,
                    "pipeline_claimed_at": None,
                    "analysis_worker_id": None,
                    "analysis_claimed_at": None,
                    "next_retry_at": None,
                    "dead_letter_stage": None,
                    "last_error": None,
                    "resolve_json": _resolve_fallback_payload(row, source_db),
                    "resolve_quality_score": row["resolve_quality_score"],
                    "analysis_json": None,
                    "updated_at": now,
                }
            )
            if not values.get("created_at"):
                values["created_at"] = now
            dst.execute(insert_sql, [values.get(column) for column in insert_columns])
            rows_out.append(
                {
                    "source_filename": source_filename,
                    "source_id": int(row["id"]),
                    "duration_sec": float(row["duration_sec"] or 0.0),
                    "manager_name": row["manager_name"],
                    "phone": row["phone"],
                    "text_len": len(str(row["transcript_text"] or "")),
                    "source_resolve_quality_score": row["resolve_quality_score"],
                }
            )
        dst.commit()

    missing = [name for name in names if name not in {row["source_filename"] for row in rows_out}]
    selected_path = out_root / "selected_calls.tsv"
    with selected_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "source_filename",
            "source_id",
            "duration_sec",
            "manager_name",
            "phone",
            "text_len",
            "source_resolve_quality_score",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    manifest = {
        "generated_at": _iso_now(),
        "source_db": str(source_db),
        "manual_list": str(manual_list),
        "out_root": str(out_root),
        "db_path": str(target_db),
        "manual_list_count": len(names),
        "selected_calls": len(rows_out),
        "missing_from_source_db": len(missing),
        "missing": missing,
        "selected_calls_tsv": str(selected_path),
        "status_policy": {
            "transcription_status": "done",
            "resolve_status": "skipped",
            "analysis_status": "pending",
            "purpose": "Analyze raw transcript_text for manual resolve tails without mutating original DB.",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
