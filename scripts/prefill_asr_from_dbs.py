from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _discover_dbs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/*.db")
        if path.is_file() and "venv" not in path.parts
    )


def _db_has_call_records(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("select name from sqlite_master where type='table' and name='call_records'")
        row = cur.fetchone()
        conn.close()
        return bool(row)
    except sqlite3.Error:
        return False


def _row_score(row: dict[str, Any]) -> tuple[int, int, str]:
    transcript_text = str(row.get("transcript_text") or "")
    variants = str(row.get("transcript_variants_json") or "")
    updated_at = str(row.get("updated_at") or "")
    return (len(transcript_text), len(variants), updated_at)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse already completed ASR results from existing DBs into a new target DB."
    )
    parser.add_argument("--target-db", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--exclude-db", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target_db = Path(args.target_db).expanduser().resolve()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    excluded = {target_db}
    excluded.update(Path(item).expanduser().resolve() for item in args.exclude_db)

    db_paths = [
        path
        for path in _discover_dbs(workspace_root)
        if path not in excluded and _db_has_call_records(path)
    ]

    target_conn = sqlite3.connect(target_db)
    target_conn.row_factory = sqlite3.Row
    target_cur = target_conn.cursor()
    target_rows = target_cur.execute(
        "select source_file from call_records"
    ).fetchall()
    target_files = {str(row["source_file"] or "") for row in target_rows if str(row["source_file"] or "")}

    best_by_source: dict[str, dict[str, Any]] = {}
    provider_counter: Counter[str] = Counter()

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        local_hits = 0
        for row in cur.execute(
            """
            select source_file,
                   transcribe_attempts,
                   transcript_manager,
                   transcript_client,
                   transcript_text,
                   transcript_variants_json,
                   updated_at
              from call_records
             where transcription_status='done'
            """
        ):
            source_file = str(row["source_file"] or "")
            if not source_file or source_file not in target_files:
                continue
            candidate = {
                "source_file": source_file,
                "transcribe_attempts": int(row["transcribe_attempts"] or 0),
                "transcript_manager": row["transcript_manager"],
                "transcript_client": row["transcript_client"],
                "transcript_text": row["transcript_text"],
                "transcript_variants_json": row["transcript_variants_json"],
                "updated_at": row["updated_at"],
                "db_path": str(db_path),
            }
            existing = best_by_source.get(source_file)
            if existing is None or _row_score(candidate) > _row_score(existing):
                best_by_source[source_file] = candidate
            local_hits += 1
        if local_hits:
            provider_counter[str(db_path)] += local_hits
        conn.close()

    updated = 0
    for source_file, row in best_by_source.items():
        target_cur.execute(
            """
            update call_records
               set transcription_status='done',
                   transcribe_attempts=?,
                   transcript_manager=?,
                   transcript_client=?,
                   transcript_text=?,
                   transcript_variants_json=?,
                   last_error=null,
                   next_retry_at=null,
                   dead_letter_stage=null,
                   updated_at=?
             where source_file=?
            """,
            (
                row["transcribe_attempts"],
                row["transcript_manager"],
                row["transcript_client"],
                row["transcript_text"],
                row["transcript_variants_json"],
                row["updated_at"],
                source_file,
            ),
        )
        updated += int(target_cur.rowcount or 0)
    target_conn.commit()

    target_cur.execute("select transcription_status, count(*) from call_records group by 1 order by 1")
    transcription_counts = target_cur.fetchall()
    target_conn.close()

    result = {
        "target_db": str(target_db),
        "workspace_root": str(workspace_root),
        "scanned_dbs": len(db_paths),
        "reused_asr_rows": updated,
        "transcription_counts": transcription_counts,
        "top_source_dbs": provider_counter.most_common(20),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
