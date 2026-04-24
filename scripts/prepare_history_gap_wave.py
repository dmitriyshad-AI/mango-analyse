from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Resolve+Analyze wave from an ASR DB by selecting only calls that already "
            "have ASR done but still have no Resolve/Analyze result in any overlay DB."
        )
    )
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overlay-db", action="append", default=[])
    return parser.parse_args()


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = "file:" + urllib.parse.quote(str(path)) + "?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _copy_db_snapshot(source_db: Path, out_db: Path) -> None:
    if out_db.exists():
        out_db.unlink()
    src_uri = "file:" + urllib.parse.quote(str(source_db)) + "?mode=ro"
    src = sqlite3.connect(src_uri, uri=True)
    dst = sqlite3.connect(out_db)
    src.backup(dst)
    dst.close()
    src.close()


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _rank_resolve(status: str | None) -> int:
    return {"done": 3, "skipped": 3, "manual": 3, "in_progress": 2, "pending": 1}.get(status or "", 0)


def _rank_analysis(status: str | None) -> int:
    return {"done": 3, "in_progress": 2, "pending": 1}.get(status or "", 0)


def _month_key(value: Any) -> str:
    text = str(value or "")
    return text[:7] if len(text) >= 7 else ""


def main() -> int:
    args = _parse_args()
    source_db = Path(args.source_db).expanduser().resolve()
    overlay_dbs = [Path(item).expanduser().resolve() for item in args.overlay_db]
    out_root = Path(args.out_root).expanduser().resolve()
    out_db = out_root / f"{out_root.name}.db"
    batch_dir = out_root / "batch_llm_wave"
    transcripts_dir = out_root / "transcripts"
    manifest_path = out_root / "selection_manifest.json"

    out_root.mkdir(parents=True, exist_ok=True)
    _clean_output_dir(batch_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    source_conn = _connect_ro(source_db)
    source_rows = source_conn.execute(
        """
        SELECT id, source_file, source_filename, phone, manager_name, started_at, duration_sec,
               transcription_status
          FROM call_records
         ORDER BY started_at ASC, id ASC
        """
    ).fetchall()
    source_conn.close()

    base_rows: dict[str, sqlite3.Row] = {
        str(row["source_filename"]): row
        for row in source_rows
        if str(row["transcription_status"] or "") == "done"
    }

    overlay_statuses: dict[str, dict[str, str | None]] = {
        key: {"resolve_status": None, "analysis_status": None}
        for key in base_rows
    }
    overlap_counter: Counter[str] = Counter()

    for db_path in overlay_dbs:
        if db_path == source_db or not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        local_hits = 0
        for row in conn.execute(
            "SELECT source_filename, resolve_status, analysis_status FROM call_records"
        ):
            name = str(row["source_filename"] or "")
            if name not in overlay_statuses:
                continue
            local = overlay_statuses[name]
            resolve_status = str(row["resolve_status"] or "") or None
            analysis_status = str(row["analysis_status"] or "") or None
            if _rank_resolve(resolve_status) > _rank_resolve(local["resolve_status"]):
                local["resolve_status"] = resolve_status
            if _rank_analysis(analysis_status) > _rank_analysis(local["analysis_status"]):
                local["analysis_status"] = analysis_status
            local_hits += 1
        conn.close()
        if local_hits:
            overlap_counter[str(db_path)] = local_hits

    selected_rows: list[sqlite3.Row] = []
    months: Counter[str] = Counter()
    phones: Counter[str] = Counter()
    managers: Counter[str] = Counter()
    recent_count = 0

    for name, row in base_rows.items():
        resolve_status = overlay_statuses[name]["resolve_status"]
        analysis_status = overlay_statuses[name]["analysis_status"]
        if analysis_status == "done":
            continue
        if analysis_status == "pending":
            continue
        if resolve_status in {"done", "skipped", "manual"}:
            continue
        selected_rows.append(row)
        month = _month_key(row["started_at"])
        if month:
            months[month] += 1
        phone = str(row["phone"] or "").strip()
        if phone:
            phones[phone] += 1
        manager = str(row["manager_name"] or "").strip()
        if manager:
            managers[manager] += 1
        day = str(row["started_at"] or "")[:10]
        if "2026-03-19" <= day <= "2026-04-07":
            recent_count += 1

    if not selected_rows:
        raise RuntimeError("No ASR-only calls without downstream Resolve/Analyze overlap were found")

    _copy_db_snapshot(source_db, out_db)
    conn = sqlite3.connect(out_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    selected_ids = [int(row["id"]) for row in selected_rows]
    placeholders = ",".join("?" for _ in selected_ids)
    cur.execute(f"DELETE FROM call_records WHERE id NOT IN ({placeholders})", selected_ids)
    cur.execute(
        """
        UPDATE call_records
           SET resolve_status = 'pending',
               analysis_status = 'pending',
               sync_status = 'pending',
               resolve_attempts = 0,
               analyze_attempts = 0,
               sync_attempts = 0,
               resolve_json = NULL,
               resolve_quality_score = NULL,
               analysis_json = NULL,
               next_retry_at = NULL,
               dead_letter_stage = NULL,
               pipeline_stage = NULL,
               pipeline_worker_id = NULL,
               pipeline_claimed_at = NULL,
               analysis_worker_id = NULL,
               analysis_claimed_at = NULL,
               last_error = NULL,
               updated_at = ?
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()

    kept_rows = cur.execute(
        """
        SELECT id, source_file, source_filename, phone, manager_name, started_at, duration_sec
          FROM call_records
         ORDER BY started_at ASC, id ASC
        """
    ).fetchall()
    for row in kept_rows:
        src = Path(str(row["source_file"]))
        dst = batch_dir / str(row["source_filename"])
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    counts = {
        "total_calls": cur.execute("SELECT COUNT(*) FROM call_records").fetchone()[0],
        "transcription_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE transcription_status='done'").fetchone()[0],
        "resolve_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='pending'").fetchone()[0],
        "analysis_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE analysis_status='pending'").fetchone()[0],
    }
    conn.close()

    started_values = [str(row["started_at"]) for row in selected_rows if row["started_at"]]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(source_db),
        "overlay_dbs": [str(path) for path in overlay_dbs],
        "out_db": str(out_db),
        "batch_dir": str(batch_dir),
        "transcripts_dir": str(transcripts_dir),
        "counts": counts,
        "selected_calls": len(selected_rows),
        "selected_unique_phones": len(phones),
        "date_min": min(started_values) if started_values else None,
        "date_max": max(started_values) if started_values else None,
        "fresh_period_count_2026_03_19_to_2026_04_07": recent_count,
        "top_months": months.most_common(12),
        "top_phones": phones.most_common(20),
        "top_managers": managers.most_common(20),
        "overlap_top_dbs": overlap_counter.most_common(20),
        "items": [
            {
                "id": int(row["id"]),
                "source_filename": str(row["source_filename"]),
                "source_file": str(row["source_file"]),
                "phone": row["phone"],
                "manager_name": row["manager_name"],
                "started_at": row["started_at"],
                "duration_sec": row["duration_sec"],
            }
            for row in kept_rows
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
