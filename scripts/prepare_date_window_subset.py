from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a date-window subset DB from an existing ASR-ready SQLite DB."
    )
    parser.add_argument("--source-db", required=True, help="Source SQLite DB with call_records")
    parser.add_argument("--out-root", required=True, help="Output directory for subset DB + manifest")
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_db = Path(args.source_db).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    subset_db = out_root / f"{out_root.name}.db"
    batch_dir = out_root / "batch_recent_window"
    transcripts_dir = out_root / "transcripts"
    manifest_path = out_root / "selection_manifest.json"

    out_root.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    if subset_db.exists():
        subset_db.unlink()
    shutil.copy2(source_db, subset_db)

    conn = sqlite3.connect(subset_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    selected = cur.execute(
        """
        SELECT id, source_file, source_filename, phone, manager_name, started_at, duration_sec
        FROM call_records
        WHERE transcription_status = 'done'
          AND date(started_at) BETWEEN ? AND ?
        ORDER BY started_at ASC, id ASC
        """,
        (args.start_date, args.end_date),
    ).fetchall()

    selected_ids = [int(row["id"]) for row in selected]
    if not selected_ids:
        raise RuntimeError(
            f"No calls found in {source_db} for date window {args.start_date}..{args.end_date}"
        )

    placeholders = ",".join("?" for _ in selected_ids)
    cur.execute(
        f"DELETE FROM call_records WHERE id NOT IN ({placeholders})",
        selected_ids,
    )
    cur.execute(
        """
        UPDATE call_records
        SET
            resolve_status = 'pending',
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
            amocrm_contact_id = NULL,
            amocrm_lead_id = NULL
        """
    )
    conn.commit()

    items: list[dict[str, object]] = []
    for row in selected:
        source_file = Path(str(row["source_file"])).expanduser()
        link_path = batch_dir / str(row["source_filename"])
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_file)
        items.append(
            {
                "id": int(row["id"]),
                "started_at": row["started_at"],
                "source_filename": row["source_filename"],
                "source_file": str(source_file),
                "phone": row["phone"],
                "manager_name": row["manager_name"],
                "duration_sec": row["duration_sec"],
            }
        )

    manifest = {
        "source_db": str(source_db),
        "subset_db": str(subset_db),
        "date_window": {
            "start": args.start_date,
            "end": args.end_date,
        },
        "selected_calls": len(items),
        "batch_dir": str(batch_dir),
        "transcripts_dir": str(transcripts_dir),
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "subset_db": str(subset_db),
                "batch_dir": str(batch_dir),
                "transcripts_dir": str(transcripts_dir),
                "manifest": str(manifest_path),
                "selected_calls": len(items),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
