from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CORE_CATEGORIES = {"run_sales", "run_service", "run_existing_client_progress"}


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a focused Resolve+Analyze wave from recommendation CSV by choosing top-N phones "
            "with the largest pending tail, then keeping only selected categories plus already-done rows."
        )
    )
    parser.add_argument("--recommendations-csv", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--top-contact-count", type=int, default=20)
    parser.add_argument(
        "--include-category",
        action="append",
        default=[],
        help="Wave categories to keep. Defaults to run_sales/run_service/run_existing_client_progress.",
    )
    args = parser.parse_args()

    recommendations_csv = Path(args.recommendations_csv).expanduser().resolve()
    source_db = Path(args.source_db).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    include_categories = set(args.include_category or CORE_CATEGORIES)

    rows = list(csv.DictReader(recommendations_csv.open(encoding="utf-8")))
    pending_rows = [row for row in rows if row.get("analysis_status") == "pending"]
    phone_counts = Counter(str(row.get("phone") or "").strip() for row in pending_rows if str(row.get("phone") or "").strip())
    chosen_phones = [phone for phone, _ in phone_counts.most_common(max(1, int(args.top_contact_count)))]
    chosen_phone_set = set(chosen_phones)

    selected_rows = [
        row
        for row in rows
        if str(row.get("phone") or "").strip() in chosen_phone_set
        and (
            row.get("analysis_status") == "done"
            or row.get("wave_category") in include_categories
        )
    ]
    selected_source_files = {str(row.get("source_filename") or "").strip() for row in selected_rows}
    done_reused = sum(1 for row in selected_rows if row.get("analysis_status") == "done")
    pending_selected = [row for row in selected_rows if row.get("analysis_status") == "pending"]

    out_root.mkdir(parents=True, exist_ok=True)
    out_db = out_root / f"top_{len(chosen_phones)}_llm_wave.db"
    if out_db.exists():
        out_db.unlink()
    shutil.copy2(source_db, out_db)

    conn = sqlite3.connect(out_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in selected_source_files) or "''"
    cur.execute(f"DELETE FROM call_records WHERE source_filename NOT IN ({placeholders})", tuple(selected_source_files))
    cur.execute(
        """
        UPDATE call_records
           SET pipeline_stage = NULL,
               pipeline_worker_id = NULL,
               pipeline_claimed_at = NULL,
               analysis_worker_id = NULL,
               analysis_claimed_at = NULL,
               updated_at = ?
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()

    batch_dir = out_root / "batch_llm_wave"
    _clean_output_dir(batch_dir)
    db_rows = cur.execute(
        """
        SELECT source_file, source_filename, phone, started_at, resolve_status, analysis_status
          FROM call_records
         ORDER BY started_at DESC, id DESC
        """
    ).fetchall()
    for row in db_rows:
        src = Path(str(row["source_file"]))
        dst = batch_dir / str(row["source_filename"])
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    counts = {
        "total_calls": cur.execute("SELECT COUNT(*) FROM call_records").fetchone()[0],
        "transcription_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE transcription_status='done'").fetchone()[0],
        "resolve_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='done'").fetchone()[0],
        "resolve_skipped": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='skipped'").fetchone()[0],
        "resolve_manual": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='manual'").fetchone()[0],
        "resolve_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='pending'").fetchone()[0],
        "analysis_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE analysis_status='done'").fetchone()[0],
        "analysis_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE analysis_status='pending'").fetchone()[0],
    }
    conn.close()

    category_counts = Counter(row.get("wave_category") for row in pending_selected)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommendations_csv": str(recommendations_csv),
        "source_db": str(source_db),
        "out_db": str(out_db),
        "batch_dir": str(batch_dir),
        "top_contact_count": len(chosen_phones),
        "selected_phones": chosen_phones,
        "include_categories": sorted(include_categories),
        "counts": counts,
        "reused_done_calls": done_reused,
        "pending_selected_counts": dict(category_counts),
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
