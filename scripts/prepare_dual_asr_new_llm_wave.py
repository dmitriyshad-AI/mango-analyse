from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Resolve+Analyze wave from a live ASR DB by selecting only calls with "
            "dual-ASR ready and excluding files that were already resolved/analyzed in other DBs."
        )
    )
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


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
        cur.execute("select 1 from sqlite_master where type='table' and name='call_records'")
        ok = cur.fetchone() is not None
        conn.close()
        return ok
    except sqlite3.Error:
        return False


def _dual_asr_ready(variants_json: str | None) -> bool:
    if not variants_json or not str(variants_json).strip():
        return False
    try:
        payload = json.loads(variants_json)
    except Exception:
        return False
    for section in ("manager", "client", "full"):
        part = payload.get(section)
        if isinstance(part, dict) and str(part.get("variant_b") or "").strip():
            return True
    return False


def _copy_db_snapshot(source_db: Path, out_db: Path) -> None:
    if out_db.exists():
        out_db.unlink()
    source_uri = "file:" + urllib.parse.quote(str(source_db)) + "?mode=ro"
    src = sqlite3.connect(source_uri, uri=True)
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


def main() -> int:
    args = _parse_args()
    source_db = Path(args.source_db).expanduser().resolve()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_db = out_root / f"{out_root.name}.db"
    batch_dir = out_root / "batch_llm_wave"
    transcripts_dir = out_root / "transcripts"
    manifest_path = out_root / "selection_manifest.json"

    out_root.mkdir(parents=True, exist_ok=True)
    _clean_output_dir(batch_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    source_uri = "file:" + urllib.parse.quote(str(source_db)) + "?immutable=1"
    source_conn = sqlite3.connect(source_uri, uri=True)
    source_conn.row_factory = sqlite3.Row
    source_cur = source_conn.cursor()
    source_rows = source_cur.execute(
        """
        SELECT id, source_file, source_filename, phone, started_at, duration_sec,
               transcription_status, transcript_variants_json
          FROM call_records
         ORDER BY started_at ASC, id ASC
        """
    ).fetchall()
    source_conn.close()

    dual_ready: dict[str, sqlite3.Row] = {}
    for row in source_rows:
        if str(row["transcription_status"] or "") != "done":
            continue
        if not _dual_asr_ready(row["transcript_variants_json"]):
            continue
        dual_ready[str(row["source_filename"])] = row

    overlap_counter: Counter[str] = Counter()
    prior_terminal: set[str] = set()
    for db_path in _discover_dbs(workspace_root):
        if db_path == source_db or not _db_has_call_records(db_path):
            continue
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        local_hits = 0
        for source_file, resolve_status, analysis_status in cur.execute(
            """
            SELECT source_file, resolve_status, analysis_status
              FROM call_records
             WHERE source_file IS NOT NULL
            """
        ):
            name = Path(str(source_file)).name
            if name not in dual_ready:
                continue
            if resolve_status in ("done", "skipped", "manual") or analysis_status == "done":
                prior_terminal.add(name)
                local_hits += 1
        if local_hits:
            overlap_counter[str(db_path)] += local_hits
        conn.close()

    selected_rows = [
        dual_ready[name]
        for name in dual_ready
        if name not in prior_terminal
    ]
    if not selected_rows:
        raise RuntimeError("No dual-ASR-ready calls without prior Resolve/Analyze overlap were found")

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
               amocrm_contact_id = NULL,
               amocrm_lead_id = NULL,
               updated_at = ?
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()

    kept_rows = cur.execute(
        """
        SELECT source_file, source_filename, phone, started_at, duration_sec
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

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(source_db),
        "out_db": str(out_db),
        "batch_dir": str(batch_dir),
        "transcripts_dir": str(transcripts_dir),
        "counts": counts,
        "dual_asr_ready_in_source": len(dual_ready),
        "prior_resolve_or_analyze_overlap": len(prior_terminal),
        "selected_for_new_llm_wave": len(selected_rows),
        "overlap_top_dbs": overlap_counter.most_common(20),
        "items": [
            {
                "source_filename": str(row["source_filename"]),
                "source_file": str(row["source_file"]),
                "phone": row["phone"],
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
