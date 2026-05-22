from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OTHER_DBS: list[str] = []


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def rank_resolve(status: Any) -> int:
    return {
        "done": 3,
        "skipped": 3,
        "manual": 3,
        "in_progress": 2,
        "pending": 1,
        None: 0,
        "": 0,
    }.get(status, 0)


def rank_analysis(status: Any) -> int:
    return {
        "done": 3,
        "in_progress": 2,
        "pending": 1,
        None: 0,
        "": 0,
    }.get(status, 0)


def is_missing_secondary(payload: dict[str, Any]) -> bool:
    mode = str(payload.get("mode") or "").strip()
    if mode == "stereo":
        manager = payload.get("manager") or {}
        client = payload.get("client") or {}
        return not str(manager.get("variant_b") or "").strip() or not str(
            client.get("variant_b") or ""
        ).strip()
    if mode == "mono_or_fallback":
        full = payload.get("full") or {}
        return not str(full.get("variant_b") or "").strip()
    return False


def build_schema(src: sqlite3.Connection, dest: sqlite3.Connection) -> None:
    rows = src.execute(
        """
        SELECT type, name, sql
          FROM sqlite_master
         WHERE sql IS NOT NULL
           AND type IN ('table', 'index')
           AND name NOT LIKE 'sqlite_%'
         ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
        """
    ).fetchall()
    for row in rows:
        dest.execute(str(row["sql"]))
    dest.commit()


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    payload_raw = row.get("transcript_variants_json")
    if isinstance(payload_raw, str) and payload_raw.strip():
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            payload.pop("secondary_backfill_meta", None)
            row["transcript_variants_json"] = json.dumps(payload, ensure_ascii=False)

    row["pipeline_stage"] = None
    row["pipeline_worker_id"] = None
    row["pipeline_claimed_at"] = None
    row["analysis_worker_id"] = None
    row["analysis_claimed_at"] = None
    row["next_retry_at"] = None
    row["dead_letter_stage"] = None
    last_error = row.get("last_error")
    if isinstance(last_error, str) and last_error.startswith("backfill-second-asr:"):
        row["last_error"] = None
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--other-db", action="append", default=[])
    args = parser.parse_args()

    source_db = Path(args.source_db)
    output_root = Path(args.output_root)
    other_dbs = args.other_db or DEFAULT_OTHER_DBS

    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = output_root / "batch_gigaam_useful"
    batch_dir.mkdir(parents=True, exist_ok=True)
    subset_db = output_root / f"{output_root.name}.db"
    if subset_db.exists():
        subset_db.unlink()

    src = connect(str(source_db))
    rows = src.execute("SELECT * FROM call_records WHERE transcription_status='done'").fetchall()
    missing_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        payload_raw = row_dict.get("transcript_variants_json")
        if not isinstance(payload_raw, str) or not payload_raw.strip():
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if is_missing_secondary(payload):
            missing_rows[str(row_dict["source_filename"])] = row_dict

    best: dict[str, dict[str, Any]] = {
        name: {"resolve_status": None, "analysis_status": None} for name in missing_rows
    }
    for path in other_dbs:
        conn = connect(path)
        for row in conn.execute(
            "SELECT source_filename, resolve_status, analysis_status FROM call_records"
        ):
            name = str(row["source_filename"])
            if name not in best:
                continue
            if rank_resolve(row["resolve_status"]) > rank_resolve(best[name]["resolve_status"]):
                best[name]["resolve_status"] = row["resolve_status"]
            if rank_analysis(row["analysis_status"]) > rank_analysis(best[name]["analysis_status"]):
                best[name]["analysis_status"] = row["analysis_status"]
        conn.close()

    selected: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()
    for name, row in missing_rows.items():
        resolve_status = best[name]["resolve_status"]
        analysis_status = best[name]["analysis_status"]
        if analysis_status == "done":
            excluded_counter["already_analyzed_elsewhere"] += 1
            continue
        reason = (
            "resolve_done_analyze_pending_elsewhere"
            if (analysis_status == "pending" or resolve_status in {"done", "skipped", "manual"})
            else "asr_only_no_resolve_elsewhere"
        )
        reason_counter[reason] += 1
        enriched = sanitize_row(dict(row))
        enriched["_reason"] = reason
        enriched["_best_resolve_status"] = resolve_status
        enriched["_best_analysis_status"] = analysis_status
        selected.append(enriched)

    selected.sort(key=lambda item: (str(item.get("started_at") or ""), str(item["source_filename"])))

    dest = connect(str(subset_db))
    build_schema(src, dest)
    src.close()

    columns = [row[1] for row in dest.execute("PRAGMA table_info(call_records)").fetchall()]
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO call_records ({', '.join(columns)}) VALUES ({placeholders})"
    for row in selected:
        values = [row.get(column) for column in columns]
        dest.execute(insert_sql, values)
    dest.commit()
    dest.close()

    for row in selected:
        source_file = Path(str(row["source_file"]))
        target = batch_dir / source_file.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source_file)

    manifest = {
        "source_db": str(source_db),
        "output_db": str(subset_db),
        "batch_dir": str(batch_dir),
        "selected_calls": len(selected),
        "selected_reasons": dict(reason_counter),
        "excluded_reasons": dict(excluded_counter),
        "other_dbs": list(other_dbs),
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_root / "selected_calls.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "source_filename",
                "source_file",
                "started_at",
                "phone",
                "manager_name",
                "reason",
                "best_resolve_status_elsewhere",
                "best_analysis_status_elsewhere",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "id": row["id"],
                    "source_filename": row["source_filename"],
                    "source_file": row["source_file"],
                    "started_at": row["started_at"],
                    "phone": row.get("phone"),
                    "manager_name": row.get("manager_name"),
                    "reason": row["_reason"],
                    "best_resolve_status_elsewhere": row["_best_resolve_status"],
                    "best_analysis_status_elsewhere": row["_best_analysis_status"],
                }
            )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
