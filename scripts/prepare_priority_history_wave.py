from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.config import get_settings
from mango_mvp.services.analyze import AnalyzeService


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()


def _load_priority_items(path: Path, top_n: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload.get("items") or [])
    if top_n > 0:
        items = items[:top_n]
    return items


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or "", re.U))


def _classify_for_wave(
    service: AnalyzeService,
    *,
    transcript_text: str,
    duration_sec: float | None,
    already_analyzed_recent: bool,
) -> tuple[str, str]:
    if already_analyzed_recent:
        return "already_analyzed_recent", "reuse_existing_analysis"

    text = (transcript_text or "").strip()
    heuristic_call_type = service._detect_call_type(text)
    substantial = service._has_substantial_dialogue(text)
    sales_signal = service._has_meaningful_sales_signal(text)
    duration = float(duration_sec or 0.0)

    if heuristic_call_type == "non_conversation":
        return heuristic_call_type, "skip_non_conversation"

    if heuristic_call_type == "technical_call":
        if substantial or duration >= 60 or sales_signal:
            return heuristic_call_type, "optional_technical"
        return heuristic_call_type, "skip_short_technical"

    if heuristic_call_type == "service_call":
        if substantial or duration >= 20 or sales_signal:
            return heuristic_call_type, "run_service"
        return heuristic_call_type, "optional_service_short"

    if heuristic_call_type == "existing_client_progress":
        return heuristic_call_type, "run_existing_client_progress"

    if heuristic_call_type == "sales_call":
        return heuristic_call_type, "run_sales"

    return heuristic_call_type, "run_service"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a top-N priority contact history wave from an ASR-only DB, "
            "reuse already analyzed recent-window rows when available, and estimate "
            "how much of the wave is worth the next Resolve+Analyze pass."
        )
    )
    parser.add_argument("--priority-json", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--fresh-db", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    priority_json = Path(args.priority_json).expanduser().resolve()
    source_db = Path(args.source_db).expanduser().resolve()
    fresh_db = Path(args.fresh_db).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    items = _load_priority_items(priority_json, args.top_n)
    if not items:
        raise SystemExit("No priority contacts found in priority JSON")

    phones = [str(item.get("phone") or "").strip() for item in items if str(item.get("phone") or "").strip()]
    phone_set = set(phones)
    rank_map = {phone: index + 1 for index, phone in enumerate(phones)}

    out_root.mkdir(parents=True, exist_ok=True)
    out_db = out_root / f"top_{len(phones)}_priority_history_wave.db"
    if out_db.exists():
        out_db.unlink()
    shutil.copy2(source_db, out_db)

    conn = sqlite3.connect(out_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in phones)
    cur.execute(f"DELETE FROM call_records WHERE phone NOT IN ({placeholders})", phones)
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

    fresh_conn = sqlite3.connect(fresh_db)
    fresh_conn.row_factory = sqlite3.Row
    fresh_cur = fresh_conn.cursor()
    fresh_rows = fresh_cur.execute(
        """
        SELECT source_file,
               resolve_status,
               resolve_attempts,
               resolve_json,
               resolve_quality_score,
               analysis_status,
               analyze_attempts,
               analysis_json,
               last_error,
               updated_at
          FROM call_records
         WHERE phone IN ({})
           AND (
                resolve_status IN ('done', 'skipped', 'manual')
                OR analysis_status = 'done'
           )
        """.format(placeholders),
        phones,
    ).fetchall()
    reuse_by_source = {str(row["source_file"]): row for row in fresh_rows}

    for source_file, row in reuse_by_source.items():
        cur.execute(
            """
            UPDATE call_records
               SET resolve_status = ?,
                   resolve_attempts = ?,
                   resolve_json = ?,
                   resolve_quality_score = ?,
                   analysis_status = ?,
                   analyze_attempts = ?,
                   analysis_json = ?,
                   last_error = ?,
                   updated_at = ?,
                   analysis_worker_id = NULL,
                   analysis_claimed_at = NULL,
                   pipeline_stage = NULL,
                   pipeline_worker_id = NULL,
                   pipeline_claimed_at = NULL
             WHERE source_file = ?
            """,
            (
                row["resolve_status"],
                row["resolve_attempts"],
                row["resolve_json"],
                row["resolve_quality_score"],
                row["analysis_status"],
                row["analyze_attempts"],
                row["analysis_json"],
                row["last_error"],
                row["updated_at"],
                source_file,
            ),
        )
    conn.commit()
    fresh_conn.close()

    batch_dir = out_root / "batch_top_priority_history"
    _clean_output_dir(batch_dir)
    selected_rows = cur.execute(
        """
        SELECT source_file, source_filename, phone, started_at, duration_sec, transcript_text,
               resolve_status, analysis_status
          FROM call_records
         ORDER BY started_at DESC, id DESC
        """
    ).fetchall()

    for row in selected_rows:
        source_file = Path(str(row["source_file"]))
        link_path = batch_dir / str(row["source_filename"])
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_file)

    settings = get_settings()
    analyzer = AnalyzeService(settings)

    recommendation_rows: list[dict[str, Any]] = []
    recommendation_counter: Counter[str] = Counter()
    heuristic_counter: Counter[str] = Counter()
    date_values: list[str] = []
    reused_analysis_count = 0

    for row in selected_rows:
        source_file = str(row["source_file"])
        phone = str(row["phone"] or "").strip()
        started_at = str(row["started_at"] or "")
        transcript_text = str(row["transcript_text"] or "")
        duration_sec = float(row["duration_sec"] or 0.0)
        already_analyzed_recent = str(row["analysis_status"] or "") == "done" and source_file in reuse_by_source
        if already_analyzed_recent:
            reused_analysis_count += 1
        heuristic_call_type, wave_category = _classify_for_wave(
            analyzer,
            transcript_text=transcript_text,
            duration_sec=duration_sec,
            already_analyzed_recent=already_analyzed_recent,
        )
        heuristic_counter[heuristic_call_type] += 1
        recommendation_counter[wave_category] += 1
        if started_at:
            date_values.append(started_at[:10])
        recommendation_rows.append(
            {
                "phone_rank": rank_map.get(phone),
                "phone": phone,
                "started_at": started_at,
                "source_filename": str(row["source_filename"]),
                "duration_sec": round(duration_sec, 3),
                "word_count": _word_count(transcript_text),
                "heuristic_call_type": heuristic_call_type,
                "wave_category": wave_category,
                "resolve_status": str(row["resolve_status"] or ""),
                "analysis_status": str(row["analysis_status"] or ""),
                "reused_recent_analysis": already_analyzed_recent,
            }
        )

    conn_counts = {
        "total_calls": cur.execute("SELECT COUNT(*) FROM call_records").fetchone()[0],
        "transcription_done": cur.execute(
            "SELECT COUNT(*) FROM call_records WHERE transcription_status='done'"
        ).fetchone()[0],
        "resolve_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='done'").fetchone()[0],
        "resolve_skipped": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='skipped'").fetchone()[0],
        "resolve_manual": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='manual'").fetchone()[0],
        "resolve_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE resolve_status='pending'").fetchone()[0],
        "analysis_done": cur.execute("SELECT COUNT(*) FROM call_records WHERE analysis_status='done'").fetchone()[0],
        "analysis_pending": cur.execute("SELECT COUNT(*) FROM call_records WHERE analysis_status='pending'").fetchone()[0],
    }
    conn.close()

    top_contacts_csv = out_root / "top_priority_contacts.csv"
    with top_contacts_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "priority_rank",
                "phone",
                "latest_id",
                "latest_started_at",
                "latest_manager_name",
                "lead_priority",
                "follow_up_score",
                "call_type",
                "next_step_action",
                "next_step_due",
                "recommended_product",
                "objections",
                "fresh_calls_count",
                "history_calls_count",
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(items, 1):
            writer.writerow(
                {
                    "priority_rank": idx,
                    "phone": item.get("phone"),
                    "latest_id": item.get("latest_id"),
                    "latest_started_at": item.get("latest_started_at"),
                    "latest_manager_name": item.get("latest_manager_name"),
                    "lead_priority": item.get("lead_priority"),
                    "follow_up_score": item.get("follow_up_score"),
                    "call_type": item.get("call_type"),
                    "next_step_action": item.get("next_step_action"),
                    "next_step_due": item.get("next_step_due"),
                    "recommended_product": item.get("recommended_product"),
                    "objections": item.get("objections"),
                    "fresh_calls_count": item.get("fresh_calls_count"),
                    "history_calls_count": item.get("history_calls_count"),
                }
            )

    recommendations_csv = out_root / "llm_wave_recommendations.csv"
    with recommendations_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "phone_rank",
                "phone",
                "started_at",
                "source_filename",
                "duration_sec",
                "word_count",
                "heuristic_call_type",
                "wave_category",
                "resolve_status",
                "analysis_status",
                "reused_recent_analysis",
            ],
        )
        writer.writeheader()
        writer.writerows(recommendation_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority_json": str(priority_json),
        "source_db": str(source_db),
        "fresh_db": str(fresh_db),
        "top_n_contacts": len(phones),
        "phones": phones,
        "out_db": str(out_db),
        "batch_dir": str(batch_dir),
        "top_contacts_csv": str(top_contacts_csv),
        "recommendations_csv": str(recommendations_csv),
        "date_min": min(date_values) if date_values else None,
        "date_max": max(date_values) if date_values else None,
        "counts": conn_counts,
        "reused_recent_analysis_count": reused_analysis_count,
        "heuristic_call_type_counts": dict(heuristic_counter),
        "wave_category_counts": dict(recommendation_counter),
        "recommended_next_wave": {
            "run_now": recommendation_counter["run_sales"]
            + recommendation_counter["run_service"]
            + recommendation_counter["run_existing_client_progress"],
            "optional_after_run_now": recommendation_counter["optional_technical"]
            + recommendation_counter["optional_service_short"],
            "skip_for_now": recommendation_counter["skip_non_conversation"]
            + recommendation_counter["skip_short_technical"],
            "reuse_existing_analysis": recommendation_counter["reuse_existing_analysis"],
        },
    }
    manifest_path = out_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
