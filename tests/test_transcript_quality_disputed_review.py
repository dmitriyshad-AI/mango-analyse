from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.quality.transcript_quality_disputed_review import (
    DisputedReviewConfig,
    build_transcript_quality_disputed_review,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(call_id: int, *, call_type: str, contentful: bool, label: str, force: bool, manual: bool) -> dict[str, object]:
    return {
        "id": call_id,
        "source_filename": f"call_{call_id}.mp3",
        "started_at": "2026-01-02 10:00:00",
        "month": "2026-01",
        "phone": "+79990000000",
        "manager_name": "Manager",
        "duration_sec": "30",
        "transcription_status": "done",
        "resolve_status": "done",
        "analysis_status": "done",
        "analysis_json_status": "ok",
        "current_call_type": call_type,
        "current_contentful": contentful,
        "guardrail_label": label,
        "guardrail_score": "-5",
        "guardrail_reason_codes": "system_no_dialogue_phrase|no_live_marker",
        "should_force_non_conversation": force,
        "requires_manual_review": manual,
        "protected_live_dialogue": False,
        "recommended_call_type": "non_conversation",
        "recommended_contentful": False,
        "next_step": "",
        "products": "",
        "subjects": "",
        "objections": "",
        "history_summary_excerpt": "",
        "transcript_excerpt": "Автоответчик. Оставьте сообщение после сигнала.",
    }


def test_disputed_review_excludes_backfilled_and_builds_review_queues(tmp_path: Path) -> None:
    db = tmp_path / "calls.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE call_records (
            id INTEGER PRIMARY KEY,
            source_filename TEXT,
            started_at TEXT,
            phone TEXT,
            manager_name TEXT,
            duration_sec REAL,
            transcription_status TEXT,
            resolve_status TEXT,
            analysis_status TEXT,
            transcript_text TEXT,
            transcript_manager TEXT,
            transcript_client TEXT,
            transcript_variants_json TEXT,
            analysis_json TEXT
        )
        """
    )
    rows = [
        (
            1,
            "call_1.mp3",
            "2026-01-02 10:00:00",
            "+79990000001",
            "Manager",
            30,
            "done",
            "skipped",
            "done",
            "Автоответчик. Оставьте сообщение после сигнала.",
            "",
            "",
            json.dumps({"full": {"variant_a": "Автоответчик", "variant_b": "Нет разговора"}}, ensure_ascii=False),
            json.dumps({"quality_flags": {"transcript_quality_backfill": {"version": "safe_non_contentful_v1"}}}, ensure_ascii=False),
        ),
        (
            2,
            "call_2.mp3",
            "2026-01-02 10:05:00",
            "+79990000002",
            "Manager",
            25,
            "done",
            "done",
            "done",
            "Автоответчик. Оставьте сообщение после сигнала.",
            "",
            "",
            "{}",
            json.dumps({"quality_flags": {"call_type": "service_call"}, "history_summary": "Менеджер якобы рассказал про обучение"}, ensure_ascii=False),
        ),
        (
            3,
            "call_3.mp3",
            "2026-01-02 10:10:00",
            "+79990000003",
            "Manager",
            45,
            "done",
            "manual",
            "pending",
            "Абонент недоступен. Перезвоните позднее.",
            "",
            "",
            "{}",
            "",
        ),
        (
            4,
            "call_4.mp3",
            "2026-01-02 10:15:00",
            "+79990000004",
            "Manager",
            60,
            "done",
            "done",
            "done",
            "Похоже, был короткий живой диалог, но запись неясная.",
            "",
            "",
            "{}",
            json.dumps({"quality_flags": {"call_type": "sales_call"}}, ensure_ascii=False),
        ),
    ]
    conn.executemany(
        "INSERT INTO call_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    guardrails = tmp_path / "guardrails"
    _write_csv(
        guardrails / "auto_fix_candidates.csv",
        [
            _base_row(1, call_type="non_conversation", contentful=False, label="non_conversation_high_confidence", force=True, manual=False),
            _base_row(2, call_type="service_call", contentful=True, label="non_conversation_high_confidence", force=True, manual=False),
        ],
    )
    _write_csv(
        guardrails / "manual_review_candidates.csv",
        [
            _base_row(3, call_type="unknown", contentful=False, label="manual_review_probable_no_live", force=False, manual=True),
            _base_row(4, call_type="sales_call", contentful=True, label="manual_review_borderline_live_context", force=False, manual=True),
        ],
    )

    out = tmp_path / "out"
    summary = build_transcript_quality_disputed_review(
        DisputedReviewConfig(
            database_url=f"sqlite:///{db}",
            guardrails_root=guardrails,
            out_root=out,
            human_sample_per_bucket=10,
        )
    )

    assert summary["input_candidate_rows"] == 4
    assert summary["already_backfilled_excluded"] == 1
    assert summary["remaining_disputed_candidates"] == 3
    assert summary["bucket_counts"] == {
        "llm_review_contentful_auto_fix_conflict": 1,
        "llm_review_non_contentful_probable_no_live": 1,
        "human_review_borderline_live_context": 1,
    }
    tasks = [json.loads(line) for line in (out / "llm_review_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 3
    assert tasks[0]["required_output_json_schema"]["decision"].startswith("keep_current_analysis")
    assert (out / "human_review_priority.csv").exists()
