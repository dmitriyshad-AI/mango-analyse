from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.quality.hard_gate_staged_backfill import HardGateStagedBackfillConfig, run_hard_gate_staged_backfill


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table call_records (
                id integer primary key,
                source_filename text,
                started_at text,
                manager_name text,
                phone text,
                transcript_manager text,
                transcript_client text,
                transcript_text text,
                resolve_status text,
                resolve_json text,
                resolve_quality_score real,
                analysis_status text,
                analysis_json text,
                analyze_attempts integer,
                sync_status text,
                dead_letter_stage text,
                last_error text,
                next_retry_at text,
                updated_at text
            )
            """
        )
        con.execute(
            """
            insert into call_records (
                id, source_filename, started_at, manager_name, phone, transcript_manager,
                transcript_client, transcript_text, resolve_status, resolve_json,
                resolve_quality_score, analysis_status, analysis_json, analyze_attempts
            ) values (1, 'call.mp3', '2025-01-01 10:00:00', 'Manager', '+7000', '', '',
                      'Абонент не отвечает. Оставьте сообщение.', 'done', '{}', 90, 'done', '{}', 1)
            """
        )
        con.commit()
    finally:
        con.close()


def _candidate(db: str) -> dict[str, str]:
    return {
        "queue": "auto_apply_ready",
        "db": db,
        "id": "1",
        "source_filename": "call.mp3",
        "current_call_type": "sales_call",
        "guardrail_label": "non_conversation_high_confidence",
        "guardrail_score": "-8",
        "guardrail_reason_codes": "system_no_dialogue_phrase|no_live_marker",
        "should_force_non_conversation": "True",
        "recommended_contact_subtype": "no_live_or_voicemail",
        "policy_queue": "gpt_auto_apply",
        "policy_auto_apply_allowed": "True",
        "gpt_decision": "safe_apply",
        "review_decision": "hard_gate_gpt_auto_apply",
        "review_hash": "abc123",
        "risk_level": "critical",
        "month": "2025-01",
    }


def test_staged_backfill_groups_by_db_and_applies(tmp_path: Path) -> None:
    db = tmp_path / "calls.db"
    _make_db(db)
    candidates = tmp_path / "auto_apply.csv"
    _write_csv(candidates, [_candidate(str(db))])

    dry = run_hard_gate_staged_backfill(
        HardGateStagedBackfillConfig(
            project_root=tmp_path,
            auto_apply_csv=candidates,
            out_root=tmp_path / "dry",
            mode="dry-run",
        )
    )
    applied = run_hard_gate_staged_backfill(
        HardGateStagedBackfillConfig(
            project_root=tmp_path,
            auto_apply_csv=candidates,
            out_root=tmp_path / "apply",
            mode="apply",
        )
    )

    assert dry["counts"]["planned_updates"] == 1
    assert applied["counts"]["applied_updates"] == 1
    con = sqlite3.connect(db)
    try:
        row = con.execute("select analysis_json from call_records where id = 1").fetchone()
    finally:
        con.close()
    payload = json.loads(row[0])
    assert payload["quality_flags"]["call_type"] == "non_conversation"
    assert payload["quality_flags"]["transcript_quality_backfill"]["source_gpt_decision"] == "safe_apply"
