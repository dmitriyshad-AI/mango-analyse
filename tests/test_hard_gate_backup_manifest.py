from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.quality.hard_gate_backup_manifest import (
    HardGateBackupManifestConfig,
    build_hard_gate_backup_manifest,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.read_text(encoding="utf-8-sig").strip():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table call_records (
                id integer primary key,
                source_filename text,
                source_file text,
                phone text,
                manager_name text,
                started_at text,
                transcription_status text,
                resolve_status text,
                analysis_status text,
                sync_status text,
                resolve_json text,
                resolve_quality_score real,
                analysis_json text,
                analyze_attempts integer,
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
                id, source_filename, source_file, phone, manager_name, started_at,
                transcription_status, resolve_status, analysis_status, sync_status,
                resolve_json, resolve_quality_score, analysis_json, analyze_attempts,
                dead_letter_stage, last_error, next_retry_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "call_1.mp3",
                "/tmp/call_1.mp3",
                "+79160000000",
                "Менеджер",
                "2025-01-01 10:00:00",
                "done",
                "done",
                "done",
                "synced",
                json.dumps({"decision": "old"}, ensure_ascii=False),
                91.0,
                json.dumps({"quality_flags": {"call_type": "sales_call"}}, ensure_ascii=False),
                2,
                "",
                "",
                "",
                "2026-05-09 10:00:00",
            ),
        )
        con.commit()
    finally:
        con.close()


def test_build_hard_gate_backup_manifest_captures_rollback_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    _make_db(db_path)
    apply_plan = tmp_path / "plan.csv"
    _write_csv(
        apply_plan,
        [
            {
                "audit_id": "hgate_full_000001",
                "task_id": f"hard_gate_gpt::{db_path}::1",
                "db": str(db_path),
                "id": "1",
                "source_filename": "call_1.mp3",
                "queue": "auto_apply_ready",
                "risk_level": "critical",
                "review_hash": "abc123",
                "current_call_type": "sales_call",
                "normalized_call_type": "non_conversation",
            },
            {
                "audit_id": "hgate_full_000002",
                "task_id": f"hard_gate_gpt::{db_path}::2",
                "db": str(db_path),
                "id": "2",
                "source_filename": "missing.mp3",
                "queue": "auto_apply_ready",
                "risk_level": "low",
                "review_hash": "def456",
                "current_call_type": "technical_call",
                "normalized_call_type": "non_conversation",
            },
        ],
    )

    out_root = tmp_path / "manifest"
    summary = build_hard_gate_backup_manifest(
        HardGateBackupManifestConfig(
            apply_plan_csv=apply_plan,
            out_root=out_root,
            project_root=tmp_path,
        )
    )

    assert summary["selected_rows"] == 2
    assert summary["dbs_affected"] == 1
    assert summary["rollback_rows"] == 1
    assert summary["missing_rows"] == 1
    assert summary["queue_counts"] == {"auto_apply_ready": 2}

    db_manifest = _read_csv(out_root / "db_manifest.csv")
    assert db_manifest[0]["exists"] == "True"
    assert db_manifest[0]["sha256"]
    assert db_manifest[0]["candidate_rows"] == "2"
    assert db_manifest[0]["rollback_rows_found"] == "1"

    rollback = _jsonl(out_root / "rollback_snapshot.jsonl")
    assert rollback[0]["before_analysis_status"] == "done"
    assert rollback[0]["before_resolve_status"] == "done"
    assert rollback[0]["before_analysis_json_sha256"]
    assert "sales_call" in rollback[0]["before_analysis_json"]

    missing = _read_csv(out_root / "missing_rows.csv")
    assert missing[0]["reason"] == "row_missing"
    assert "cp -p" in (out_root / "backup_copy_plan.sh").read_text(encoding="utf-8")


def test_build_hard_gate_backup_manifest_can_filter_queue(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    _make_db(db_path)
    apply_plan = tmp_path / "plan.csv"
    _write_csv(
        apply_plan,
        [
            {
                "audit_id": "a1",
                "task_id": f"hard_gate_gpt::{db_path}::1",
                "db": str(db_path),
                "id": "1",
                "source_filename": "call_1.mp3",
                "queue": "auto_apply_ready",
                "risk_level": "critical",
                "review_hash": "abc123",
                "current_call_type": "sales_call",
                "normalized_call_type": "non_conversation",
            },
            {
                "audit_id": "a2",
                "task_id": f"hard_gate_gpt::{db_path}::1",
                "db": str(db_path),
                "id": "1",
                "source_filename": "call_1.mp3",
                "queue": "gpt_review_required",
                "risk_level": "critical",
                "review_hash": "abc123",
                "current_call_type": "sales_call",
                "normalized_call_type": "non_conversation",
            },
        ],
    )

    summary = build_hard_gate_backup_manifest(
        HardGateBackupManifestConfig(
            apply_plan_csv=apply_plan,
            out_root=tmp_path / "manifest",
            project_root=tmp_path,
            queue_filter="auto_apply_ready",
            hash_db_files=False,
        )
    )

    assert summary["selected_rows"] == 1
    assert summary["rollback_rows"] == 1
    assert summary["queue_counts"] == {"auto_apply_ready": 1}
    db_manifest = _read_csv(tmp_path / "manifest" / "db_manifest.csv")
    assert db_manifest[0]["sha256"] == ""
