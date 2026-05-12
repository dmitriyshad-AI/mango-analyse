from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.quality.hard_gate_gpt_apply_plan import (
    HardGateGptApplyPlanConfig,
    build_hard_gate_gpt_apply_plan,
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
                transcript_manager text,
                transcript_client text,
                transcript_text text
            )
            """
        )
        for idx in range(1, 4):
            con.execute(
                """
                insert into call_records (
                    id, source_filename, transcript_manager, transcript_client, transcript_text
                ) values (?, ?, ?, ?, ?)
                """,
                (
                    idx,
                    f"call_{idx}.mp3",
                    "Добрый день, это Фотон.",
                    "Абонент сейчас не может ответить. Оставьте сообщение после сигнала.",
                    "MANAGER: Добрый день.\nCLIENT: Абонент сейчас не может ответить.",
                ),
            )
        con.commit()
    finally:
        con.close()


def _candidate(db_path: Path, idx: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "db": str(db_path),
        "id": str(idx),
        "source_filename": f"call_{idx}.mp3",
        "source_file": f"/tmp/call_{idx}.mp3",
        "started_at": "2025-01-01 10:00:00",
        "month": "2025-01",
        "phone": "+79160000000",
        "manager_name": "Менеджер",
        "duration_sec": "22",
        "status": "would_update",
        "update_reasons": "call_type_to_non_conversation|clear_sales_fields|hard_validation_applied",
        "current_call_type": "sales_call" if idx == 1 else "technical_call",
        "normalized_call_type": "non_conversation",
        "transition": "sales_call->non_conversation",
        "current_follow_up_score": "80",
        "normalized_follow_up_score": "0",
        "current_next_step": "Перезвонить",
        "normalized_next_step": "",
        "current_products": "",
        "normalized_products": "",
        "current_subjects": "",
        "normalized_subjects": "",
        "current_objections": "время",
        "normalized_objections": "",
        "guardrail_label": "non_conversation_high_confidence",
        "guardrail_score": "-8",
        "guardrail_reason_codes": "system_no_dialogue_phrase|no_live_marker",
        "guardrail_should_force_non_conversation": "True",
        "guardrail_requires_manual_review": "False",
        "guardrail_protected_live_dialogue": "False",
        "guardrail_recommended_contact_subtype": "no_live_or_voicemail",
        "hard_validation_applied": "True",
        "current_history_summary_excerpt": "Старый анализ.",
        "normalized_history_summary_excerpt": "Нет живого диалога.",
        "transcript_excerpt": "Абонент сейчас не может ответить.",
    }
    row.update(overrides)
    return row


def test_build_hard_gate_gpt_apply_plan_requires_gpt_review_without_decisions(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    _make_db(db_path)
    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(
        candidates_csv,
        [
            _candidate(db_path, 1),
            _candidate(db_path, 2),
            _candidate(db_path, 3, guardrail_requires_manual_review="True"),
        ],
    )

    out_root = tmp_path / "plan"
    summary = build_hard_gate_gpt_apply_plan(
        HardGateGptApplyPlanConfig(
            candidates_csv=candidates_csv,
            out_root=out_root,
            project_root=tmp_path,
        )
    )

    assert summary["input_candidates"] == 3
    assert summary["auto_apply_ready"] == 0
    assert summary["gpt_review_required"] == 2
    assert summary["blocked_candidates"] == 1
    assert summary["review_tasks"] == 2

    tasks = _jsonl(out_root / "gpt_review_tasks.jsonl")
    assert tasks[0]["policy_version"] == "hard_gate_gpt_policy_v1"
    assert "Абонент сейчас не может ответить" in tasks[0]["transcript"]["full_text"]

    blocked = _read_csv(out_root / "blocked_candidates.csv")
    assert blocked[0]["deterministic_blockers"] == "guardrail_requires_manual_review"


def test_build_hard_gate_gpt_apply_plan_promotes_safe_gpt_decisions(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    _make_db(db_path)
    candidates_csv = tmp_path / "candidates.csv"
    rows = [_candidate(db_path, 1), _candidate(db_path, 2)]
    _write_csv(candidates_csv, rows)
    decisions_jsonl = tmp_path / "decisions.jsonl"
    task_1 = f"hard_gate_gpt::{db_path}::1"
    task_2 = f"hard_gate_gpt::{db_path}::2"
    decisions_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"task_id": task_1, "decision": "safe_apply", "confidence": 0.98, "reason_ru": "no live dialogue"}, ensure_ascii=False),
                json.dumps({"task_id": task_2, "decision": "keep_current", "confidence": 0.92, "reason_ru": "live dialogue risk"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "plan"
    summary = build_hard_gate_gpt_apply_plan(
        HardGateGptApplyPlanConfig(
            candidates_csv=candidates_csv,
            out_root=out_root,
            project_root=tmp_path,
            gpt_decisions_jsonl=decisions_jsonl,
        )
    )

    assert summary["auto_apply_ready"] == 1
    assert summary["keep_current"] == 1
    assert summary["gpt_review_required"] == 0

    auto = _read_csv(out_root / "auto_apply_ready.csv")
    assert auto[0]["review_decision"] == "hard_gate_gpt_auto_apply"
    assert auto[0]["policy_queue"] == "gpt_auto_apply"
    assert auto[0]["policy_auto_apply_allowed"] == "True"
    assert auto[0]["gpt_decision"] == "safe_apply"

    keep = _read_csv(out_root / "keep_current.csv")
    assert keep[0]["recommended_action"] == "do_not_apply_keep_existing_analysis"
