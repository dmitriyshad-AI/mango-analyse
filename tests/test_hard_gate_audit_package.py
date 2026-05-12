from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.quality.hard_gate_audit_package import (
    HardGateAuditPackageConfig,
    build_hard_gate_audit_package,
    select_stratified_candidates,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_db(path: Path, count: int) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table call_records (
                id integer primary key,
                source_file text,
                source_filename text,
                phone text,
                manager_name text,
                duration_sec real,
                started_at text,
                transcript_manager text,
                transcript_client text,
                transcript_text text,
                transcript_variants_json text,
                resolve_json text,
                analysis_json text
            )
            """
        )
        for idx in range(1, count + 1):
            con.execute(
                """
                insert into call_records (
                    id, source_file, source_filename, phone, manager_name, duration_sec, started_at,
                    transcript_manager, transcript_client, transcript_text, transcript_variants_json,
                    resolve_json, analysis_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idx,
                    f"/tmp/call_{idx}.mp3",
                    f"call_{idx}.mp3",
                    "+79160000000",
                    "Менеджер",
                    20 + idx,
                    "2025-01-01 10:00:00",
                    "Здравствуйте, это Фотон.",
                    "Абонент сейчас не может ответить. Оставьте сообщение после сигнала.",
                    "MANAGER: Здравствуйте, это Фотон.\nCLIENT: Абонент сейчас не может ответить.",
                    "{}",
                    "{}",
                    json.dumps(
                        {
                            "analysis_schema_version": "v2",
                            "history_summary": "Старый анализ ошибочно считал звонок продажным.",
                            "quality_flags": {"call_type": "sales_call"},
                            "follow_up_score": 80,
                            "target_product": "курс",
                            "tags": ["sales_call"],
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        con.commit()
    finally:
        con.close()


def test_select_stratified_candidates_covers_all_strata_before_extras() -> None:
    rows: list[dict[str, str]] = []
    idx = 1
    for month in ("2025-01", "2025-02"):
        for call_type in ("sales_call", "service_call"):
            for subtype in ("no_live_or_voicemail", "outbound_voicemail"):
                for _ in range(3):
                    rows.append(
                        {
                            "db": "calls.db",
                            "id": str(idx),
                            "source_filename": f"call_{idx}.mp3",
                            "month": month,
                            "current_call_type": call_type,
                            "guardrail_recommended_contact_subtype": subtype,
                        }
                    )
                    idx += 1

    selected = select_stratified_candidates(rows, limit=8)
    strata = {
        (row["month"], row["current_call_type"], row["guardrail_recommended_contact_subtype"])
        for row in selected
    }
    assert len(selected) == 8
    assert len(strata) == 8


def test_build_hard_gate_audit_package_writes_full_jsonl_and_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    _make_db(db_path, count=4)
    candidates = [
        {
            "db": str(db_path),
            "id": str(idx),
            "source_filename": f"call_{idx}.mp3",
            "source_file": f"/tmp/call_{idx}.mp3",
            "started_at": "2025-01-01 10:00:00",
            "month": "2025-01" if idx <= 2 else "2025-02",
            "phone": "+79160000000",
            "manager_name": "Менеджер",
            "duration_sec": "25",
            "status": "would_update",
            "update_reasons": "call_type_to_non_conversation|clear_sales_fields",
            "current_call_type": "sales_call" if idx <= 2 else "service_call",
            "normalized_call_type": "non_conversation",
            "transition": "sales_call->non_conversation",
            "current_follow_up_score": "80",
            "normalized_follow_up_score": "0",
            "current_next_step": "Перезвонить",
            "normalized_next_step": "",
            "current_products": "курс",
            "normalized_products": "",
            "current_subjects": "математика",
            "normalized_subjects": "",
            "current_objections": "цена",
            "normalized_objections": "",
            "guardrail_label": "non_conversation_high_confidence",
            "guardrail_score": "100",
            "guardrail_reason_codes": "operator_unavailable|voicemail",
            "guardrail_should_force_non_conversation": "True",
            "guardrail_requires_manual_review": "False",
            "guardrail_protected_live_dialogue": "False",
            "guardrail_recommended_contact_subtype": "no_live_or_voicemail" if idx % 2 else "outbound_voicemail",
            "hard_validation_applied": "True",
            "current_history_summary_excerpt": "Клиент заинтересован.",
            "normalized_history_summary_excerpt": "Нет живого диалога с клиентом.",
            "transcript_excerpt": "Абонент сейчас не может ответить.",
        }
        for idx in range(1, 5)
    ]
    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(candidates_csv, candidates)

    out_root = tmp_path / "package"
    summary = build_hard_gate_audit_package(
        HardGateAuditPackageConfig(
            candidates_csv=candidates_csv,
            out_root=out_root,
            project_root=tmp_path,
            limit=4,
        )
    )

    assert summary["selected"] == 4
    assert (out_root / "AUDIT_PROMPT_RU.md").exists()
    assert (out_root / "README_FOR_CLAUDE_AND_GPT.md").exists()
    assert (out_root / "decisions_template.jsonl").exists()
    assert (out_root / "expected_output_schema.json").exists()
    assert len(_read_csv(out_root / "audit_items_preview.csv")) == 4
    assert len(_read_csv(out_root / "PRIVATE_mapping_for_apply_do_not_edit.csv")) == 4
    assert len(_read_csv(out_root / "strata_summary.csv")) == 4

    items = _jsonl(out_root / "audit_items.jsonl")
    assert items[0]["audit_id"] == "hgate200_0001"
    assert "Абонент сейчас не может ответить" in items[0]["transcript"]["final_text"]
    assert items[0]["auditor_required_output"]["decision"] == "safe_apply | keep_current | manual_review"
