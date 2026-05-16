from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.deal_aware.stage1_snapshot import (
    Stage1Paths,
    build_call_rollup,
    build_phone_rollup,
    build_stage1_snapshot,
    build_tallanto_students_snapshot,
    summarize_writeoffs,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv_cp1251(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="cp1251", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_build_phone_rollup_normalizes_phone_and_policies(tmp_path: Path) -> None:
    source = tmp_path / "master_contacts.csv"
    _write_csv(
        source,
        [
            {
                "Телефон клиента": "+7 (916) 123-45-67",
                "Всего звонков в истории": "3",
                "Содержательных звонков в истории": "2",
                "Несодержательных звонков в истории": "1",
                "Статус матчинга Tallanto": "exact_phone_single",
                "AMO contact IDs": "123",
                "CRM writeback policy": "live_update_ready",
                "Готово к записи в AMO": "Да",
            }
        ],
    )

    rows = build_phone_rollup(source)

    assert rows[0]["phone"] == "+79161234567"
    assert rows[0]["total_calls"] == 3
    assert rows[0]["contentful_calls"] == 2
    assert rows[0]["crm_writeback_policy"] == "live_update_ready"


def test_build_tallanto_students_snapshot_reads_cp1251_tsv(tmp_path: Path) -> None:
    source = tmp_path / "Ученики.csv"
    _write_tsv_cp1251(
        source,
        [
            {
                "ID": "student-1",
                "Имя": "Иван",
                "Фамилия": "Петров",
                "Тел. (родителя)": "+7 (916) 111-22-33",
                "Текстовое значение штрихкода": "4600000000001",
                "amoCRM ID": "123",
                "Баланс": "руб.1000.00",
            }
        ],
    )

    rows = build_tallanto_students_snapshot(source)

    assert rows[0]["tallanto_id"] == "student-1"
    assert rows[0]["full_name"] == "Петров Иван"
    assert rows[0]["phone_parent"] == "+79161112233"
    assert rows[0]["barcode"] == "4600000000001"
    assert rows[0]["amo_contact_id"] == "123"


def test_build_call_rollup_counts_call_types(tmp_path: Path) -> None:
    source = tmp_path / "calls.csv"
    _write_csv(
        source,
        [
            {"phone": "79161234567", "contentful": "true", "call_type": "sales_call", "started_at": "2026-01-02"},
            {"phone": "79161234567", "contentful": "false", "call_type": "non_conversation", "started_at": "2026-01-01"},
            {"phone": "79161234567", "contentful": "true", "call_type": "service_call", "started_at": "2026-01-03"},
        ],
    )

    rows = build_call_rollup(source)

    assert rows[0]["calls"] == 3
    assert rows[0]["contentful_calls"] == 2
    assert rows[0]["non_conversation_calls"] == 1
    assert rows[0]["sales_calls"] == 1
    assert rows[0]["service_calls"] == 1
    assert rows[0]["first_call_at"] == "2026-01-01"
    assert rows[0]["last_call_at"] == "2026-01-03"


def test_summarize_writeoffs_groups_by_student() -> None:
    rows = [
        {
            "last_name": "Иванов",
            "first_middle_name": "Петр",
            "birth_date": "01.01.2010",
            "writeoff_amount": "1200.50",
            "class_at": "01.02.2026 10:00",
            "class_title": "Мат 7 кл",
            "class_branch": "Онлайн",
            "subscription": "Абонемент",
        },
        {
            "last_name": "Иванов",
            "first_middle_name": "Петр",
            "birth_date": "01.01.2010",
            "writeoff_amount": "800",
            "class_at": "03.02.2026 10:00",
            "class_title": "Физ 7 кл",
            "class_branch": "МФТИ",
            "subscription": "Абонемент",
        },
    ]

    summary = summarize_writeoffs(rows)

    assert summary[0]["writeoff_count"] == 2
    assert summary[0]["total_writeoff_amount"] == 2000.5
    assert "Мат 7 кл" in summary[0]["classes"]
    assert "Физ 7 кл" in summary[0]["classes"]


def test_build_stage1_snapshot_writes_manifest_and_fail_closed_summary(tmp_path: Path) -> None:
    master_contacts = tmp_path / "master_contacts.csv"
    master_calls = tmp_path / "master_calls.csv"
    amo_ready = tmp_path / "amo_ready.csv"
    calls = tmp_path / "calls.csv"
    students = tmp_path / "students.csv"
    writeoffs = tmp_path / "writeoffs.csv"
    writeoff_summary = tmp_path / "writeoff_summary.csv"
    quality = tmp_path / "quality.json"
    runtime = tmp_path / "runtime.json"
    out_root = tmp_path / "out"

    _write_csv(
        master_contacts,
        [
            {
                "Телефон клиента": "+7 916 123-45-67",
                "Всего звонков в истории": "1",
                "Содержательных звонков в истории": "1",
                "Статус матчинга Tallanto": "exact_phone_single",
            }
        ],
    )
    _write_csv(master_calls, [{"ID звонка": "1", "Телефон клиента": "+79161234567", "Тип звонка": "sales_call"}])
    _write_csv(amo_ready, [{"Телефон клиента": "+79161234567", "Готово к записи в AMO": "Да"}])
    _write_csv(calls, [{"phone": "79161234567", "contentful": "true", "call_type": "sales_call"}])
    _write_tsv_cp1251(students, [{"ID": "student-1", "Имя": "Иван", "Фамилия": "Петров"}])
    _write_csv(writeoffs, [{"Фамилия": "Петров", "Имя": "Иван", "Сумма списания": "100", "Дата занятия": "2026-01-01"}])
    _write_csv(writeoff_summary, [{"student_key": "student-1", "visit_count": "1", "sum_writeoff": "100"}])
    quality.write_text(json.dumps({"schema_version": "gate", "passed": True, "rows": 1}), encoding="utf-8")
    runtime.write_text(json.dumps({"summary": {"schema_version": "runtime"}}), encoding="utf-8")

    summary = build_stage1_snapshot(
        Stage1Paths(
            master_contacts_csv=master_contacts,
            master_calls_csv=master_calls,
            amo_ready_csv=amo_ready,
            calls_csv=calls,
            out_root=out_root,
            current_runtime_json=runtime,
            tallanto_students_csv=students,
            tallanto_writeoff_combined_csv=writeoffs,
            tallanto_writeoff_summary_csv=writeoff_summary,
            quality_summary_paths=(quality,),
        )
    )

    assert summary["safety"]["write_amo"] is False
    assert summary["readiness"]["local_snapshot_built"] is True
    assert summary["readiness"]["safe_to_use_for_deal_writeback"] is False
    assert summary["coverage"]["call_snapshot_rows"] == 1
    assert summary["coverage"]["tallanto_students_rows"] == 1
    assert summary["coverage"]["quality_gate_rows"] == 1
    assert (out_root / "deal_aware_stage1_snapshot.sqlite").exists()
    assert (out_root / "source_manifest.csv").exists()
