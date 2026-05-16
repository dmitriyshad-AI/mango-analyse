from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from mango_mvp.deal_aware.deal_text_builder import (
    DealTextPaths,
    build_deal_payload,
    build_deal_text_preview,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_paid_deal_next_step_is_context_only_not_payment_collection() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ",
            "selected_status_name": "Оплата получена",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "context_only_paid_or_success",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
            "stage3_risk_flags": "paid_deal_has_payment_next_step_in_call",
        },
        [],
        [
            {
                "call_summary": "Клиент уже прислал чек, сделка оплачена.",
                "next_step": "Отправить ссылку на оплату",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: точный ученик найден."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    next_step = payload["AI-рекомендованный следующий шаг"].casefold()
    assert "оплат" not in next_step
    assert "платеж" not in next_step
    assert "коммерческий дожим" in next_step


def test_payload_normalizes_known_tenant_asr_errors_and_removes_ellipsis() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "НПК МФТИ июнь",
            "selected_status_name": "В работе",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
        },
        [],
        [
            {
                "call_summary": "Клиент спрашивал про МПК МФТИ и летние ночные школы... Нужны даты.",
                "next_step": "Отправить материалы",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    text = "\n".join(payload.values())
    assert "..." not in text
    assert "…" not in text
    assert "МПК МФТИ" not in text
    assert not re.search(r"(?<!У)НПК МФТИ", text)
    assert "УНПК МФТИ" in text
    assert "ночн" not in text.casefold()


def test_stage2_confidence_low_warning_stays_out_of_manager_warning_field() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ",
            "selected_status_name": "Перспектива",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
            "stage3_risk_flags": "stage2_confidence_low",
        },
        [],
        [
            {
                "call_summary": "Клиент интересуется летней школой.",
                "next_step": "Отправить материалы",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    assert "Stage 2 confidence" not in payload["AI-предупреждение по сделке"]


def test_next_step_rewrites_customer_side_actions_to_manager_control() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ август",
            "selected_status_name": "Ожидание оплаты",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
        },
        [],
        [
            {
                "call_summary": "Клиент согласует участие с сыном и затем оплатит выбранную программу.",
                "next_step": "Согласовать участие с сыном и оплатить",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    assert payload["AI-рекомендованный следующий шаг"].startswith("Проконтролировать оплату")
    assert "Согласовать участие с сыном и оплатить" not in payload["AI-сводка по сделке"]


def test_next_step_rewrites_passive_waiting_to_active_control() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "ЛВШ",
            "selected_status_name": "Ожидание оплаты",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-05-05 10:00:00",
        },
        [],
        [
            {
                "call_summary": "Клиент сказал, что оплатит до 15 мая.",
                "next_step": "Ждать оплату до 15 мая",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    assert "Контролировать поступление оплаты до 15 мая" in payload["AI-рекомендованный следующий шаг"]
    assert "до до" not in payload["AI-рекомендованный следующий шаг"]


def test_commercial_course_call_with_homework_description_is_not_service_feedback() -> None:
    payload = build_deal_payload(
        {
            "selected_deal_id": "100",
            "selected_deal_name": "Заявка с сайта",
            "selected_status_name": "Принимают решение",
            "selected_pipeline_name": "Сделки B2C",
            "deal_writeback_mode": "full_active",
            "candidate_call_count": "1",
            "candidate_phone_count": "1",
            "last_call_at": "2026-04-27 16:43:46",
        },
        [],
        [
            {
                "call_summary": (
                    "Клиент интересуется курсом по математике для ученика 7-го класса. "
                    "Менеджер объяснила формат занятий с домашними заданиями и контрольными работами."
                ),
                "next_step": "Отправить материалы",
                "started_at": "2026-04-27 16:43:46",
                "manager_name": "Менеджер",
            }
        ],
        tallanto_context={"text": "Tallanto: нет точного сопоставления."},
        generated_at="2026-05-13T00:00:00+00:00",
        analysis_date="2026-05-13",
    )

    assert "сервисную обратную связь" not in payload["AI-сводка по сделке"]
    assert payload["AI-рекомендованный следующий шаг"].startswith("Отправить клиенту материалы")


def test_build_stage4_preview_outputs_read_only_quality_artifacts(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1"
    stage3 = tmp_path / "stage3"
    out = tmp_path / "out"
    _write_csv(
        stage3 / "deal_stage4_deal_candidates.csv",
        [
            {
                "selected_deal_id": "100",
                "selected_deal_name": "ЛВШ июнь",
                "selected_pipeline_name": "Сделки B2C",
                "selected_status_name": "Ожидание оплаты",
                "deal_writeback_mode": "full_active",
                "candidate_call_count": "1",
                "candidate_phone_count": "1",
                "phones": "+79000000000",
                "last_call_at": "2026-05-05 10:00:00",
                "latest_call_id": "c1",
                "latest_call_next_step": "Отправить материалы",
                "safe_for_stage4_generation": "Да",
            }
        ],
    )
    _write_csv(
        stage3 / "deal_call_writeback_policy.csv",
        [
            {
                "call_id": "c1",
                "selected_deal_id": "100",
                "phone": "+79000000000",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
                "safe_for_stage4_generation": "Да",
                "confidence_bucket": "high",
            }
        ],
    )
    (stage3 / "summary.json").write_text('{"schema_version": "stage3"}', encoding="utf-8")
    _write_csv(
        stage1 / "call_snapshot.csv",
        [
            {
                "call_id": "c1",
                "started_at": "2026-05-05 10:00:00",
                "manager_name": "Менеджер",
                "call_summary": "Клиент подтвердил интерес к летней очной школе и попросил материалы.",
                "next_step": "Отправить материалы",
                "objections": "цена",
            }
        ],
    )
    _write_csv(
        stage1 / "phone_rollup.csv",
        [
            {
                "phone": "+79000000000",
                "tallanto_match_status": "exact_phone_single",
                "tallanto_id": "t1",
            }
        ],
    )
    _write_csv(
        stage1 / "tallanto_students_snapshot.csv",
        [
            {
                "tallanto_id": "t1",
                "full_name": "Иван Иванов",
                "barcode": "b1",
                "student_type": "ученик",
                "branch": "онлайн",
            }
        ],
    )
    _write_csv(
        stage1 / "tallanto_writeoff_summary_by_student.csv",
        [{"barcode": "b1", "visit_count": "3", "last_lesson_at": "2026-04-30"}],
    )
    (stage1 / "summary.json").write_text('{"schema_version": "stage1"}', encoding="utf-8")

    summary = build_deal_text_preview(
        DealTextPaths(stage1_snapshot_root=stage1, stage3_deal_state_root=stage3, out_root=out)
    )

    assert summary["safety"]["write_amo"] is False
    assert summary["readiness"]["safe_to_write_deal_fields"] is False
    assert summary["coverage"]["preview_rows"] == 1
    assert (out / "deal_stage4_preview.csv").exists()
    assert (out / "deal_stage4_payloads.jsonl").exists()
    payload_line = (out / "deal_stage4_payloads.jsonl").read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(payload_line)["payload"]
    assert set(payload).issuperset({"AI-сводка по сделке", "AI-история по сделке"})
    assert "летний лагерь: 8" not in "\n".join(payload.values())
