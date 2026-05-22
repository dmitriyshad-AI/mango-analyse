from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_telegram_public_pilot_feedback_report import (
    build_message_rows,
    detect_asked_known_data_again,
    load_log_records,
    main,
    mask_text,
    score_human_tone,
)


def test_feedback_report_creates_expected_files_and_masks_phone(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "2026-05-21_unpk.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-05-21T10:00:00+00:00",
                        "event": "message_queued",
                        "brand": "unpk",
                        "chat_id": 123,
                        "input_text": "Представь, что я пишу с номера 79092009933",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ts": "2026-05-21T10:00:05+00:00",
                        "event": "reply_sent",
                        "brand": "unpk",
                        "chat_id": 123,
                        "input_text": "Какая цена для 9 класса?",
                        "answer_text": "Для 9 класса цена 49 000 ₽. Менеджер проверит актуальность.",
                        "route": "bot_answer_self_for_pilot",
                        "topic_id": "theme:001_pricing",
                        "message_type": "question",
                        "safety_flags": ["manager_approval_required"],
                        "known_client_fields": {"phone": "79092009933"},
                        "known_dialog_fields": {"grade": "9"},
                        "context_flags": {"known_client_fields": True},
                        "latency_seconds": 3.1,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    rc = main(["--log-dir", str(log_dir), "--date", "2026-05-21", "--brand", "unpk", "--output-dir", str(output_dir)])

    assert rc == 0
    for name in (
        "pilot_messages.csv",
        "pilot_messages.jsonl",
        "pilot_summary.json",
        "pilot_summary.md",
        "semantic_review_queue.csv",
        "regression_candidates.csv",
        "employee_review_sheet.csv",
        "daily_pilot_report_2026-05-21.md",
    ):
        assert (output_dir / name).exists()
    csv_text = (output_dir / "pilot_messages.csv").read_text(encoding="utf-8")
    assert "79092009933" not in csv_text
    assert "[PHONE_MASKED]" in csv_text
    assert "49 000 ₽" in csv_text
    with (output_dir / "semantic_review_queue.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "precise_number_date_or_percent" in rows[0]["why_review"]


def test_empty_logs_create_zero_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "out"

    main(["--log-dir", str(log_dir), "--date", "2026-05-21", "--brand", "all", "--output-dir", str(output_dir)])

    summary = json.loads((output_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    assert summary["messages_total"] == 0
    assert summary["review_queue_count"] == 0


def test_high_risk_and_fallback_enter_review_queue() -> None:
    rows = build_message_rows(
        [
            {
                "ts": "2026-05-21T10:00:00+00:00",
                "event": "reply_sent",
                "brand": "foton",
                "chat_id": 1,
                "input_text": "Хочу вернуть деньги и подать жалобу",
                "answer_text": "Спасибо за обращение. Менеджер свяжется.",
                "route": "manager_only",
                "topic_id": "theme:009_refund",
                "safety_flags": ["high_risk_manager_only"],
            }
        ]
    )

    assert len(rows) == 1
    assert "high_risk_or_p0" in rows[0]["why_review"]
    assert "fallback_or_handoff" in rows[0]["why_review"]
    assert "template_like_answer" in rows[0]["human_tone_flags"]


def test_detects_known_student_phone_and_grade_asked_again() -> None:
    repeated = detect_asked_known_data_again(
        "Напишите, пожалуйста, ФИО ребёнка, телефон и какой класс у ребёнка.",
        known_client={"student_name": "Даниил", "phone": "79092009933"},
        known_dialog={"grade": "9"},
    )

    assert repeated == ["student_name", "phone", "grade"]


def test_human_tone_score_penalizes_template_and_rewards_specific_answer() -> None:
    weak = score_human_tone("Спасибо за обращение. Менеджер свяжется.", input_text="Какая цена?")
    good = score_human_tone(
        "Да, для 9 класса есть очная подготовка по физике. Если удобно, подберём ближайшую группу и формат.",
        input_text="9 класс физика",
    )

    assert weak["score"] < 55
    assert "template_like_answer" in weak["flags"]
    assert good["score"] >= 75


def test_mask_text_removes_tokens_email_and_phone() -> None:
    masked = mask_text("token 8630108521:AAFaaaa_bbbb-ccccddddffffgggghhhh user@test.ru +7 909 200 99 33")

    assert "AAF" not in masked
    assert "user@test.ru" not in masked
    assert "909 200" not in masked


def test_load_log_records_ignores_invalid_json(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "2026-05-21_foton.jsonl").write_text("{bad json}\n{}\n", encoding="utf-8")

    records = load_log_records(log_dir, date_filter="2026-05-21", brand="foton")

    assert records == [{"brand": "foton"}]
