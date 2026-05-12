from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.channels.actions import (
    ACTION_CREATE_FOLLOW_UP_TASK,
    ACTION_DRAFT_CLIENT_MESSAGE,
    ACTION_HANDOFF_TO_MANAGER,
    ACTION_MARK_MANUAL_REVIEW,
    ACTION_NOTIFY_ROP_HOT_LEAD,
    ACTION_REQUEST_CRM_CONTEXT,
)
from mango_mvp.customer_timeline.preview_quality_audit import (
    build_preview_quality_audit,
    classify_intents,
    compare_product_with_employee,
    default_synthetic_preview_cases,
    expected_actions_for_message,
    extract_latest_telegram_reply_pairs,
    render_preview_quality_audit_markdown,
    score_employee_reply,
)


FIXED_TIME = datetime(2026, 5, 13, 15, 0, tzinfo=timezone.utc)


def test_preview_quality_audit_runs_synthetic_and_real_pairs_without_raw_text_leak(tmp_path: Path) -> None:
    export_dir = build_telegram_export(tmp_path / "telegram_export")

    report = build_preview_quality_audit(
        project_root=tmp_path,
        telegram_export_dir=export_dir,
        real_pair_limit=3,
        generated_at=FIXED_TIME,
    )
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["validation_ok"] is True
    assert report["summary"]["synthetic_cases"] == len(default_synthetic_preview_cases())
    assert report["summary"]["telegram_pairs_sampled"] == 3
    assert report["summary"]["telegram_export_status"] == "loaded"
    assert report["checkpoints"]["stage_1_9_synthetic_preview"] == "done"
    assert report["checkpoints"]["quality_scoring"] == "done"
    assert report["checkpoints"]["real_telegram_100_pair_comparison"] == "done"
    assert report["checkpoints"]["problem_classes_and_fix_plan"] == "done"
    assert report["safety"]["live_send"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["send_messenger"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False
    assert "Добрый день! Пришлите" not in serialized
    assert "г. Москва" not in serialized
    assert "+79161234567" not in serialized
    assert "parent@example.com" not in serialized


def test_extract_latest_telegram_reply_pairs_uses_user_inbound_to_outbound_text_pairs(tmp_path: Path) -> None:
    export_dir = build_telegram_export(tmp_path / "telegram_export")

    pairs = extract_latest_telegram_reply_pairs(export_dir, limit=10)

    assert len(pairs) == 4
    assert [pair.inbound_text for pair in pairs][:2] == [
        "Добрый день! Пришлите курсы по математике для 6 класса",
        "Очно где проводятся занятия?",
    ]
    assert all(pair.employee_reply_text for pair in pairs)
    assert all(pair.channel_thread_id == "111" for pair in pairs)


def test_intent_classification_and_expected_actions_cover_manager_signals() -> None:
    text = "Срочно хочу записаться и оплатить. Можно перезвонить менеджеру?"

    intents = classify_intents(text, has_attachments=False)
    actions = expected_actions_for_message(text, has_attachments=False)

    assert {"callback", "hot_lead", "manager_handoff", "price_or_payment"}.issubset(intents)
    assert ACTION_DRAFT_CLIENT_MESSAGE in actions
    assert ACTION_REQUEST_CRM_CONTEXT in actions
    assert ACTION_CREATE_FOLLOW_UP_TASK in actions
    assert ACTION_HANDOFF_TO_MANAGER in actions
    assert ACTION_MARK_MANUAL_REVIEW in actions
    assert ACTION_NOTIFY_ROP_HOT_LEAD in actions


def test_employee_score_can_beat_generic_product_when_employee_is_specific() -> None:
    product_like = score_employee_reply(
        inbound_text="Где проходят очные занятия?",
        employee_reply_text="Здравствуйте! Менеджер уточнит детали и вернется с ответом.",
    )
    specific_employee = score_employee_reply(
        inbound_text="Где проходят очные занятия?",
        employee_reply_text="Добрый день! Очные занятия проходят в Москве, Скорняжный пер., д. 3, стр. 1.",
    )

    assert specific_employee.score > product_like.score
    assert specific_employee.specificity_score > product_like.specificity_score
    assert compare_product_with_employee(product_like, specific_employee) == "employee_better"


def test_markdown_report_is_operator_readable_and_safety_focused(tmp_path: Path) -> None:
    export_dir = build_telegram_export(tmp_path / "telegram_export")
    report = build_preview_quality_audit(
        project_root=tmp_path,
        telegram_export_dir=export_dir,
        real_pair_limit=2,
        generated_at=FIXED_TIME,
    )

    markdown = render_preview_quality_audit_markdown(report)

    assert "без отправки сообщений" in markdown
    assert "Реальных Telegram-пар найдено" in markdown
    assert "План устранения" in markdown
    assert "Живая отправка сообщений: нет" in markdown
    assert "Добрый день! Пришлите" not in markdown


def build_telegram_export(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "since": "2024-04-01T00:00:00+00:00",
                "total_dialogs": 2,
                "total_messages": 10,
                "finished_at": "2026-04-15T12:00:00+00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        root / "dialogs.jsonl",
        [
            {
                "dialog_id": 111,
                "name": "Client",
                "peer_kind": "user",
                "is_user": True,
                "is_group": False,
                "is_channel": False,
                "top_message_date": "2026-03-26T11:00:00+00:00",
            },
            {
                "dialog_id": 222,
                "name": "Group",
                "peer_kind": "chat",
                "is_user": False,
                "is_group": True,
                "is_channel": False,
                "top_message_date": "2026-03-26T11:00:00+00:00",
            },
        ],
    )
    write_jsonl(
        root / "messages.jsonl",
        [
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 1,
                "date": "2026-03-26T09:00:00+00:00",
                "sender_id": 111,
                "text": "Добрый день! Пришлите курсы по математике для 6 класса",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 2,
                "date": "2026-03-26T09:05:00+00:00",
                "sender_id": 999,
                "text": "Добрый день! Подскажите, очно или онлайн рассматриваете?",
                "out": True,
                "reply_to_msg_id": 1,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 3,
                "date": "2026-03-26T09:06:00+00:00",
                "sender_id": 111,
                "text": "Очно где проводятся занятия?",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 4,
                "date": "2026-03-26T09:07:00+00:00",
                "sender_id": 999,
                "text": "г. Москва, Скорняжный пер. д. 3, стр. 1.",
                "out": True,
                "reply_to_msg_id": 3,
                "has_media": False,
            },
            {
                "dialog_id": 222,
                "dialog_name": "Group",
                "peer_kind": "chat",
                "message_id": 5,
                "date": "2026-03-26T09:08:00+00:00",
                "sender_id": 222,
                "text": "Групповое сообщение не берем",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 222,
                "dialog_name": "Group",
                "peer_kind": "chat",
                "message_id": 6,
                "date": "2026-03-26T09:09:00+00:00",
                "sender_id": 999,
                "text": "Групповой ответ тоже не берем",
                "out": True,
                "reply_to_msg_id": 5,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 7,
                "date": "2026-03-26T09:10:00+00:00",
                "sender_id": 111,
                "text": "Сколько стоит пробный абонемент? Телефон +79161234567",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 8,
                "date": "2026-03-26T09:11:00+00:00",
                "sender_id": 999,
                "text": "Пробный абонемент стоит 8 900 руб.",
                "out": True,
                "reply_to_msg_id": 7,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 9,
                "date": "2026-03-26T09:12:00+00:00",
                "sender_id": 111,
                "text": "Что нужно для записи? parent@example.com",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client",
                "peer_kind": "user",
                "message_id": 10,
                "date": "2026-03-26T09:13:00+00:00",
                "sender_id": 999,
                "text": "Для записи пришлите ФИО ученика, класс и дату рождения.",
                "out": True,
                "reply_to_msg_id": 9,
                "has_media": False,
            },
        ],
    )
    return root


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
