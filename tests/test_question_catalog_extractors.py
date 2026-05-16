from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.question_catalog.extractors import (
    extract_call_questions,
    extract_mail_questions,
    extract_telegram_questions,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_extract_call_questions_from_enriched_reviews(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched_reviews.csv"
    _write_csv(
        csv_path,
        [
            {
                "moment_id": "m1",
                "started_at": "2026-05-01T10:00:00+00:00",
                "customer_question": "Сколько стоит ЕГЭ по математике для 11 класса 10 июня?",
                "manager_answer": "Стоимость 50000 рублей, отправлю ссылку.",
                "llm_customer_signal_type": "price_question",
                "llm_hidden_sales_stage": "price_discussion",
                "answer_pattern": "price_payment_handled",
                "bot_seed_status": "needs_rop_validation",
                "outcome_group": "follow_up",
            }
        ],
    )

    items, report = extract_call_questions(csv_path, tenant_id="foton")

    assert report["items_extracted"] == 1
    assert items[0].source_channel == "call"
    assert items[0].price_related is True
    assert "10 июня" in str(items[0].metadata["customer_text_for_rop"])
    assert "50000" not in (items[0].manager_text_redacted or "")


def test_extract_telegram_questions_pairs_next_outbound(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    rows = [
        {
            "dialog_id": 100,
            "peer_kind": "user",
            "message_id": 1,
            "date": "2026-03-01T10:00:00+00:00",
            "out": False,
            "text": "Очно где проходят занятия?",
            "has_media": False,
        },
        {
            "dialog_id": 100,
            "peer_kind": "user",
            "message_id": 2,
            "date": "2026-03-01T10:05:00+00:00",
            "out": True,
            "text": "Адрес уточнит менеджер.",
            "has_media": False,
        },
    ]
    messages.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    items, report = extract_telegram_questions(messages, tenant_id="foton")

    assert report["items_extracted"] == 1
    assert items[0].source_channel == "telegram"
    assert items[0].manager_text_redacted
    assert items[0].dynamic_fact_types == ("location",)


def test_extract_mail_questions_from_read_only_sqlite(tmp_path: Path) -> None:
    archive_root = tmp_path / "mail"
    folder = archive_root / "regru_edu" / "folder_inbox"
    text_dir = folder / "extracted_text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "m1.txt"
    text_path.write_text("Добрый день! Подскажите стоимость летней школы?", encoding="utf-8")
    db_path = folder / "mail_archive.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              message_date_iso TEXT,
              subject TEXT,
              from_header TEXT,
              to_header TEXT,
              mailbox TEXT NOT NULL,
              mailbox_raw TEXT NOT NULL,
              message_kind TEXT NOT NULL,
              extracted_text_path TEXT,
              extracted_text_chars INTEGER
            )
            """
        )
        con.execute(
            """
            INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "abc",
                "2026-05-01T10:00:00+00:00",
                "Вопрос",
                "client@example.com",
                "edu@kmipt.ru",
                "INBOX",
                "INBOX",
                "external",
                str(text_path),
                55,
            ),
        )

    items, report = extract_mail_questions(archive_root, tenant_id="foton")

    assert report["items_extracted"] == 1
    assert items[0].source_channel == "email"
    assert items[0].price_related is True
    assert "client@example.com" not in items[0].customer_text_redacted


def test_extract_mail_questions_skips_bank_and_marketing_noise(tmp_path: Path) -> None:
    archive_root = tmp_path / "mail"
    folder = archive_root / "regru_edu" / "folder_inbox"
    text_dir = folder / "extracted_text"
    text_dir.mkdir(parents=True)
    bank_text = text_dir / "bank.txt"
    news_text = text_dir / "news.txt"
    notice_text = text_dir / "notice.txt"
    bank_text.write_text("Счёт: 30101810400000000225 ИНН: 7700000000 КПП: 770001001", encoding="utf-8")
    news_text.write_text("Будьте в курсе всех новостей и интересных событий!", encoding="utf-8")
    notice_text.write_text("Вы записаны на Подготовительные курсы. Ваше расписание занятий в 2026 учебном году.", encoding="utf-8")
    db_path = folder / "mail_archive.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              message_date_iso TEXT,
              subject TEXT,
              from_header TEXT,
              to_header TEXT,
              mailbox TEXT NOT NULL,
              mailbox_raw TEXT NOT NULL,
              message_kind TEXT NOT NULL,
              extracted_text_path TEXT,
              extracted_text_chars INTEGER
            )
            """
        )
        con.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "bank",
                    "2026-05-01T10:00:00+00:00",
                    "Реквизиты",
                    "client@example.com",
                    "edu@kmipt.ru",
                    "INBOX",
                    "INBOX",
                    "external",
                    str(bank_text),
                    70,
                ),
                (
                    "news",
                    "2026-05-01T11:00:00+00:00",
                    "Новости",
                    "client@example.com",
                    "edu@kmipt.ru",
                    "INBOX",
                    "INBOX",
                    "external",
                    str(news_text),
                    55,
                ),
                (
                    "notice",
                    "2026-05-01T12:00:00+00:00",
                    "Вы записаны на Подготовительные курсы",
                    "edu@kmipt.ru",
                    "client@example.com",
                    "INBOX",
                    "INBOX",
                    "external",
                    str(notice_text),
                    90,
                ),
            ],
        )

    items, report = extract_mail_questions(archive_root, tenant_id="foton")

    assert items == []
    assert report["items_extracted"] == 0
