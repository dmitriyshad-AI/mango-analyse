from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.question_catalog.builder import CatalogBuildConfig, build_customer_question_catalog


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_customer_question_catalog_e2e(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls = project / "calls.csv"
    _write_csv(
        calls,
        [
            {
                "moment_id": "m1",
                "started_at": "2026-05-01T10:00:00+00:00",
                "customer_question": "Сколько стоит ЕГЭ по математике для 11 класса?",
                "manager_answer": "Стоимость 50000 рублей.",
                "llm_customer_signal_type": "price_question",
            }
        ],
    )
    telegram_dir = project / "telegram"
    telegram_dir.mkdir()
    telegram = telegram_dir / "messages.jsonl"
    telegram.write_text(
        json.dumps(
            {
                "dialog_id": 100,
                "peer_kind": "user",
                "message_id": 1,
                "date": "2026-03-01T10:00:00+00:00",
                "out": False,
                "text": "Где проходят занятия очно?",
                "has_media": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mail_root = project / "mail"
    folder = mail_root / "regru_edu" / "folder_inbox"
    text_dir = folder / "extracted_text"
    text_dir.mkdir(parents=True)
    text_path = text_dir / "m1.txt"
    text_path.write_text("Добрый день! Пришлите расписание группы.", encoding="utf-8")
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
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "abc",
                "2026-05-01T10:00:00+00:00",
                "Расписание",
                "client@example.com",
                "edu@kmipt.ru",
                "INBOX",
                "INBOX",
                "external",
                str(text_path),
                40,
            ),
        )
    facts = project / "facts"
    facts.mkdir()
    (facts / "Стоимость 2026.txt").write_text("цены требуют ручной проверки", encoding="utf-8")
    out = project / "product_data" / "question_catalog"
    summary = build_customer_question_catalog(
        CatalogBuildConfig(
            project_root=project,
            out_root=out,
            tenant_id="foton",
            since=datetime(2025, 1, 1, tzinfo=timezone.utc),
            calls_enriched_reviews=calls,
            telegram_messages_jsonl=telegram,
            mail_archive_root=mail_root,
            fact_source_roots=(facts,),
        )
    )

    assert summary["totals"]["question_items"] >= 3
    assert summary["totals"]["question_classes"] >= 3
    assert summary["totals"]["fact_sources"] == 1
    assert (out / "customer_question_items.jsonl").exists()
    assert (out / "customer_question_classes.xlsx").exists()
    assert (out / "rop_question_review_pack.xlsx").exists()
    assert (out / "current_fact_source_registry.json").exists()
    assert "50000" not in (out / "customer_question_items.jsonl").read_text(encoding="utf-8")
