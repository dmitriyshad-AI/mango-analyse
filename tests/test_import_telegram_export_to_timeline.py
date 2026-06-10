from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.contracts import CustomerIdentity, IdentityStatus
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore

from scripts.import_telegram_export_to_timeline import (
    TelegramExportImportConfig,
    run_telegram_export_import,
    tg_message_to_payload,
)


def test_mapper_builds_channel_payload_with_thread_source_id_and_phone() -> None:
    payload = tg_message_to_payload(
        {
            "dialog_id": 123,
            "dialog_name": "Иван",
            "message_id": 456,
            "date": "2026-03-28T09:38:18+00:00",
            "sender_id": 123,
            "text": "  Добрый день  ",
            "out": False,
        },
        {"dialog_id": 123, "name": "Иван", "peer_kind": "user", "phone": "9991112233"},
        "foton",
    )

    assert payload["channel"] == "telegram"
    assert payload["channel_thread_id"] == "123"
    assert payload["channel_message_id"] == "456"
    assert payload["direction"] == "inbound"
    assert payload["text"] == "Добрый день"
    assert payload["brand_hint"] == "foton"
    assert payload["dialog_phone_normalized"] == "+79991112233"
    assert payload["timeline_source_id"] == "telegram:123:456"


def test_group_dialog_is_skipped_with_counter(tmp_path: Path) -> None:
    export_dir = write_export(
        tmp_path,
        dialogs=[
            {"dialog_id": 10, "name": "Группа", "peer_kind": "chat", "phone": None},
            {"dialog_id": 11, "name": "Пустой", "peer_kind": "user", "phone": None},
        ],
        messages=[
            {
                "dialog_id": 10,
                "peer_kind": "chat",
                "message_id": 1,
                "date": "2026-03-28T09:38:18+00:00",
                "text": "Сообщение группы",
                "out": False,
            },
            {
                "dialog_id": 11,
                "peer_kind": "user",
                "message_id": 2,
                "date": "2026-03-28T09:39:18+00:00",
                "text": "   ",
                "out": False,
            }
        ],
    )

    report = run_telegram_export_import(
        TelegramExportImportConfig(
            export_dir=export_dir,
            allowed_root=tmp_path,
            timeline_db=tmp_path / "timeline.sqlite",
        )
    )

    assert report["validation_ok"] is True
    assert report["counters"]["dialogs"] == 2
    assert report["counters"]["messages"] == 2
    assert report["counters"]["groups"] == 1
    assert report["counters"]["skipped"] == 2
    assert report["counters"]["imported"] == 0


def test_out_message_imports_as_outbound_and_keeps_brand_tag(tmp_path: Path) -> None:
    export_dir = write_export(
        tmp_path,
        dialogs=[
            {"dialog_id": 100, "name": "Мария", "peer_kind": "user", "phone": None},
        ],
        messages=[
            {
                "dialog_id": 100,
                "dialog_name": "Мария",
                "peer_kind": "user",
                "message_id": 200,
                "date": "2026-03-28T09:38:18+00:00",
                "sender_id": 999,
                "text": "Отправили расписание",
                "out": True,
            }
        ],
    )
    timeline_db = tmp_path / "timeline.sqlite"

    report = run_telegram_export_import(
        TelegramExportImportConfig(
            export_dir=export_dir,
            allowed_root=tmp_path,
            timeline_db=timeline_db,
            brand="unpk",
            apply=True,
        )
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    chunk = fetch_one_json(timeline_db, "bot_context_chunks")
    assert report["counters"]["imported"] == 1
    assert report["counters"]["session_only"] == 1
    assert event["direction"] == "outbound"
    assert event["source_id"] == "telegram:100:200"
    assert "brand:unpk" in chunk["relevance_tags"]


def test_phone_null_does_not_crash_and_stays_session_only(tmp_path: Path) -> None:
    export_dir = write_export(
        tmp_path,
        dialogs=[
            {"dialog_id": 101, "name": "Клиент", "peer_kind": "user", "phone": None},
        ],
        messages=[
            {
                "dialog_id": 101,
                "peer_kind": "user",
                "message_id": 201,
                "date": "2026-03-28T09:38:18+00:00",
                "sender_id": 101,
                "text": "Подскажите по математике",
                "out": False,
            }
        ],
    )
    timeline_db = tmp_path / "timeline.sqlite"

    report = run_telegram_export_import(
        TelegramExportImportConfig(
            export_dir=export_dir,
            allowed_root=tmp_path,
            timeline_db=timeline_db,
            apply=True,
        )
    )

    links = fetch_all_json(timeline_db, "identity_links")
    assert report["counters"]["imported"] == 1
    assert report["counters"]["linked_by_phone"] == 0
    assert report["counters"]["session_only"] == 1
    assert {item["link_type"] for item in links} == {"telegram_user_id", "channel_session_id"}


def test_phone_links_existing_customer_and_repeat_import_does_not_duplicate(tmp_path: Path) -> None:
    timeline_db = tmp_path / "timeline.sqlite"
    existing_customer_id = seed_customer(timeline_db, tmp_path, phone="+79991112233")
    export_dir = write_export(
        tmp_path,
        dialogs=[
            {"dialog_id": 102, "name": "Пётр", "peer_kind": "user", "phone": "9991112233"},
        ],
        messages=[
            {
                "dialog_id": 102,
                "dialog_name": "Пётр",
                "peer_kind": "user",
                "message_id": 202,
                "date": "2026-03-28T09:38:18+00:00",
                "sender_id": 102,
                "text": "Есть курс для 7 класса?",
                "out": False,
            }
        ],
    )
    config = TelegramExportImportConfig(
        export_dir=export_dir,
        allowed_root=tmp_path,
        timeline_db=timeline_db,
        brand="foton",
        apply=True,
    )

    first = run_telegram_export_import(config)
    second = run_telegram_export_import(config)

    event = fetch_one_json(timeline_db, "timeline_events")
    assert first["counters"]["imported"] == 1
    assert first["counters"]["linked_by_phone"] == 1
    assert first["counters"]["duplicates"] == 0
    assert second["counters"]["imported"] == 0
    assert second["counters"]["duplicates"] == 1
    assert event["customer_id"] == existing_customer_id
    assert count_rows(timeline_db, "timeline_events") == 1


def write_export(
    root: Path,
    *,
    dialogs: list[dict[str, object]],
    messages: list[dict[str, object]],
) -> Path:
    export_dir = root / "telegram_export"
    export_dir.mkdir()
    write_jsonl(export_dir / "dialogs.jsonl", dialogs)
    write_jsonl(export_dir / "messages.jsonl", messages)
    return export_dir


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def seed_customer(db_path: Path, allowed_root: Path, *, phone: str) -> str:
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root)
    customer = CustomerIdentity(
        tenant_id="foton",
        identity_status=IdentityStatus.STRONG,
        display_name="Existing",
        primary_phone=phone,
        source_ref="amocrm:contact:1",
    )
    try:
        store.upsert_customer(customer, actor="test")
    finally:
        store.close()
    return customer.customer_id


def fetch_one_json(db_path: Path, table: str) -> dict[str, object]:
    rows = fetch_all_json(db_path, table)
    assert len(rows) == 1
    return rows[0]


def fetch_all_json(db_path: Path, table: str) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as con:
        return [json.loads(row[0]) for row in con.execute(f"SELECT record_json FROM {table}")]


def count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
