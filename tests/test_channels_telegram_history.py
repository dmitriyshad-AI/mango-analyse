from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.channels import (
    TELEGRAM_HISTORY_CHANNEL,
    TELEGRAM_IDENTITY_AMBIGUOUS,
    TELEGRAM_IDENTITY_STRONG_UNIQUE,
    TELEGRAM_IDENTITY_UNMATCHED,
    ChannelSQLiteStore,
    CustomerIdentityRecord,
    TelegramIdentityObservation,
    build_telegram_history_inventory,
    build_telegram_identity_links,
    build_telegram_matching_report,
    import_telegram_history_export,
    iter_telegram_history_messages,
    read_tallanto_identity_records,
    read_telegram_dialog_identity_observations,
    scrub_sensitive_report_payload,
    telegram_history_safety_contract,
    telegram_message_timeline_event,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_export(root: Path) -> Path:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "since": "2024-04-01T00:00:00+00:00",
                "total_dialogs": 3,
                "total_messages": 4,
                "skipped_dialogs": 0,
                "base_tdata": "/private/source/should/not/be/reported",
                "out_dir": "/private/out/should/not/be/reported",
                "finished_at": "2026-04-15T23:48:59+00:00",
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        root / "dialogs.jsonl",
        [
            {
                "dialog_id": 111,
                "name": "Client One",
                "peer_kind": "user",
                "unread_count": 0,
                "folder_id": None,
                "is_user": True,
                "is_group": False,
                "is_channel": False,
                "top_message_id": 10,
                "top_message_date": "2026-04-10T10:00:00+00:00",
            },
            {
                "dialog_id": 222,
                "name": "",
                "peer_kind": "chat",
                "unread_count": 0,
                "folder_id": None,
                "is_user": False,
                "is_group": True,
                "is_channel": False,
                "top_message_id": 20,
                "top_message_date": "2026-04-11T10:00:00+00:00",
            },
        ],
    )
    write_jsonl(
        root / "messages.jsonl",
        [
            {
                "dialog_id": 111,
                "dialog_name": "Client One",
                "peer_kind": "user",
                "message_id": 1,
                "date": "2026-04-10T10:00:00+00:00",
                "sender_id": 111,
                "text": "private message text",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
            {
                "dialog_id": 111,
                "dialog_name": "Client One",
                "peer_kind": "user",
                "message_id": 2,
                "date": "2026-04-10T10:01:00+00:00",
                "sender_id": 999,
                "text": "",
                "out": True,
                "reply_to_msg_id": 1,
                "has_media": True,
            },
            {
                "dialog_id": 222,
                "dialog_name": "",
                "peer_kind": "chat",
                "message_id": 3,
                "date": "2026-04-11T10:01:00+00:00",
                "sender_id": 222,
                "text": "",
                "out": False,
                "reply_to_msg_id": None,
                "has_media": False,
            },
        ],
    )
    return root


def test_telegram_history_inventory_is_aggregate_only(tmp_path) -> None:
    export_dir = build_export(tmp_path / "telegram_export")

    inventory = build_telegram_history_inventory(export_dir)
    payload = inventory.to_json_dict()

    assert payload["dialogs_total"] == 2
    assert payload["messages_total"] == 3
    assert payload["peer_kind_counts"] == {"user": 1, "chat": 1}
    assert payload["message_peer_kind_counts"] == {"user": 2, "chat": 1}
    assert payload["direction_counts"] == {"inbound": 2, "outbound": 1}
    assert payload["content_counts"] == {"text_only": 1, "media_only": 1, "empty_no_media": 1}
    assert payload["identity_field_presence"]["telegram_id"] == 2
    assert payload["identity_field_presence"]["name"] == 1
    assert "private message text" not in str(payload)
    assert "should/not/be/reported" not in str(payload)
    assert payload["safety"]["telegram_api_called"] is False


def test_telegram_history_parser_outputs_channel_messages_and_timeline_events(tmp_path) -> None:
    export_dir = build_export(tmp_path / "telegram_export")

    messages = tuple(iter_telegram_history_messages(export_dir))

    assert len(messages) == 2
    assert messages[0].channel == TELEGRAM_HISTORY_CHANNEL
    assert messages[0].channel_thread_id == "111"
    assert messages[0].channel_message_id == "1"
    assert messages[0].direction.value == "inbound"
    assert messages[0].metadata["parser_mode"] == "read_only"
    assert messages[0].metadata["telegram_dialog_name_present"] is True
    assert messages[1].direction.value == "outbound"
    assert messages[1].attachments[0].kind == "telegram_history_media"

    redacted_event = telegram_message_timeline_event(messages[0])
    visible_event = telegram_message_timeline_event(messages[0], include_text_preview=True)
    assert redacted_event["event_type"] == "telegram_message"
    assert redacted_event["text_preview"] is None
    assert redacted_event["text_preview_redacted"] is True
    assert visible_event["text_preview"] == "private message text"


def test_telegram_history_import_is_idempotent_and_uses_sqlite_store(tmp_path) -> None:
    export_dir = build_export(tmp_path / "telegram_export")
    db_path = tmp_path / "channel_archive.sqlite"

    first = import_telegram_history_export(export_dir, db_path)
    second = import_telegram_history_export(export_dir, db_path)

    assert first.messages_seen == 3
    assert first.messages_created == 2
    assert first.messages_skipped_empty == 1
    assert second.messages_created == 0
    assert second.messages_duplicate == 2
    with ChannelSQLiteStore.open_read_only(db_path) as store:
        snapshot = store.snapshot(include_raw_payload=True)
    assert snapshot["summary"]["messages"] == 2
    assert "raw_payload" not in str(snapshot["messages"])
    assert snapshot["safety"]["write_crm"] is False


def test_telegram_identity_matching_classes_strong_ambiguous_unmatched() -> None:
    observations = (
        TelegramIdentityObservation(channel_thread_id="111", telegram_user_id="111", phone="+7 900 000-00-01"),
        TelegramIdentityObservation(channel_thread_id="222", username="@shared_user"),
        TelegramIdentityObservation(channel_thread_id="333", display_name="Name Only"),
    )
    candidates = (
        CustomerIdentityRecord(
            customer_id="tallanto:1",
            source_system="tallanto",
            phones=("+79000000001",),
            telegram_user_ids=("111",),
        ),
        CustomerIdentityRecord(
            customer_id="amo:2",
            source_system="amocrm",
            telegram_usernames=("shared_user",),
        ),
        CustomerIdentityRecord(
            customer_id="tallanto:3",
            source_system="tallanto",
            telegram_usernames=("shared_user",),
        ),
    )

    links = build_telegram_identity_links(observations, candidates)
    report = build_telegram_matching_report(links, high_utility_thread_ids=("111", "333"))

    assert [link.match_class for link in links] == [
        TELEGRAM_IDENTITY_STRONG_UNIQUE,
        TELEGRAM_IDENTITY_AMBIGUOUS,
        TELEGRAM_IDENTITY_UNMATCHED,
    ]
    assert links[0].candidate_customer_ids == ("tallanto:1",)
    assert links[0].confidence == 0.98
    assert "multiple_candidate_customers" in links[1].conflict_flags
    assert "name_only_not_matched" in links[2].conflict_flags
    assert report["class_counts"][TELEGRAM_IDENTITY_AMBIGUOUS] == 1
    assert report["high_utility_class_counts"][TELEGRAM_IDENTITY_STRONG_UNIQUE] == 1
    assert report["safety"]["write_tallanto"] is False


def test_tallanto_identity_records_use_telegram_id_without_leaking_password_columns(tmp_path) -> None:
    csv_path = tmp_path / "students.csv"
    csv_path.write_text(
        (
            "ID;Тел. цифровой (моб.);Telegram ID;Telegram;Подписан в Telegram;Хэш пароля ЛК/МП\n"
            "42;8 900 000 00 02;777;@student_parent;да;must-not-leak\n"
        ),
        encoding="cp1251",
    )

    records = read_tallanto_identity_records(csv_path)
    observation = TelegramIdentityObservation(channel_thread_id="777", telegram_user_id="777")
    links = build_telegram_identity_links((observation,), records)

    assert records[0].customer_id == "tallanto:42"
    assert records[0].phones == ("+79000000002",)
    assert records[0].telegram_user_ids == ("777",)
    assert records[0].telegram_usernames == ("student_parent",)
    assert records[0].metadata["telegram_subscribed"] is True
    assert "must-not-leak" not in str(records[0].to_json_dict())
    assert links[0].match_class == TELEGRAM_IDENTITY_STRONG_UNIQUE


def test_telegram_history_safety_blocks_live_send_api_and_sensitive_report_values() -> None:
    safety = telegram_history_safety_contract()
    scrubbed = scrub_sensitive_report_payload(
        {
            "bot_token": "123:secret",
            "nested": {"password": "plain", "safe": "ok"},
            "items": [{"authorization": "Bearer secret"}],
        }
    )

    assert safety["network_calls"] is False
    assert safety["telegram_api_called"] is False
    assert safety["live_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["imports_legacy_rag"] is False
    assert scrubbed["bot_token"] == "[redacted]"
    assert scrubbed["nested"]["password"] == "[redacted]"
    assert scrubbed["nested"]["safe"] == "ok"
    assert scrubbed["items"][0]["authorization"] == "[redacted]"


def test_dialog_observations_keep_names_out_of_default_json(tmp_path) -> None:
    export_dir = build_export(tmp_path / "telegram_export")

    observations = read_telegram_dialog_identity_observations(export_dir)
    payload = observations[0].to_json_dict()

    assert observations[0].telegram_user_id == "111"
    assert payload["display_name_present"] is True
    assert "Client One" not in str(payload)
