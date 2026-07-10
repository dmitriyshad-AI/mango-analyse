from __future__ import annotations

import json
from pathlib import Path

from scripts import build_wappi_replay_cases as build_cases


def test_build_cases_scrubs_raw_dialog_and_preserves_prefix_memory(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw_root = tmp_path / "raw"
    scrubbed_root = tmp_path / "scrubbed"
    monkeypatch.setattr(build_cases, "RAW_ROOT", raw_root)
    monkeypatch.setattr(build_cases, "SCRUBBED_ROOT", scrubbed_root)

    raw_run = raw_root / "run1"
    raw_run.mkdir(parents=True)
    raw_dialog = raw_run / "dialog.json"
    raw_dialog.write_text(
        json.dumps(
            {
                "schema_version": "wappi_replay_raw_v1",
                "profile_id": "profile-real",
                "chat_id": "79001234567",
                "messages": [
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m1",
                        "text": "Мария Иванова, телефон +7 999 123-45-67",
                        "timestamp": 10,
                        "from_me": False,
                        "sender_name": "Мария Иванова",
                        "raw": {"lead_id": "123456"},
                    },
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m2",
                        "text": "Интересует физика",
                        "timestamp": 20,
                        "from_me": False,
                        "sender_name": "Мария Иванова",
                    },
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m3",
                        "text": "Добрый день, расскажу про формат занятий и условия записи.",
                        "timestamp": 80,
                        "from_me": True,
                        "sender_name": "Менеджер",
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_manifest = raw_run / "manifest.json"
    raw_manifest.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "selected_dialogs": [
                            {
                                "profile_id": "profile-real",
                                "chat_id": "79001234567",
                                "brand": "foton",
                                "raw_file": str(raw_dialog),
                            }
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_cases.build_cases_from_raw_manifest(raw_manifest_path=raw_manifest, out_root=scrubbed_root, max_dialogs=1)

    assert manifest["leak_count"] == 0
    cases_path = Path(str(manifest["cases_jsonl"]))
    case = json.loads(cases_path.read_text(encoding="utf-8").splitlines()[0])
    text = repr(manifest) + cases_path.read_text(encoding="utf-8")
    assert "+7 999" not in text
    assert "79001234567" not in text
    assert "profile-real" not in text
    assert "123456" not in text
    assert case["prefix_messages"] == []
    assert "Мария Иванова" not in case["client_message"]
    assert case["schema_version"] == "wappi_replay_case_v4"
    assert isinstance(case["turn_index"], int)
    assert set(case).issuperset({"exam_id", "contour", "dialog_key_masked", "meta"})


def test_build_cases_prefix_messages_use_whitelist_keys(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw_root = tmp_path / "raw"
    scrubbed_root = tmp_path / "scrubbed"
    monkeypatch.setattr(build_cases, "RAW_ROOT", raw_root)
    monkeypatch.setattr(build_cases, "SCRUBBED_ROOT", scrubbed_root)

    raw_run = raw_root / "run1"
    raw_run.mkdir(parents=True)
    raw_dialog = raw_run / "dialog.json"
    raw_dialog.write_text(
        json.dumps(
            {
                "profile_id": "profile-real",
                "chat_id": "79001234567",
                "messages": [
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m1",
                        "text": "Здравствуйте",
                        "timestamp": 10,
                        "from_me": False,
                        "raw": {"from": "79001234567", "to": "79991234567", "phone": "79001234567"},
                    },
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m2",
                        "text": "Добрый день",
                        "timestamp": 20,
                        "from_me": True,
                        "raw": {"chatId": "79001234567@c.us", "username": "real_user"},
                    },
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m3",
                        "text": "Нужна математика",
                        "timestamp": 30,
                        "from_me": False,
                    },
                    {
                        "profile_id": "profile-real",
                        "chat_id": "79001234567",
                        "message_id": "m4",
                        "text": "Ответ менеджера с достаточным смысловым содержанием для эталона",
                        "timestamp": 90,
                        "from_me": True,
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_manifest = raw_run / "manifest.json"
    raw_manifest.write_text(
        json.dumps({"profiles": [{"selected_dialogs": [{"brand": "foton", "raw_file": str(raw_dialog)}]}]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    manifest = build_cases.build_cases_from_raw_manifest(raw_manifest_path=raw_manifest, out_root=scrubbed_root, max_dialogs=1)

    cases = [json.loads(line) for line in Path(str(manifest["cases_jsonl"])).read_text(encoding="utf-8").splitlines()]
    case = next(item for item in cases if item["prefix_messages"])
    prefix = case["prefix_messages"]
    assert prefix
    assert set(prefix[0]) == {"from_me", "text", "ts_masked"}
    assert prefix[0]["from_me"] is False
    assert prefix[1]["from_me"] is True
    serialized = json.dumps(case, ensure_ascii=False)
    forbidden_keys = {"raw", "from", "to", "phone", "chatId", "contact_name", "username", "wappi_bot_id", "task_id", "stanzaId"}
    seen_keys: set[str] = set()

    def walk(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            seen_keys.update(str(key) for key in value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(case)
    assert not forbidden_keys.intersection(seen_keys)
    assert "79001234567" not in serialized
    assert "real_user" not in serialized
