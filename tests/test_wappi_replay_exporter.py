from __future__ import annotations

from pathlib import Path

import pytest

from mango_mvp.integrations.draft_loop import DraftLoopProfile
from mango_mvp.replay_exam import exporter
from mango_mvp.replay_exam.exporter import (
    assert_raw_output_path,
    export_recent_dialogs,
    fetch_chats_paginated,
    fetch_messages_paginated,
)


class FakeWappiClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_chats(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"kind": "list_chats", **dict(kwargs)})
        offset = int(kwargs["offset"])
        pages = {
            0: [
                {"id": "chat-1", "name": "Первый"},
                {"id": "chat-2", "name": "Второй"},
            ],
            2: [
                {"id": "chat-3", "name": "Третий"},
            ],
        }
        return {"chats": pages.get(offset, [])}

    def get_chat_messages(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"kind": "get_chat_messages", **dict(kwargs)})
        offset = int(kwargs["offset"])
        chat_id = kwargs.get("chat_id")
        if chat_id == "chat-2":
            return {"messages": [{"id": "m-only", "chatId": "chat-2", "type": "text", "body": "Один", "time": 1, "fromMe": False}]}
        pages = {
            0: [
                {"id": "m2", "chatId": "c1", "type": "text", "body": "Второе", "time": 20, "fromMe": True},
                {"id": "m1", "chatId": "c1", "type": "text", "body": "Первое", "time": 10, "fromMe": False},
            ],
            2: [
                {"id": "m0", "chatId": "c1", "type": "text", "body": "Нулевое", "time": 5, "fromMe": False},
            ],
        }
        return {"messages": pages.get(offset, [])}


def test_fetch_chats_paginated_uses_readonly_show_all_false() -> None:
    client = FakeWappiClient()
    chats = fetch_chats_paginated(client, channel="max", profile_id="p1", page_limit=2)

    assert [chat["id"] for chat in chats] == ["chat-1", "chat-2", "chat-3"]
    calls = [call for call in client.calls if call["kind"] == "list_chats"]
    assert [call["offset"] for call in calls] == [0, 2]
    assert all(call["show_all"] is False for call in calls)
    assert all(call["limit"] == 2 for call in calls)


def test_fetch_messages_paginated_uses_mark_all_false_and_sorts() -> None:
    client = FakeWappiClient()
    messages = fetch_messages_paginated(client, channel="telegram", profile_id="p1", chat_id="c1", page_limit=2)

    calls = [call for call in client.calls if call["kind"] == "get_chat_messages"]
    assert [call["offset"] for call in calls] == [0, 2]
    assert all(call["mark_all"] is False for call in calls)
    assert all(call["limit"] == 2 for call in calls)
    assert [message.message_id for message in messages] == ["m0", "m1", "m2"]


def test_export_recent_dialogs_writes_only_qualified_dialogs_under_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "RAW_ROOT", tmp_path / "raw")
    client = FakeWappiClient()
    manifest = export_recent_dialogs(
        client,
        profiles=[DraftLoopProfile(profile_id="p1", brand="foton", channel="telegram")],
        raw_root=tmp_path / "raw/unit-test",
        per_profile=2,
        chat_page_limit=2,
        message_page_limit=2,
        max_chat_pages=2,
        max_message_pages=2,
    )

    assert manifest["dialog_count"] == 2
    assert manifest["read_contract"]["writes_external_systems"] is False
    assert manifest["read_contract"]["mark_all"] is False
    assert all(call.get("method", "GET") == "GET" for call in client.calls)
    message_calls = [call for call in client.calls if call["kind"] == "get_chat_messages"]
    assert all(call["mark_all"] is False for call in message_calls)
    chat_calls = [call for call in client.calls if call["kind"] == "list_chats"]
    assert all(call["show_all"] is False for call in chat_calls)


def test_raw_output_path_must_stay_under_local_replay_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_raw_output_path(tmp_path / "raw.json")
