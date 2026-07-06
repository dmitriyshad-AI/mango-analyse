from __future__ import annotations

from pathlib import Path

import pytest

from mango_mvp.replay_exam.exporter import assert_raw_output_path, fetch_messages_paginated


class FakeWappiClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_chat_messages(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(dict(kwargs))
        offset = int(kwargs["offset"])
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


def test_fetch_messages_paginated_uses_mark_all_false_and_sorts() -> None:
    client = FakeWappiClient()
    messages = fetch_messages_paginated(client, channel="telegram", profile_id="p1", chat_id="c1", page_limit=2)

    assert [call["offset"] for call in client.calls] == [0, 2]
    assert all(call["mark_all"] is False for call in client.calls)
    assert all(call["limit"] == 2 for call in client.calls)
    assert [message.message_id for message in messages] == ["m0", "m1", "m2"]


def test_raw_output_path_must_stay_under_local_replay_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_raw_output_path(tmp_path / "raw.json")
