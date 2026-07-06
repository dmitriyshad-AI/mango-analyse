from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mango_mvp.integrations.draft_loop import wappi_message_from_raw

from .models import ReplayMessage


RAW_ROOT = Path("~/.mango_local/replay_exam/raw").expanduser()


class WappiReadClient(Protocol):
    def get_chat_messages(
        self,
        *,
        channel: str,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        ...


def assert_raw_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    raw_root = RAW_ROOT.resolve()
    try:
        resolved.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"raw replay export must stay under {raw_root}") from exc
    return resolved


def _payload_messages(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: Any = payload.get("messages")
    if candidates is None:
        candidates = payload.get("data")
    if isinstance(candidates, Mapping):
        candidates = candidates.get("messages") or candidates.get("items") or candidates.get("data")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        return []
    return [item for item in candidates if isinstance(item, Mapping)]


def fetch_messages_paginated(
    client: WappiReadClient,
    *,
    channel: str,
    profile_id: str,
    chat_id: str,
    page_limit: int = 100,
    max_pages: int = 50,
) -> list[ReplayMessage]:
    if page_limit < 1 or page_limit > 100:
        raise ValueError("page_limit must be 1..100")
    seen: set[str] = set()
    messages: list[ReplayMessage] = []
    offset = 0
    for _ in range(max_pages):
        payload = client.get_chat_messages(
            channel=channel,
            profile_id=profile_id,
            chat_id=chat_id,
            limit=page_limit,
            offset=offset,
            order="desc",
            mark_all=False,
        )
        raw_items = _payload_messages(payload)
        if not raw_items:
            break
        for raw in raw_items:
            parsed = wappi_message_from_raw(profile_id, raw)
            if parsed is None or parsed.message_type != "text" or not parsed.text.strip():
                continue
            key = f"{parsed.profile_id}:{parsed.chat_id}:{parsed.message_id}"
            if key in seen:
                continue
            seen.add(key)
            messages.append(
                ReplayMessage(
                    profile_id=parsed.profile_id,
                    chat_id=parsed.chat_id,
                    message_id=parsed.message_id,
                    text=parsed.text,
                    timestamp=parsed.timestamp,
                    from_me=parsed.from_me,
                    sender_name=parsed.contact_name,
                    raw=parsed.raw,
                )
            )
        if len(raw_items) < page_limit:
            break
        offset += page_limit
    messages.sort(key=lambda item: (item.timestamp, item.message_id))
    return messages


def write_raw_dialog_dump(path: Path, *, profile_id: str, chat_id: str, messages: Sequence[ReplayMessage]) -> Path:
    resolved = assert_raw_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wappi_replay_raw_v1",
        "profile_id": profile_id,
        "chat_id": chat_id,
        "messages": [
            {
                "profile_id": item.profile_id,
                "chat_id": item.chat_id,
                "message_id": item.message_id,
                "text": item.text,
                "timestamp": item.timestamp,
                "from_me": item.from_me,
                "sender_name": item.sender_name,
                "raw": dict(item.raw),
            }
            for item in messages
        ],
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved
