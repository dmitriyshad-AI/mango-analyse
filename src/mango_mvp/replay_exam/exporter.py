from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mango_mvp.integrations.draft_loop import DraftLoopProfile, wappi_message_from_raw

from .models import ReplayMessage


RAW_ROOT = Path("~/.mango_local/replay_exam/raw").expanduser()


class WappiReadClient(Protocol):
    def list_chats(
        self,
        *,
        channel: str,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        ...

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


def _payload_chats(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: Any = payload.get("chats")
    if candidates is None:
        candidates = payload.get("dialogs")
    if candidates is None:
        candidates = payload.get("data")
    if isinstance(candidates, Mapping):
        candidates = candidates.get("chats") or candidates.get("dialogs") or candidates.get("items") or candidates.get("data")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        return []
    return [item for item in candidates if isinstance(item, Mapping)]


def chat_id_from_raw(raw: Mapping[str, Any]) -> str:
    return str(
        raw.get("chatId")
        or raw.get("chat_id")
        or raw.get("id")
        or raw.get("dialog_id")
        or raw.get("phone")
        or ""
    ).strip()


def fetch_chats_paginated(
    client: WappiReadClient,
    *,
    channel: str,
    profile_id: str,
    page_limit: int = 100,
    max_pages: int = 20,
    show_all: bool = False,
) -> list[Mapping[str, Any]]:
    if page_limit < 1 or page_limit > 100:
        raise ValueError("page_limit must be 1..100")
    seen: set[str] = set()
    chats: list[Mapping[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        payload = client.list_chats(
            channel=channel,
            profile_id=profile_id,
            limit=page_limit,
            offset=offset,
            order="desc",
            show_all=show_all,
        )
        raw_items = _payload_chats(payload)
        if not raw_items:
            break
        for raw in raw_items:
            chat_id = chat_id_from_raw(raw)
            if not chat_id or chat_id in seen:
                continue
            seen.add(chat_id)
            chats.append(dict(raw))
        if len(raw_items) < page_limit:
            break
        offset += page_limit
    return chats


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


def _timestamp_seconds(value: int) -> int:
    raw = int(value or 0)
    return raw // 1000 if raw > 10_000_000_000 else raw


def _looks_like_internal_test(messages: Sequence[ReplayMessage]) -> bool:
    text = "\n".join(item.text for item in messages).casefold()
    return any(marker in text for marker in ("тестовый диалог", "test dialog", "codex test", "тест codex", "проверка бота"))


def qualifies_for_replay(
    messages: Sequence[ReplayMessage],
    *,
    min_client_messages: int = 2,
    min_manager_messages: int = 1,
    min_manager_reference_chars: int = 30,
    max_age_days: int = 90,
) -> bool:
    client_messages = sum(1 for item in messages if item.is_client)
    manager_messages = sum(1 for item in messages if item.is_manager)
    if client_messages < min_client_messages or manager_messages < min_manager_messages:
        return False
    if _looks_like_internal_test(messages):
        return False
    manager_chars = sum(len(item.text.strip()) for item in messages if item.is_manager)
    if manager_chars < min_manager_reference_chars:
        return False
    if max_age_days > 0 and messages:
        latest = max(_timestamp_seconds(item.timestamp) for item in messages)
        if latest > 1_500_000_000 and time.time() - latest > max_age_days * 24 * 60 * 60:
            return False
    return True


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


def export_recent_dialogs(
    client: WappiReadClient,
    *,
    profiles: Sequence[DraftLoopProfile],
    raw_root: Path = RAW_ROOT,
    per_profile: int = 25,
    chat_page_limit: int = 100,
    max_chat_pages: int = 20,
    message_page_limit: int = 100,
    max_message_pages: int = 50,
) -> Mapping[str, Any]:
    root = assert_raw_output_path(raw_root / "manifest.json").parent
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "wappi_replay_raw_manifest_v1",
        "raw_root": str(root),
        "profiles": [],
        "dialog_count": 0,
        "message_count": 0,
        "read_contract": {
            "wappi_methods": ["list_chats", "get_chat_messages"],
            "mark_all": False,
            "show_all": False,
            "writes_external_systems": False,
        },
    }
    for profile in profiles:
        profile_rows: list[dict[str, Any]] = []
        chats = fetch_chats_paginated(
            client,
            channel=profile.channel,
            profile_id=profile.profile_id,
            page_limit=chat_page_limit,
            max_pages=max_chat_pages,
            show_all=False,
        )
        for raw_chat in chats:
            if len(profile_rows) >= max(1, per_profile):
                break
            chat_id = chat_id_from_raw(raw_chat)
            if not chat_id:
                continue
            messages = fetch_messages_paginated(
                client,
                channel=profile.channel,
                profile_id=profile.profile_id,
                chat_id=chat_id,
                page_limit=message_page_limit,
                max_pages=max_message_pages,
            )
            if not qualifies_for_replay(messages):
                continue
            safe_profile = profile.profile_id.replace("/", "_")
            safe_chat = chat_id.replace("/", "_")
            file_path = root / f"{profile.channel}_{profile.brand}_{safe_profile}_{safe_chat}.json"
            write_raw_dialog_dump(file_path, profile_id=profile.profile_id, chat_id=chat_id, messages=messages)
            profile_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "channel": profile.channel,
                    "brand": profile.brand,
                    "chat_id": chat_id,
                    "raw_file": str(file_path),
                    "messages": len(messages),
                    "client_messages": sum(1 for item in messages if item.is_client),
                    "manager_messages": sum(1 for item in messages if item.is_manager),
                }
            )
            manifest["dialog_count"] += 1
            manifest["message_count"] += len(messages)
        manifest["profiles"].append(
            {
                "profile_id": profile.profile_id,
                "channel": profile.channel,
                "brand": profile.brand,
                "selected_dialogs": profile_rows,
                "selected_count": len(profile_rows),
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
