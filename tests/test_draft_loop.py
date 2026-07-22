from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.dialogue_memory import DIALOG_SUMMARY_ROLLING_ENV, MEMORY_PROVENANCE_ENV
from mango_mvp.integrations.amo_wappi_phase1 import AmoWappiHttpError
from mango_mvp.integrations.draft_loop import (
    AmoWappiDraftLoop,
    DraftLoopConfig,
    DraftLoopConfigError,
    DraftLoopJournal,
    DraftLoopKey,
    DraftLoopPair,
    DraftLoopProfile,
    DraftLoopState,
    DraftWindow,
    MAX_DEFERRED_PAIR_MISSING,
    OutgoingWindowMessage,
    WappiHistoryMessage,
    _auto_pair_note,
    _prompt_history_lines,
    build_draft_loop_config_fingerprint,
    build_draft_loop_code_identity,
    classify_manager_edit_windows,
    load_pairs_file,
    load_profiles_file,
    persist_auto_pair,
)


class FakeWappi:
    def __init__(self, dialogs, messages_by_chat) -> None:
        self.dialogs = dialogs
        self.messages_by_chat = messages_by_chat
        self.list_calls = 0
        self.message_calls = []

    def list_telegram_chats(self, *, profile_id: str, limit: int = 50):
        self.list_calls += 1
        return {"dialogs": self.dialogs.get(profile_id, [])}

    def get_telegram_chat_messages(self, *, profile_id: str, chat_id: str, **kwargs):
        self.message_calls.append({"channel": "telegram", "profile_id": profile_id, "chat_id": chat_id, **kwargs})
        return {"messages": self.messages_by_chat.get((profile_id, chat_id), [])}

    def list_chats(self, *, channel: str, profile_id: str, limit: int = 50, offset: int = 0):
        self.list_calls += 1
        return {"dialogs": self.dialogs.get(profile_id, [])[offset : offset + limit]}

    def get_chat_messages(self, *, channel: str, profile_id: str, chat_id: str, **kwargs):
        self.message_calls.append({"channel": channel, "profile_id": profile_id, "chat_id": chat_id, **kwargs})
        return {"messages": self.messages_by_chat.get((profile_id, chat_id), [])}


class FakeAmo:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.notes = []

    def add_draft_note_to_test_lead(self, lead_id, **kwargs):
        if self.fail:
            raise RuntimeError("amo down")
        self.notes.append({"lead_id": str(lead_id), **kwargs})
        return {"ok": True}


class FakeBot:
    def __init__(self) -> None:
        self.calls = []

    def build_draft(self, client_message: str, *, context=None):
        self.calls.append({"client_message": client_message, "context": context})
        return SubscriptionDraftResult(
            route="bot_answer_self",
            draft_text=f"Черновик: {client_message}",
            safety_flags=("client_safe_fact_verified",),
        )


def _config(tmp_path: Path, *, pairs=None, config_fingerprint=None) -> DraftLoopConfig:
    profile = DraftLoopProfile(profile_id="profile-foton", brand="foton", channel="telegram")
    return DraftLoopConfig(
        profiles={profile.profile_id: profile},
        pairs=pairs or {},
        allowed_test_lead_ids=frozenset({"49832125"}),
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "journal.jsonl",
        manager_edit_log_path=tmp_path / "manager_edits.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        stop_path=tmp_path / "STOP_DRAFT_LOOP",
        debounce_seconds=60,
        config_fingerprint=config_fingerprint or {},
    )


def _message(message_id: str, *, chat_id: str = "chat-1", text: str = "Цена?", ts: int = 1000, from_me: bool = False, typ: str = "text"):
    return {
        "id": message_id,
        "chatId": chat_id,
        "body": text,
        "type": typ,
        "time": ts,
        "fromMe": from_me,
        "contact_name": "Client",
    }


def test_prompt_history_uses_rolling_dialog_summary_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv(DIALOG_SUMMARY_ROLLING_ENV, "1")
    messages = tuple(
        WappiHistoryMessage(
            profile_id="profile-foton",
            chat_id="chat-1",
            message_id=f"m{idx}",
            text=f"старое сообщение {idx}",
            message_type="text",
            timestamp=idx,
            from_me=False,
        )
        for idx in range(20)
    )

    lines = _prompt_history_lines(
        messages,
        recent_limit=3,
        brand="foton",
        dialogue_memory={
            "conversation_summary_short": "Клиент выбирает физику для 7 класса; ждёт следующий шаг.",
        },
    )

    assert lines[0] == "Ранее в диалоге: Клиент выбирает физику для 7 класса; ждёт следующий шаг."
    assert "старое сообщение 0" not in " ".join(lines)


def test_prompt_history_falls_back_to_raw_summary_when_rolling_memory_empty(monkeypatch) -> None:
    monkeypatch.setenv(DIALOG_SUMMARY_ROLLING_ENV, "1")
    messages = tuple(
        WappiHistoryMessage(
            profile_id="profile-foton",
            chat_id="chat-1",
            message_id=f"m{idx}",
            text=f"старое сообщение {idx}",
            message_type="text",
            timestamp=idx,
            from_me=False,
        )
        for idx in range(20)
    )

    lines = _prompt_history_lines(
        messages,
        recent_limit=3,
        brand="foton",
        dialogue_memory={},
    )

    assert lines[0].startswith("Ранее в диалоге: Клиент: старое сообщение 0")


def test_prompt_history_falls_back_when_rolling_summary_has_foreign_brand(monkeypatch) -> None:
    monkeypatch.setenv(DIALOG_SUMMARY_ROLLING_ENV, "1")
    messages = tuple(
        WappiHistoryMessage(
            profile_id="profile-foton",
            chat_id="chat-1",
            message_id=f"m{idx}",
            text=f"старое сообщение {idx}",
            message_type="text",
            timestamp=idx,
            from_me=False,
        )
        for idx in range(20)
    )

    lines = _prompt_history_lines(
        messages,
        recent_limit=3,
        brand="foton",
        dialogue_memory={"conversation_summary_short": "Клиент сравнивает Фотон и УНПК МФТИ."},
    )

    assert lines[0].startswith("Ранее в диалоге: Клиент: старое сообщение 0")


def _loop(tmp_path: Path, *, messages, pairs=None, stop: bool = False, auto_resolver=None, amo=None, bot=None) -> AmoWappiDraftLoop:
    cfg = _config(tmp_path, pairs=pairs)
    if stop:
        cfg.stop_path.write_text("stop", encoding="utf-8")
    wappi = FakeWappi({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {("profile-foton", "chat-1"): messages})
    return AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=amo or FakeAmo(),
        bot_provider=bot or FakeBot(),
        context_builder=lambda key, history, client_message, brand: {
            "key": key.value,
            "history": list(history),
            "client_message": client_message,
            "brand": brand,
        },
        auto_resolver=auto_resolver,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )


def test_draft_loop_uses_composite_key_and_writes_single_note(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1"), _message("m2", text="А онлайн есть?", ts=1010)], pairs={key: pair}, amo=amo, bot=bot)

    summary = loop.run_once(dry_run=False)

    assert summary["processed"] == 2
    assert summary["bot_calls"] == 1
    assert bot.calls[0]["client_message"] == "А онлайн есть?"
    assert bot.calls[0]["context"]["history"][-2:] == ["Клиент: Цена?", "Клиент: А онлайн есть?"]
    assert len(amo.notes) == 1
    assert amo.notes[0]["lead_id"] == "49832125"
    assert amo.notes[0]["route"] == "bot_answer_self"
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert {item["message_id"] for item in state["processed"]} == {"m1", "m2"}


def test_draft_loop_wappi_prompt_summarizes_older_context_and_keeps_recent_order(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    older = [
        _message("m00", text="Сын в 7 классе, интересует физика онлайн", ts=1000),
        _message("m01", text="message_id=abc chat_id=chat-1 profile_id=p lead_id=1 source_system={wappi}", ts=1001),
        _message("m02", text="УНПК МФТИ тоже спрашивали, но это другой бренд", ts=1002),
    ]
    middle = [_message(f"m{idx:02d}", text=f"старый уточняющий текст {idx}", ts=1000 + idx) for idx in range(3, 35)]
    recent = [_message(f"m{idx:02d}", text=f"последний сырой текст {idx}", ts=1000 + idx) for idx in range(35, 50)]
    wappi = FakeWappi({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {("profile-foton", "chat-1"): [*older, *middle, *recent]})
    bot = FakeBot()
    cfg = _config(tmp_path, pairs={key: pair})
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {
            "history": list(history),
            "client_message": client_message,
            "brand": brand,
        },
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=True)

    assert summary["bot_calls"] == 1
    assert wappi.message_calls[0]["limit"] == 50
    history = bot.calls[0]["context"]["history"]
    assert history[0].startswith("Ранее в диалоге:")
    assert "7 классе" in history[0]
    assert "физика онлайн" in history[0]
    forbidden = ("message_id", "chat_id", "profile_id", "lead_id", "source_system", "{", "}", "УНПК", "МФТИ")
    assert not any(marker in "\n".join(history) for marker in forbidden)
    assert history[1:] == [f"Клиент: последний сырой текст {idx}" for idx in range(35, 50)]
    assert bot.calls[0]["client_message"] == "последний сырой текст 49"


def test_draft_loop_skips_messages_at_or_before_not_before_ts_and_zero_timestamp(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton", not_before_ts=1000)
    bot = FakeBot()
    loop = _loop(
        tmp_path,
        messages=[
            _message("m0", ts=0, text="битое время"),
            _message("m1", ts=1000, text="старое"),
            _message("m2", ts=1010, text="новое"),
        ],
        pairs={key: pair},
        bot=bot,
    )

    summary = loop.run_once(dry_run=False)

    assert summary["processed"] == 1
    assert summary["skipped"] == 2
    assert summary["bot_calls"] == 1
    assert bot.calls[0]["client_message"] == "новое"
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    skipped = [row for row in rows if row["event"] == "not_before_skipped"]
    assert {row["message_id"] for row in skipped} == {"m0", "m1"}


def test_draft_loop_auto_pair_is_persisted_once_and_keeps_current_message(tmp_path: Path) -> None:
    auto_pairs = tmp_path / "auto_pairs.json"
    cfg = _config(tmp_path)
    cfg = DraftLoopConfig(
        profiles=cfg.profiles,
        pairs=cfg.pairs,
        auto_pairs_path=auto_pairs,
        allowed_test_lead_ids=cfg.allowed_test_lead_ids,
        state_path=cfg.state_path,
        journal_path=cfg.journal_path,
        manager_edit_log_path=cfg.manager_edit_log_path,
        heartbeat_path=cfg.heartbeat_path,
        stop_path=cfg.stop_path,
        debounce_seconds=cfg.debounce_seconds,
    )
    wappi = FakeWappi({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {("profile-foton", "chat-1"): [_message("m1")]})
    bot = FakeBot()

    def resolver(**kwargs):
        return {"status": "matched", "lead_id": "49762441", "contact_id": "111", "match_key": "Telegram ID"}

    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {},
        auto_resolver=resolver,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=False)
    summary_second = loop.run_once(dry_run=False)

    assert summary["processed"] == 1
    assert summary["skipped"] == 0
    assert summary["bot_calls"] == 1
    assert summary_second["bot_calls"] == 0
    loaded = load_pairs_file(auto_pairs, default_source="auto")
    pair = loaded[DraftLoopKey("profile-foton", "chat-1")]
    assert pair.lead_id == "49762441"
    assert pair.not_before_ts == 1200
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for row in rows if row["event"] == "auto_pair_created") == 1


def test_draft_loop_auto_pair_replays_deferred_pair_missing_once(tmp_path: Path) -> None:
    auto_pairs = tmp_path / "auto_pairs.json"
    cfg = _config(tmp_path)
    cfg = DraftLoopConfig(
        profiles=cfg.profiles,
        pairs=cfg.pairs,
        auto_pairs_path=auto_pairs,
        allowed_test_lead_ids=cfg.allowed_test_lead_ids,
        state_path=cfg.state_path,
        journal_path=cfg.journal_path,
        manager_edit_log_path=cfg.manager_edit_log_path,
        heartbeat_path=cfg.heartbeat_path,
        stop_path=cfg.stop_path,
        debounce_seconds=cfg.debounce_seconds,
    )
    message = _message("m1", ts=1000)
    wappi = FakeWappi(
        {"profile-foton": [{"id": "chat-1", "type": "user"}]},
        {("profile-foton", "chat-1"): [message]},
    )
    amo = FakeAmo()
    bot = FakeBot()
    resolver_calls = 0

    def resolver(**kwargs):
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls == 1:
            return {"status": "rejected", "reason": "no_match"}
        return {"status": "matched", "lead_id": "49832125", "match_key": "Telegram ID"}

    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=amo,
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {},
        auto_resolver=resolver,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    first = loop.run_once(dry_run=False)
    second = loop.run_once(dry_run=False)
    third = loop.run_once(dry_run=False)

    assert first["processed"] == first["bot_calls"] == 0
    assert second["processed"] == second["bot_calls"] == 1
    assert third["processed"] == third["bot_calls"] == 0
    assert len(bot.calls) == len(amo.notes) == 1
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows if row["event"] == "pair_missing"] == ["pair_missing"]
    assert [row["event"] for row in rows if row["event"] == "auto_pair_created"] == ["auto_pair_created"]
    assert not any(row["event"] == "not_before_skipped" and row["message_id"] == "m1" for row in rows)
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert {item["message_id"] for item in state["processed"]} == {"m1"}
    assert state["deferred_pair_missing"] == {}


def test_draft_loop_auto_pair_keeps_message_arriving_with_match(tmp_path: Path) -> None:
    auto_pairs = tmp_path / "auto_pairs.json"
    cfg = _config(tmp_path)
    cfg = DraftLoopConfig(
        profiles=cfg.profiles,
        pairs=cfg.pairs,
        auto_pairs_path=auto_pairs,
        allowed_test_lead_ids=cfg.allowed_test_lead_ids,
        state_path=cfg.state_path,
        journal_path=cfg.journal_path,
        manager_edit_log_path=cfg.manager_edit_log_path,
        heartbeat_path=cfg.heartbeat_path,
        stop_path=cfg.stop_path,
        debounce_seconds=cfg.debounce_seconds,
    )
    first_message = _message("m1", ts=1000)
    second_message = _message("m2", ts=1100)
    wappi = FakeWappi(
        {"profile-foton": [{"id": "chat-1", "type": "user"}]},
        {("profile-foton", "chat-1"): [first_message]},
    )
    resolver_calls = 0

    def resolver(**kwargs):
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls == 1:
            return {"status": "rejected", "reason": "no_match"}
        return {"status": "matched", "lead_id": "49832125", "match_key": "Telegram ID"}

    bot = FakeBot()
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {},
        auto_resolver=resolver,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    first = loop.run_once(dry_run=False)
    wappi.messages_by_chat[("profile-foton", "chat-1")] = [first_message, second_message]
    second = loop.run_once(dry_run=False)

    assert first["processed"] == 0
    assert second["processed"] == 2
    assert second["bot_calls"] == 1
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert {item["message_id"] for item in state["processed"]} == {"m1", "m2"}
    assert state["deferred_pair_missing"] == {}
    assert len(bot.calls) == 1


def test_draft_loop_auto_resolver_failure_is_manual_review_not_crash(tmp_path: Path) -> None:
    def resolver(**kwargs):
        raise RuntimeError("MCP HTTP 429: rate limit")

    loop = _loop(tmp_path, messages=[_message("m1")], auto_resolver=resolver)

    summary = loop.run_once(dry_run=True)

    assert summary["skipped"] == 1
    assert summary["bot_calls"] == 0
    assert summary["auto_resolver_counts"] == {"auto_resolver_unavailable": 1}
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "pair_missing"
    assert rows[-1]["auto_candidate"]["reason"] == "auto_resolver_unavailable"


def test_persist_auto_pair_does_not_duplicate_or_move_watermark(tmp_path: Path) -> None:
    path = tmp_path / "auto_pairs.json"
    key = DraftLoopKey("profile-foton", "chat-1")
    first = DraftLoopPair(key=key, lead_id="49762441", expected_brand="foton", not_before_ts=1200, source="auto")
    second = DraftLoopPair(key=key, lead_id="49762441", expected_brand="foton", not_before_ts=1300, source="auto")

    assert persist_auto_pair(path, first) is True
    assert persist_auto_pair(path, second) is False

    loaded = load_pairs_file(path, default_source="auto")
    assert loaded[key].not_before_ts == 1200


def test_draft_loop_quarantines_one_pair_on_allowlist_403_and_continues(tmp_path: Path) -> None:
    profile = DraftLoopProfile(profile_id="profile-foton", brand="foton", channel="telegram")
    key_bad = DraftLoopKey(profile.profile_id, "chat-bad")
    key_ok = DraftLoopKey(profile.profile_id, "chat-ok")
    cfg = DraftLoopConfig(
        profiles={profile.profile_id: profile},
        pairs={
            key_bad: DraftLoopPair(key=key_bad, lead_id="49762441", expected_brand="foton"),
            key_ok: DraftLoopPair(key=key_ok, lead_id="49832125", expected_brand="foton"),
        },
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "journal.jsonl",
        manager_edit_log_path=tmp_path / "manager_edits.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        stop_path=tmp_path / "STOP_DRAFT_LOOP",
        debounce_seconds=60,
    )

    class AllowlistAmo(FakeAmo):
        def add_draft_note_to_test_lead(self, lead_id, **kwargs):
            if str(lead_id) == "49762441":
                raise RuntimeError("HTTP 403: lead_id 49762441 is not in allowlist")
            return super().add_draft_note_to_test_lead(lead_id, **kwargs)

    amo = AllowlistAmo()
    wappi = FakeWappi(
        {profile.profile_id: [{"id": "chat-bad", "type": "user"}, {"id": "chat-ok", "type": "user"}]},
        {
            (profile.profile_id, "chat-bad"): [_message("bad1", chat_id="chat-bad")],
            (profile.profile_id, "chat-ok"): [_message("ok1", chat_id="chat-ok")],
        },
    )
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=amo,
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=False)

    assert summary["auth_error"] is False
    assert summary["bot_calls"] == 2
    assert [note["lead_id"] for note in amo.notes] == ["49832125"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["quarantined_pairs"][key_bad.value]["reason"] == "allowlist_desync"
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "allowlist_desync" and row["status"] == "quarantined" for row in rows)


def test_draft_loop_processes_max_profile_with_explicit_pair(tmp_path: Path) -> None:
    profile = DraftLoopProfile(profile_id="profile-max-foton", brand="foton", channel="max")
    key = DraftLoopKey(profile.profile_id, "max-chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    cfg = DraftLoopConfig(
        profiles={profile.profile_id: profile},
        pairs={key: pair},
        allowed_test_lead_ids=frozenset({"49832125"}),
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "journal.jsonl",
        manager_edit_log_path=tmp_path / "manager_edits.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        stop_path=tmp_path / "STOP_DRAFT_LOOP",
        debounce_seconds=60,
    )
    amo = FakeAmo()
    bot = FakeBot()
    wappi = FakeWappi(
        {profile.profile_id: [{"id": "max-chat-1", "type": "DIALOG"}]},
        {(profile.profile_id, "max-chat-1"): [_message("mx1", chat_id="max-chat-1", text="Цена?", ts=1000)]},
    )
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=amo,
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand, **kwargs: {
            "key": key.value,
            "channel": kwargs.get("channel"),
        },
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=False)

    assert summary["processed"] == 1
    assert summary["bot_calls"] == 1
    assert bot.calls[0]["context"]["channel"] == "max"
    assert amo.notes[0]["lead_id"] == "49832125"


def test_draft_loop_journal_records_config_fingerprint_on_draft_created(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    fingerprint = {
        "schema_version": "draft_loop_config_fingerprint_v1_2026_06_10",
        "tree_hash": "abc12345",
        "kb_release_dir": "kb_release_20260610_v6_7_staging_r3",
        "gold_pack_version": "real_manager_gold_2026-06-08",
    }
    cfg = _config(tmp_path, pairs={key: pair}, config_fingerprint=fingerprint)
    wappi = FakeWappi({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {("profile-foton", "chat-1"): [_message("m1")]})
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    loop.run_once(dry_run=True)

    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    draft_created = next(row for row in rows if row["event"] == "draft_created")
    assert draft_created["config_fingerprint"] == fingerprint
    assert draft_created["config_fingerprint"]["schema_version"] == "draft_loop_config_fingerprint_v1_2026_06_10"


def test_draft_loop_persists_provenance_memory_by_profile_chat(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(MEMORY_PROVENANCE_ENV, "1")
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    calls: list[dict] = []

    class SemanticFrameBot(FakeBot):
        def build_draft(self, client_message: str, *, context=None):
            self.calls.append({"client_message": client_message, "context": context})
            return SubscriptionDraftResult(
                route="bot_answer_self",
                draft_text=f"Черновик: {client_message}",
                safety_flags=("client_safe_fact_verified",),
                metadata={
                    "direct_path_model_intent": {"primary_intent": "schedule", "sense": "schedule", "confidence": 0.9},
                    "semantic_frame": {
                        "source": "inline",
                        "requested_action": "answer_question",
                        "requested_product": {"grade": "7 класс", "subject": "физика", "format": "онлайн"},
                        "confidence": 0.88,
                    },
                },
            )

    bot = SemanticFrameBot()

    def context_builder(key, history, client_message, brand, *, dialogue_memory=None, current_message_id=""):
        from mango_mvp.channels.dialogue_memory import build_dialogue_memory

        memory = build_dialogue_memory(
            current_message=client_message,
            active_brand=brand,
            recent_messages=history,
            previous_memory=dialogue_memory or {},
            context={"current_message_id": current_message_id},
            session_id=f"test:{key.value}",
        )
        payload = {"dialogue_memory_view": memory.to_prompt_view(), "dialogue_memory_state": memory.to_json_dict()}
        calls.append({"memory": dialogue_memory or {}, "message_id": current_message_id, "payload": payload})
        return payload

    cfg = _config(tmp_path, pairs={key: pair})
    wappi = FakeWappi(
        {"profile-foton": [{"id": "chat-1", "type": "user"}]},
        {("profile-foton", "chat-1"): [_message("m1", text="Сын в 7 классе, физика онлайн")]},
    )
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=context_builder,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    loop.run_once(dry_run=False)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    stored = state["dialogue_memory"][key.value]
    assert stored["known_slots"]["grade"]["value"] == "7"
    assert stored["known_slots"]["grade"]["message_id"] == "m1"
    assert stored["last_semantic_reading"]["source"] == "inline"
    assert stored["last_semantic_reading"]["product_subject"] == "физика"
    assert "last_semantic_reading" not in calls[0]["payload"]["dialogue_memory_view"]
    assert calls[0]["message_id"] == "m1"


def test_draft_loop_journal_reads_old_rows_without_config_fingerprint(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "event": "note_written",
                "status": "note_written",
                "profile_id": "profile-foton",
                "chat_id": "chat-1",
                "message_id": "old-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    journal = DraftLoopJournal(journal_path)

    assert journal.rows()[0]["message_id"] == "old-1"
    assert journal.processed_message_keys() == {("profile-foton", "chat-1", "old-1")}


def test_build_draft_loop_config_fingerprint_uses_snapshot_dir_and_gold_version(tmp_path: Path) -> None:
    snapshot = tmp_path / "kb_release_test" / "kb_release_v3_snapshot.json"
    snapshot.parent.mkdir()
    snapshot.write_text("{}", encoding="utf-8")

    fingerprint = build_draft_loop_config_fingerprint(snapshot, gold_pack_version="gold-v1", repo_root=tmp_path)

    assert fingerprint["schema_version"] == "draft_loop_config_fingerprint_v1_2026_06_10"
    assert fingerprint["kb_release_dir"] == "kb_release_test"
    assert fingerprint["gold_pack_version"] == "gold-v1"
    assert fingerprint["tree_hash"] == "unknown"


def test_draft_loop_never_writes_note_for_auto_candidate_without_explicit_pair(tmp_path: Path) -> None:
    amo = FakeAmo()
    loop = _loop(
        tmp_path,
        messages=[_message("m1")],
        auto_resolver=lambda key, message: {"lead_id": "49832125", "source": "auto"},
        amo=amo,
    )

    summary = loop.run_once(dry_run=False)

    assert summary["bot_calls"] == 0
    assert amo.notes == []
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "pair_missing"
    assert rows[0]["auto_candidate"]["lead_id"] == "49832125"


def test_manager_note_marks_unconfirmed_auto_pair_memory_unavailable(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(
        key=key,
        lead_id="49832125",
        expected_brand="foton",
        source="auto",
        auto_note="Привязка автоматическая.",
    )
    amo = FakeAmo()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, amo=amo)

    loop.run_once(dry_run=False)

    note = amo.notes[0]
    assert "Память клиента: недоступна — автоматическая привязка не подтверждена." in note["outgoing_visibility_note"]
    assert "49832125" not in note["outgoing_visibility_note"]


def test_auto_pair_manager_note_is_actionable_without_internal_roles() -> None:
    text = _auto_pair_note(
        profile=DraftLoopProfile("profile-foton", "foton", "telegram"),
        candidate={"match_key": "Telegram ID"},
    )

    assert "Фотон, Telegram" in text
    assert "Проверьте карточку" in text
    assert "архитектор" not in text.casefold()


def test_manager_note_marks_manual_pair_memory_connected(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    amo = FakeAmo()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, amo=amo)
    loop.context_builder = lambda *_args, **_kwargs: {"read_only_customer_context": {"found": True}}

    loop.run_once(dry_run=False)

    assert "Память клиента: подключена." in amo.notes[0]["outgoing_visibility_note"]


def test_unconfirmed_auto_pair_core_strips_customer_memory_from_custom_builder(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton", source="auto")
    bot = FakeBot()
    received_dialogue_memory: list[Mapping[str, object]] = []

    def malicious_builder(*_args, dialogue_memory=None, **_kwargs):
        received_dialogue_memory.append(dict(dialogue_memory or {}))
        return {
            "read_only_customer_context": {"summary": "Чужая память"},
            "timeline_context": {"bot_context": ["Чужой факт"]},
            "dialogue_memory_view": {
                "conversation_summary_short": "Текущий диалог",
                "crm_known_slots": {"grade": "7"},
            },
        }

    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, bot=bot)
    loop.context_builder = malicious_builder
    loop.state.set_dialogue_memory(key, {"crm_known_slots": {"phone": "+70000000000"}})

    loop.run_once(dry_run=True)

    context = bot.calls[0]["context"]
    assert received_dialogue_memory == [{}]
    assert "read_only_customer_context" not in context
    assert "timeline_context" not in context
    assert "crm_known_slots" not in context["dialogue_memory_view"]
    assert context["dialogue_memory_view"]["conversation_summary_short"] == "Текущий диалог"


def test_draft_loop_stop_fetches_but_does_not_call_bot_or_mark_processed(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, stop=True, amo=amo, bot=bot)

    summary = loop.run_once(dry_run=False)

    assert summary["stop_active"] is True
    assert summary["bot_calls"] == 0
    assert bot.calls == []
    assert amo.notes == []
    assert not (tmp_path / "state.json").exists()
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "stop_raw_inbound"
    assert rows[0]["status"] == "stop_not_processed"


def test_draft_loop_chat_limit_zero_pages_all_dialogs(tmp_path: Path) -> None:
    profile = DraftLoopProfile(profile_id="profile-foton", brand="foton", channel="telegram")
    cfg = DraftLoopConfig(
        profiles={profile.profile_id: profile},
        pairs={},
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "journal.jsonl",
        manager_edit_log_path=tmp_path / "manager_edits.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        stop_path=tmp_path / "STOP_DRAFT_LOOP",
        chat_limit=0,
    )
    dialogs = [{"id": f"chat-{idx}", "type": "user"} for idx in range(101)]
    wappi = FakeWappi({profile.profile_id: dialogs}, {})
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=True)

    assert summary["bot_calls"] == 0
    assert wappi.list_calls == 2


def test_draft_loop_filters_non_text_and_recent_debounce(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    bot = FakeBot()
    loop = _loop(
        tmp_path,
        messages=[
            _message("voice", typ="voice", text=""),
            _message("recent", text="Подождите", ts=1190),
        ],
        pairs={key: pair},
        bot=bot,
    )

    summary = loop.run_once(dry_run=False)

    assert summary["deferred"] == 1
    assert summary["bot_calls"] == 0
    assert bot.calls == []


def test_draft_loop_defers_wappi_fetch_messages_queued_400_and_retries_next_cycle(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")

    class DeferredOnceWappi(FakeWappi):
        def __init__(self) -> None:
            super().__init__({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {("profile-foton", "chat-1"): [_message("m1")]})
            self.fetch_calls = 0

        def get_chat_messages(self, *, channel: str, profile_id: str, chat_id: str, **kwargs):
            self.fetch_calls += 1
            if self.fetch_calls == 1:
                raise AmoWappiHttpError(
                    'HTTP 400: {"status":"error","detail":"Команда fetchMessages сохранена для повторной отправки. TaskID: abc"}'
                )
            return super().get_chat_messages(channel=channel, profile_id=profile_id, chat_id=chat_id, **kwargs)

    wappi = DeferredOnceWappi()
    bot = FakeBot()
    cfg = _config(tmp_path, pairs={key: pair})
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    first = loop.run_once(dry_run=False)
    first_heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    second = loop.run_once(dry_run=False)

    assert first["auth_error"] is False
    assert first["auth_error_count"] == 0
    assert first["deferred_fetch"] == 1
    assert first["bot_calls"] == 0
    assert first_heartbeat["status"] == "ok"
    assert first_heartbeat["summary"]["deferred_fetch"] == 1
    assert second["processed"] == 1
    assert second["bot_calls"] == 1
    assert bot.calls[0]["client_message"] == "Цена?"
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "ok"
    assert heartbeat["summary"]["deferred_fetch"] == 0
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "deferred_fetch" and row["chat_id"] == "chat-1" for row in rows)


def test_draft_loop_regular_wappi_400_still_raises(tmp_path: Path) -> None:
    class BadRequestWappi(FakeWappi):
        def get_chat_messages(self, *, channel: str, profile_id: str, chat_id: str, **kwargs):
            raise AmoWappiHttpError('HTTP 400: {"status":"error","detail":"bad request"}')

    cfg = _config(tmp_path)
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=BadRequestWappi({"profile-foton": [{"id": "chat-1", "type": "user"}]}, {}),
        amo_client=FakeAmo(),
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    with pytest.raises(AmoWappiHttpError, match="bad request"):
        loop.run_once(dry_run=False)


def test_draft_loop_state_loss_does_not_duplicate_written_note_from_journal(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "event": "note_written",
                "status": "note_written",
                "profile_id": "profile-foton",
                "chat_id": "chat-1",
                "message_id": "m1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, amo=amo, bot=bot)

    summary = loop.run_once(dry_run=False)

    assert summary["bot_calls"] == 0
    assert amo.notes == []
    assert bot.calls == []


def test_draft_loop_dry_run_replays_without_persisting_processed_state(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, amo=amo, bot=bot)

    first = loop.run_once(dry_run=True)
    second = loop.run_once(dry_run=True)

    assert first["processed"] == 1
    assert first["bot_calls"] == 1
    assert second["processed"] == 1
    assert second["bot_calls"] == 1
    assert amo.notes == []
    assert len(bot.calls) == 2
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows if row["event"] == "draft_created"] == ["draft_created", "draft_created"]
    assert not (tmp_path / "state.json").exists()


def test_draft_loop_dry_run_pair_missing_does_not_persist_state(tmp_path: Path) -> None:
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={}, amo=amo, bot=bot)

    first = loop.run_once(dry_run=True)
    second = loop.run_once(dry_run=True)

    assert first["processed"] == 1
    assert first["skipped"] == 1
    assert first["bot_calls"] == 0
    assert second["processed"] == 1
    assert second["skipped"] == 1
    assert second["bot_calls"] == 0
    assert amo.notes == []
    assert bot.calls == []
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows if row["event"] == "pair_missing"] == ["pair_missing", "pair_missing"]
    assert not (tmp_path / "state.json").exists()


def test_draft_loop_live_defers_pair_missing_without_losing_message(tmp_path: Path) -> None:
    message = _message("m1")
    loop = _loop(tmp_path, messages=[message], pairs={})

    first = loop.run_once(dry_run=False)
    second = _loop(tmp_path, messages=[message], pairs={}).run_once(dry_run=False)

    assert first["processed"] == second["processed"] == 0
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows if row["event"] == "pair_missing"] == ["pair_missing"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["processed"] == []
    assert list(state["deferred_pair_missing"]) == ["profile-foton\tchat-1\tm1"]
    assert state["deferred_pair_missing"]["profile-foton\tchat-1\tm1"]["text"] == "Цена?"

    key = DraftLoopKey("profile-foton", "chat-1")
    amo = FakeAmo()
    bot = FakeBot()
    configured = _loop(
        tmp_path,
        messages=[],
        pairs={
            key: DraftLoopPair(
                key=key,
                lead_id="49832125",
                expected_brand="foton",
                not_before_ts=int(message["time"]) + 60,
            )
        },
        amo=amo,
        bot=bot,
    )
    configured.wappi_client = FakeWappi({"profile-foton": []}, {})

    recovered = configured.run_once(dry_run=False)

    assert recovered["processed"] == recovered["bot_calls"] == 1
    assert len(bot.calls) == len(amo.notes) == 1
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert {item["message_id"] for item in state["processed"]} == {"m1"}
    assert state["deferred_pair_missing"] == {}


def test_deferred_pair_missing_buffer_is_bounded(tmp_path: Path) -> None:
    state = DraftLoopState(tmp_path / "state.json", persist=False)
    for index in range(MAX_DEFERRED_PAIR_MISSING + 1):
        state.defer_pair_missing(
            WappiHistoryMessage(
                profile_id="profile-foton",
                chat_id="chat-1",
                message_id=f"m-{index}",
                text="Цена?",
                message_type="text",
                timestamp=index,
                from_me=False,
            )
        )

    deferred = state.payload["deferred_pair_missing"]
    assert len(deferred) == MAX_DEFERRED_PAIR_MISSING
    assert "profile-foton\tchat-1\tm-0" not in deferred
    assert f"profile-foton\tchat-1\tm-{MAX_DEFERRED_PAIR_MISSING}" in deferred


def test_draft_loop_dry_run_brand_mismatch_does_not_persist_state(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="unpk")
    amo = FakeAmo()
    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: pair}, amo=amo, bot=bot)

    first = loop.run_once(dry_run=True)
    second = loop.run_once(dry_run=True)

    assert first["processed"] == 1
    assert first["skipped"] == 1
    assert first["bot_calls"] == 0
    assert second["processed"] == 1
    assert second["skipped"] == 1
    assert second["bot_calls"] == 0
    assert amo.notes == []
    assert bot.calls == []
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows if row["event"] == "brand_pair_mismatch"] == [
        "brand_pair_mismatch",
        "brand_pair_mismatch",
    ]
    assert not (tmp_path / "state.json").exists()


def test_draft_loop_retries_pending_note_once(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    cfg = _config(tmp_path, pairs={key: DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")})
    state = DraftLoopState(cfg.state_path)
    state.payload["pending_notes"] = {
        "profile-foton\tchat-1\tm1": {
            "profile_id": "profile-foton",
            "chat_id": "chat-1",
            "message_id": "m1",
            "lead_id": "49832125",
            "brand": "foton",
            "route": "bot_answer_self",
            "safety_flags": [],
            "bot_draft_text": "Готовый черновик",
            "status": "note_pending",
        }
    }
    state.save()
    amo = FakeAmo()
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=FakeWappi({"profile-foton": []}, {}),
        amo_client=amo,
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=False)

    assert summary["retried_pending"] == 1
    assert len(amo.notes) == 1
    assert json.loads(cfg.state_path.read_text(encoding="utf-8"))["pending_notes"] == {}


def test_load_pairs_rejects_bare_chat_id(tmp_path: Path) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([{"chat_id": "chat-1", "lead_id": "49832125", "expected_brand": "foton"}]), encoding="utf-8")

    with pytest.raises(DraftLoopConfigError):
        load_pairs_file(path)


def test_load_profiles_accepts_telegram_and_max_profiles(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            [
                {"profile_id": "tg-foton", "brand": "foton", "channel": "telegram"},
                {"profile_id": "max-unpk", "brand": "unpk", "channel": "max"},
                {"profile_id": "bad", "brand": "foton", "channel": "whatsapp"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    profiles = load_profiles_file(path)

    assert set(profiles) == {"tg-foton", "max-unpk"}
    assert profiles["max-unpk"].channel == "max"


def test_manager_edit_classifies_superseded_draft_sent_later_and_single_best_match() -> None:
    drafts = [
        DraftWindow(profile_id="p", chat_id="c", message_id="d1", bot_draft_text="Добрый день! Цена 49 000.", draft_ts=100, superseded=True),
        DraftWindow(profile_id="p", chat_id="c", message_id="d2", bot_draft_text="Добрый день! Стоимость 49 000.", draft_ts=200),
    ]
    outgoing = [OutgoingWindowMessage(message_id="o1", text="Добрый день! Цена 49 000.", sent_ts=300)]

    rows = classify_manager_edit_windows(drafts, outgoing, now_ts=500)

    matched = {row["message_id"]: row for row in rows}
    assert matched["d1"]["match_class"] == "unedited"
    assert matched["d1"]["matched_message_id"] == "o1"
    assert "d2" not in matched


def test_manager_edit_window_keeps_evening_draft_until_next_business_day() -> None:
    draft_ts = int(datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc).timestamp())
    drafts = [DraftWindow(profile_id="p", chat_id="c", message_id="d1", bot_draft_text="Адрес: Красносельская, 30.", draft_ts=draft_ts)]
    outgoing = [OutgoingWindowMessage(message_id="o1", text="Адрес: Красносельская, 30.", sent_ts=draft_ts + 10 * 60 * 60)]

    rows = classify_manager_edit_windows(drafts, outgoing, now_ts=draft_ts + 10 * 60 * 60)

    assert rows[0]["match_class"] == "unedited"
    assert rows[0]["matched_message_id"] == "o1"


def test_draft_loop_run_once_writes_manager_edit_match_from_outgoing_history(tmp_path: Path) -> None:
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(key=key, lead_id="49832125", expected_brand="foton")
    draft_ts = int(datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc).timestamp())
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "event": "note_written",
                "status": "note_written",
                "profile_id": "profile-foton",
                "chat_id": "chat-1",
                "message_id": "m1",
                "lead_id": "49832125",
                "brand": "foton",
                "route": "draft_for_manager",
                "safety_flags": ["draft_only"],
                "bot_draft_text": "Фотон находится на Верхней Красносельской, 30.",
                "created_at": datetime.fromtimestamp(draft_ts, tz=timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = _config(tmp_path, pairs={key: pair})
    wappi = FakeWappi(
        {"profile-foton": [{"id": "chat-1", "type": "user"}]},
        {
            ("profile-foton", "chat-1"): [
                _message("m1", text="Какой адрес?", ts=draft_ts),
                _message("18242", text="Фотон находится на Верхней Красносельской, 30.", ts=draft_ts + 12 * 60 * 60, from_me=True),
            ]
        },
    )
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=FakeBot(),
        context_builder=lambda key, history, client_message, brand: {},
        journal=DraftLoopJournal(journal_path),
        now_fn=lambda: datetime.fromtimestamp(draft_ts + 13 * 60 * 60, tz=timezone.utc),
    )

    summary = loop.run_once(dry_run=True)

    assert summary["manager_edits_classified"] == 1
    rows = [json.loads(line) for line in (tmp_path / "manager_edits.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["message_id"] == "m1"
    assert rows[0]["matched_message_id"] == "18242"
    assert rows[0]["match_class"] == "unedited"
    assert rows[0]["lead_id"] == "49832125"

    summary_again = loop.run_once(dry_run=True)
    assert summary_again["manager_edits_classified"] == 0
    assert len((tmp_path / "manager_edits.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_draft_loop_writes_heartbeat_on_success(tmp_path: Path) -> None:
    loop = _loop(tmp_path, messages=[], pairs={})
    loop.code_identity = {"code_root": "/repo/live", "git_sha": "a" * 40}

    summary = loop.run_once(dry_run=True)

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "ok"
    assert heartbeat["summary"]["processed"] == summary["processed"]
    assert heartbeat["last_cycle_at"]
    assert heartbeat["code_root"] == "/repo/live"
    assert heartbeat["git_sha"] == "a" * 40


def test_build_draft_loop_code_identity_reads_git_root_and_full_head() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    identity = build_draft_loop_code_identity(repo_root)

    assert identity["code_root"] == str(repo_root)
    expected_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert identity["git_sha"] == expected_sha


def test_draft_loop_auth_error_series_stops_without_calling_bot(tmp_path: Path) -> None:
    class AuthFailWappi(FakeWappi):
        def list_chats(self, *, channel: str, profile_id: str, limit: int = 50):
            self.list_calls += 1
            raise RuntimeError("HTTP 401 Unauthorized")

    cfg = _config(tmp_path, pairs={})
    cfg = DraftLoopConfig(
        profiles=cfg.profiles,
        pairs=cfg.pairs,
        allowed_test_lead_ids=cfg.allowed_test_lead_ids,
        state_path=cfg.state_path,
        journal_path=cfg.journal_path,
        manager_edit_log_path=cfg.manager_edit_log_path,
        heartbeat_path=cfg.heartbeat_path,
        stop_path=cfg.stop_path,
        debounce_seconds=cfg.debounce_seconds,
        history_limit=cfg.history_limit,
        auth_error_limit=2,
    )
    wappi = AuthFailWappi({"profile-foton": []}, {})
    bot = FakeBot()
    loop = AmoWappiDraftLoop(
        config=cfg,
        wappi_client=wappi,
        amo_client=FakeAmo(),
        bot_provider=bot,
        context_builder=lambda key, history, client_message, brand: {},
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
    )

    first = loop.run_once(dry_run=True)
    second = loop.run_once(dry_run=True)
    third = loop.run_once(dry_run=True)

    assert first["auth_error"] is True
    assert first["stopped"] is False
    assert second["auth_error"] is True
    assert second["stopped"] is True
    assert third["stopped"] is True
    assert wappi.list_calls == 2
    assert bot.calls == []
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "auth_error"
    assert heartbeat["auth_error_count"] == 2


def test_draft_loop_modules_do_not_import_public_telegram_transport() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/mango_mvp/integrations/draft_loop.py", "src/mango_mvp/pilot_context_assembly.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "run_telegram_public_pilot_bots" not in source
        assert "reply_text" not in source
        assert "send_chat_action" not in source
        assert "telegram.ext" not in source
