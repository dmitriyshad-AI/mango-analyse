from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.dialogue_memory import MEMORY_PROVENANCE_ENV
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
    OutgoingWindowMessage,
    build_draft_loop_config_fingerprint,
    classify_manager_edit_windows,
    load_pairs_file,
    load_profiles_file,
)


class FakeWappi:
    def __init__(self, dialogs, messages_by_chat) -> None:
        self.dialogs = dialogs
        self.messages_by_chat = messages_by_chat
        self.list_calls = 0

    def list_telegram_chats(self, *, profile_id: str, limit: int = 50):
        self.list_calls += 1
        return {"dialogs": self.dialogs.get(profile_id, [])}

    def get_telegram_chat_messages(self, *, profile_id: str, chat_id: str, **kwargs):
        return {"messages": self.messages_by_chat.get((profile_id, chat_id), [])}

    def list_chats(self, *, channel: str, profile_id: str, limit: int = 50):
        self.list_calls += 1
        return {"dialogs": self.dialogs.get(profile_id, [])}

    def get_chat_messages(self, *, channel: str, profile_id: str, chat_id: str, **kwargs):
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
    bot = FakeBot()
    calls: list[dict] = []

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

    summary = loop.run_once(dry_run=True)

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "ok"
    assert heartbeat["summary"]["processed"] == summary["processed"]
    assert heartbeat["last_cycle_at"]


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
