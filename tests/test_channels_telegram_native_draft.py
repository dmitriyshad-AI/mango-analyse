from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mango_mvp.channels import (
    ChannelMemoryStore,
    FakeTelegramNativeDraftClient,
    NATIVE_DRAFT_OPERATION_CLEAR,
    NATIVE_DRAFT_OPERATION_SAVE,
    NATIVE_DRAFT_STATUS_BLOCKED,
    NATIVE_DRAFT_STATUS_CLEARED,
    NATIVE_DRAFT_STATUS_CONFLICT,
    NATIVE_DRAFT_STATUS_EMPTY,
    NATIVE_DRAFT_STATUS_SAVED,
    NATIVE_DRAFT_STATUS_UNCHANGED,
    NativeDraftOrchestrator,
    TDLibTelegramNativeDraftClient,
    TELEGRAM_BUSINESS_CHANNEL,
    TelegramBusinessRuntime,
    TelegramNativeDraftConfig,
    TelegramNativeDraftIntent,
    TelegramNativeDraftMemoryStore,
    build_and_store_channel_draft_preview,
    build_native_draft_intent_from_channel_draft,
    build_native_draft_manager_summary,
    draft_text_hash,
    guard_tdlib_database_dir,
    resolve_native_draft_chat_ref,
    scrub_native_draft_payload,
    telegram_native_draft_safety_contract,
)


START = datetime(2026, 5, 13, 10, 0, tzinfo=timezone.utc)
START_TS = int(START.timestamp())


class StepClock:
    def __init__(self) -> None:
        self.value = START

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def business_update(text: str = "Сколько стоит курс?") -> dict:
    return {
        "update_id": 1101,
        "business_message": {
            "business_connection_id": "bc-123",
            "message_id": 44,
            "date": START_TS,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 555},
            "text": text,
        },
    }


def enabled_config(*, kill_switch: bool = False, allowed_chat_ids: tuple[str, ...] = ()) -> TelegramNativeDraftConfig:
    return TelegramNativeDraftConfig(enabled=True, kill_switch=kill_switch, allowed_chat_ids=allowed_chat_ids)


def test_native_draft_config_from_env_uses_explicit_env_and_redacts_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED", "true")
    monkeypatch.setenv("CHANNEL_TELEGRAM_NATIVE_DRAFT_KILL_SWITCH", "false")
    monkeypatch.setenv("TDLIB_API_HASH", "real-env-secret")

    explicit_empty = TelegramNativeDraftConfig.from_env({})
    explicit_values = TelegramNativeDraftConfig.from_env(
        {
            "CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED": "true",
            "CHANNEL_TELEGRAM_NATIVE_DRAFT_KILL_SWITCH": "false",
            "CHANNEL_TELEGRAM_NATIVE_DRAFT_ALLOWED_CHAT_IDS": "555, 777",
            "TDLIB_API_ID": "123",
            "TDLIB_API_HASH": "secret-hash",
            "TDLIB_DATABASE_ENCRYPTION_KEY": "secret-key",
            "TDLIB_PHONE_NUMBER": "+79990000000",
            "TDLIB_DATABASE_DIR": "/secure/tdlib",
        }
    )

    assert explicit_empty.enabled is False
    assert explicit_empty.kill_switch is True
    assert explicit_empty.api_hash_present is False
    assert explicit_values.enabled is True
    assert explicit_values.kill_switch is False
    assert explicit_values.allowed_chat_ids == ("555", "777")
    exported = explicit_values.to_json_dict()
    assert exported["api_hash_present"] is True
    assert exported["database_encryption_key_present"] is True
    assert "secret-hash" not in str(exported)
    assert "+79990000000" not in str(exported)


def test_native_draft_intent_validates_text_operation_and_stable_key() -> None:
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Здравствуйте! Уточним детали и вернемся.",
        draft_id="draft-1",
        channel_thread_id="bc-123:555",
        source_message_idempotency_key="msg-key",
        created_at=START,
    )
    repeat = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Здравствуйте! Уточним детали и вернемся.",
        draft_id="draft-1",
        channel_thread_id="bc-123:555",
        source_message_idempotency_key="msg-key",
        created_at=START + timedelta(seconds=5),
    )

    assert intent.idempotency_key == repeat.idempotency_key
    assert intent.text_hash == draft_text_hash("Здравствуйте! Уточним детали и вернемся.")
    assert intent.to_json_dict()["text"] is None
    assert "Здравствуйте" not in str(intent.to_json_dict())
    with pytest.raises(ValueError, match="non-empty text"):
        TelegramNativeDraftIntent(operation=NATIVE_DRAFT_OPERATION_SAVE, chat_id="555", text="", draft_id="draft-1")
    with pytest.raises(ValueError, match="unsupported native draft operation"):
        TelegramNativeDraftIntent(operation="send_message", chat_id="555")
    with pytest.raises(ValueError, match="must not include text"):
        TelegramNativeDraftIntent(operation=NATIVE_DRAFT_OPERATION_CLEAR, chat_id="555", text="no")


def test_native_draft_public_api_has_no_send_surface() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    public = {name for name in dir(client) if not name.startswith("_")}

    assert {"save_draft", "get_draft_state", "clear_draft"}.issubset(public)
    forbidden = {"send", "send_message", "sendMessage", "forward", "raw_call", "execute", "batch_send", "outreach"}
    assert public.isdisjoint(forbidden)
    assert not hasattr(client, "send_message")
    assert not hasattr(client, "sendMessage")


def test_default_config_blocks_save_before_feature_flag() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(client, config=TelegramNativeDraftConfig())
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик",
        draft_id="draft-1",
    )

    result = orchestrator.save_intent(intent)

    assert result.status == NATIVE_DRAFT_STATUS_BLOCKED
    assert result.blocked_reason == "native_drafts_disabled"
    assert result.metadata["telegram_api_called"] is False
    assert client.operations == []


def test_kill_switch_blocks_save_and_clear_but_not_read_state() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(client, config=enabled_config(kill_switch=True))
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик",
        draft_id="draft-1",
    )

    saved = orchestrator.save_intent(intent)
    cleared = orchestrator.clear_draft("555")
    state = orchestrator.get_draft_state("555")

    assert saved.status == NATIVE_DRAFT_STATUS_BLOCKED
    assert saved.blocked_reason == "native_draft_kill_switch"
    assert cleared.status == NATIVE_DRAFT_STATUS_BLOCKED
    assert state.owner == "empty"
    assert client.operations == []


def test_allowlist_blocks_non_allowed_chat() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(client, config=enabled_config(allowed_chat_ids=("777",)))
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик",
        draft_id="draft-1",
    )

    result = orchestrator.save_intent(intent)

    assert result.status == NATIVE_DRAFT_STATUS_BLOCKED
    assert result.blocked_reason == "chat_not_allowlisted"
    assert client.operations == []


def test_fake_client_save_get_clear_lifecycle_and_idempotency() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик для менеджера",
        draft_id="draft-1",
        created_at=START,
    )

    saved = client.save_draft(intent)
    unchanged = client.save_draft(intent)
    state = client.get_draft_state("555")
    cleared = client.clear_draft("555", reason="test_clear")
    empty = client.clear_draft("555", reason="test_clear")

    assert saved.status == NATIVE_DRAFT_STATUS_SAVED
    assert saved.changed is True
    assert unchanged.status == NATIVE_DRAFT_STATUS_UNCHANGED
    assert unchanged.changed is False
    assert state.owner == "mango"
    assert state.text == "Черновик для менеджера"
    assert state.last_written_hash == intent.text_hash
    assert cleared.status == NATIVE_DRAFT_STATUS_CLEARED
    assert cleared.changed is True
    assert empty.status == NATIVE_DRAFT_STATUS_EMPTY
    assert client.get_draft_state("555").owner == "empty"


def test_conflict_policy_never_overwrites_or_clears_manager_draft() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    client.inject_manager_draft("555", "Я сам уже пишу клиенту")
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик Mango",
        draft_id="draft-1",
    )

    conflict = client.save_draft(intent)
    clear_conflict = client.clear_draft("555")

    assert conflict.status == NATIVE_DRAFT_STATUS_CONFLICT
    assert "manager_draft_present" in conflict.conflict_flags
    assert client.get_draft_state("555").text == "Я сам уже пишу клиенту"
    assert clear_conflict.status == NATIVE_DRAFT_STATUS_CONFLICT
    assert client.get_draft_state("555").owner == "manager"


def test_native_draft_memory_store_redacts_text_by_default_and_dedupes() -> None:
    store = TelegramNativeDraftMemoryStore()
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Секретный текст черновика",
        draft_id="draft-1",
    )
    result = client.save_draft(intent)

    assert store.record_intent(intent) is True
    assert store.record_intent(intent) is False
    assert store.record_result(result) is True
    assert store.record_result(result) is False
    snapshot = store.snapshot()

    assert snapshot["summary"] == {"intents": 1, "results": 1}
    assert "Секретный текст" not in str(snapshot)
    assert snapshot["intents"][0]["text_hash"] == intent.text_hash
    assert snapshot["results"][0]["state"]["text_redacted"] is True


def test_guard_tdlib_database_dir_rejects_repo_stable_runtime_and_public_tmp(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    safe_parent = tmp_path / "secure"
    safe_parent.mkdir()
    safe_dir = safe_parent / "tdlib-main-account"

    assert guard_tdlib_database_dir(safe_dir, repo_root=repo_root) == safe_dir.resolve(strict=False)
    with pytest.raises(ValueError, match="absolute"):
        guard_tdlib_database_dir("relative/tdlib", repo_root=repo_root)
    with pytest.raises(ValueError, match="repository"):
        guard_tdlib_database_dir(repo_root / "tdlib", repo_root=repo_root)
    with pytest.raises(ValueError, match="stable_runtime"):
        guard_tdlib_database_dir(safe_parent / "stable_runtime" / "tdlib", repo_root=repo_root)
    with pytest.raises(ValueError, match="public temp"):
        guard_tdlib_database_dir("/tmp/tdlib-mango", repo_root=repo_root)


def test_secret_redaction_hides_tdlib_credentials_paths_and_raw_payload() -> None:
    payload = {
        "TDLIB_API_HASH": "abc",
        "TDLIB_DATABASE_ENCRYPTION_KEY": "key",
        "TDLIB_PHONE_NUMBER": "+79990000000",
        "TDLIB_DATABASE_DIR": "/Users/name/secret/tdlib",
        "raw_payload": {"telegram_update": "secret"},
        "safe": {"value": "ok"},
    }

    scrubbed = scrub_native_draft_payload(payload)

    assert scrubbed["TDLIB_API_HASH"] == "[REDACTED]"
    assert scrubbed["TDLIB_DATABASE_ENCRYPTION_KEY"] == "[REDACTED]"
    assert scrubbed["TDLIB_PHONE_NUMBER"] == "[REDACTED]"
    assert scrubbed["TDLIB_DATABASE_DIR"] == "[REDACTED]"
    assert scrubbed["raw_payload"] == "[REDACTED]"
    assert scrubbed["safe"]["value"] == "ok"


def test_tdlib_stub_never_logs_in_or_calls_network() -> None:
    client = TDLibTelegramNativeDraftClient(TelegramNativeDraftConfig(enabled=True, kill_switch=False))
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик",
        draft_id="draft-1",
    )

    result = client.save_draft(intent)

    assert result.status == NATIVE_DRAFT_STATUS_BLOCKED
    assert result.blocked_reason == "tdlib_transport_not_configured"
    assert result.metadata["network_calls"] is False
    assert result.metadata["telegram_api_called"] is False


def test_resolve_chat_ref_marks_unverified_fallbacks() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    message = runtime.process_update(business_update()).messages[0]

    mapped, mapped_meta = resolve_native_draft_chat_ref(message, chat_ref_map={message.channel_thread_id: "tdlib-555"})
    fallback, fallback_meta = resolve_native_draft_chat_ref(message)

    assert mapped == "tdlib-555"
    assert mapped_meta["verified"] is True
    assert fallback == "555"
    assert fallback_meta["verified"] is False


def test_business_message_to_native_draft_offline_dry_run() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    channel_store = ChannelMemoryStore(clock=StepClock())
    native_store = TelegramNativeDraftMemoryStore()
    native_client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(
        native_client,
        config=enabled_config(),
        store=native_store,
    )
    message = runtime.process_update(business_update()).messages[0]

    assert message.channel == TELEGRAM_BUSINESS_CHANNEL
    preview, store_result = build_and_store_channel_draft_preview(
        channel_store,
        message,
        actor="telegram_business_runtime",
        context={"identity_status": "unmatched", "requires_manager_review": True},
    )
    intent = build_native_draft_intent_from_channel_draft(preview, chat_ref_map={message.channel_thread_id: "555"})
    result = orchestrator.save_intent(intent)

    assert store_result.created is True
    assert result.status == NATIVE_DRAFT_STATUS_SAVED
    assert result.state is not None
    assert result.state.chat_id == "555"
    assert result.state.owner == "mango"
    assert result.audit_event["event_type"] == "native_draft_saved"
    assert result.audit_event["live_send"] is False
    assert native_store.snapshot()["summary"] == {"intents": 1, "results": 1}
    assert channel_store.summary()["draft_status_counts"] == {"needs_review": 1}


def test_manager_summary_shows_native_draft_status_without_send_button() -> None:
    runtime = TelegramBusinessRuntime(clock=StepClock())
    channel_store = ChannelMemoryStore(clock=StepClock())
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(client, config=enabled_config())
    message = runtime.process_update(business_update()).messages[0]
    preview, _ = build_and_store_channel_draft_preview(channel_store, message, actor="test")

    result = orchestrator.save_from_draft(preview, chat_ref_map={message.channel_thread_id: "555"})
    summary = build_native_draft_manager_summary(
        preview,
        result,
        identity_status="unmatched",
        source_refs=("telegram_business:update:1101",),
    )

    assert summary["identity_status"] == "unmatched"
    assert summary["exact_draft_text"] == preview.reply.text
    assert summary["native_draft_status"] == NATIVE_DRAFT_STATUS_SAVED
    assert summary["actions"]["client_send_button_available"] is False
    assert summary["actions"]["native_draft_send_available"] is False
    assert summary["safety"]["live_send"] is False
    assert summary["native_draft_state"]["text"] is None
    assert summary["native_draft_state"]["text_redacted"] is True


def test_new_inbound_marks_existing_native_draft_stale_and_reconciliation_is_read_only() -> None:
    client = FakeTelegramNativeDraftClient(clock=StepClock())
    orchestrator = NativeDraftOrchestrator(client, config=enabled_config())
    runtime = TelegramBusinessRuntime(clock=StepClock())
    message = runtime.process_update(business_update()).messages[0]
    intent = TelegramNativeDraftIntent(
        operation=NATIVE_DRAFT_OPERATION_SAVE,
        chat_id="555",
        text="Черновик Mango",
        draft_id="draft-1",
    )
    saved = orchestrator.save_intent(intent)

    stale = orchestrator.mark_stale_after_inbound("555", message)
    same_text = orchestrator.reconcile_manual_send("555", sent_text="Черновик Mango", last_written_hash=saved.state.last_written_hash)
    changed_text = orchestrator.reconcile_manual_send("555", sent_text="Другой текст", last_written_hash=saved.state.last_written_hash)
    unknown = orchestrator.reconcile_manual_send("555", sent_text=None, last_written_hash=saved.state.last_written_hash)

    assert stale.stale is True
    assert stale.stale_reason.startswith("new_inbound:")
    assert same_text["status"] == "manual_send_observed"
    assert changed_text["status"] == "manager_sent_modified_text"
    assert unknown["status"] == "unknown_send_state"
    assert same_text["telegram_send_called"] is False


def test_native_draft_safety_contract_blocks_live_effects() -> None:
    safety = telegram_native_draft_safety_contract()

    assert safety["native_draft_allowed"] is True
    assert safety["live_send"] is False
    assert safety["send_endpoint"] is False
    assert safety["raw_tdlib_rpc_endpoint"] is False
    assert safety["batch_operations"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
