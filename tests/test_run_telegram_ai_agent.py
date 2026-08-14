from __future__ import annotations

import json
import plistlib
import subprocess
from pathlib import Path

import pytest

import scripts.run_telegram_ai_agent as agent
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt
from mango_mvp.channels.subscription_llm import AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, SAFE_FALLBACK_DRAFT_TEXT, SubscriptionDraftResult
from mango_mvp.integrations.draft_loop import DraftLoopConfigError


@pytest.mark.parametrize("brand", ("foton", "unpk"))
def test_launchd_renderer_pins_current_head_and_portable_paths(brand: str) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(root / "scripts/start_telegram_ai_agent_launchd.sh"), brand, "--render-only"],
        check=True,
        stdout=subprocess.PIPE,
    )

    payload = plistlib.loads(completed.stdout)
    assert payload["Label"] == f"com.mango.telegram-ai-agent.{brand}"
    assert payload["WorkingDirectory"] == str(root)
    assert payload["ProgramArguments"] == [
        "/bin/bash",
        str(root / "scripts/start_telegram_ai_agent_launchd.sh"),
        brand,
        "--run",
    ]
    assert payload["EnvironmentVariables"]["MANGO_TELEGRAM_EXPECTED_HEAD"] == subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert "/opt/homebrew/bin" in payload["EnvironmentVariables"]["PATH"]
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["StandardOutPath"] == str(Path.home() / f".mango_local/telegram_ai_agent/{brand}.out.log")


def test_runtime_status_preserves_release_identity_and_contains_no_dialogue(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(
        json.dumps({"head": "abc123", "kb_sha256": "kb-sha", "state_path": "/runtime/state.json"}),
        encoding="utf-8",
    )
    monkeypatch.setenv(agent.RUNTIME_STATUS_ENV, str(path))
    state = agent.DraftLoopState(tmp_path / "dialogue.json")
    state.payload["next_offset"] = 42
    state.payload["dialogue_memory"] = {"foton\tchat": {"turns": [{"text": "Секретный вопрос"}]}}

    agent.write_runtime_status("foton", status="ok", state=state)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["head"] == "abc123"
    assert payload["kb_sha256"] == "kb-sha"
    assert payload["next_offset"] == 42
    assert "Секретный вопрос" not in path.read_text(encoding="utf-8")


def test_error_status_can_be_written_when_offset_is_corrupt(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    monkeypatch.setenv(agent.RUNTIME_STATUS_ENV, str(path))
    state = agent.DraftLoopState(tmp_path / "dialogue.json")
    state.payload["next_offset"] = "bad"

    agent.write_runtime_status("foton", status="error", state=state, error="telegram_offset_corrupt")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error"] == "telegram_offset_corrupt"
    assert payload["next_offset"] is None


def test_launchd_wrapper_exposes_operational_commands_and_fresh_status_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/start_telegram_ai_agent_launchd.sh").read_text(encoding="utf-8")

    for mode in ("--status", "--smoke", "--stop", "--rollback"):
        assert mode in text
    assert "last_cycle_at" in text
    assert "kb_sha256" in text
    assert 'launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || exit 4' in text
    assert "status_ok || exit 4" in text
    assert "os.kill(pid, 0)" in text
    assert 'payload.get("code_root")' in text


class FakeTelegram:
    """Bot API поверх памяти: getUpdates честно фильтрует по offset, как настоящий."""

    def __init__(self, updates=()) -> None:
        self.updates = list(updates)
        self.sent: list[dict] = []
        self.actions: list[dict] = []
        self.tokens: list[str] = []
        self.status_code = 200

    def __call__(self, token, method, payload):
        self.tokens.append(token)
        if self.status_code != 200:
            raise RuntimeError(f"telegram_{method}_http_{self.status_code}")
        if method == "getUpdates":
            offset = int(payload.get("offset") or 0)
            if offset < 0:
                return {"ok": True, "result": self.updates[-1:]}
            return {"ok": True, "result": [item for item in self.updates if int(item["update_id"]) >= offset]}
        if method == "getWebhookInfo":
            return {"ok": True, "result": {"url": ""}}
        if method == "sendChatAction":
            self.actions.append(dict(payload))
            return {"ok": True, "result": {}}
        self.sent.append(dict(payload))
        return {"ok": True, "result": {}}


class FakeProvider:
    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = result if result is not None else _result()
        self.raises = raises
        self.calls: list[dict] = []

    def build_draft(self, client_message, *, context=None):
        self.calls.append({"client_message": client_message, "context": context})
        if self.raises is not None:
            raise self.raises
        return self.result


def _result(
    text: str = "Годовой курс стоит 37 000 ₽.",
    *,
    route: str = "bot_answer_self_for_pilot",
    action: str = "pass",
    checked: bool = True,
    verifier_checked: bool = True,
    verifier_action: str = "pass",
    error: str | None = None,
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=text,
        error=error,
        metadata={
            "authoritative_output_gate": {
                "schema_version": AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION,
                "checked": checked,
                "action": action,
            },
            "semantic_output_verifier": {
                "checked": verifier_checked,
                "action": verifier_action,
            },
        },
    )


def _update(update_id: int = 100, *, text: str = "Сколько стоит год?", chat_id: str = "555", **message) -> dict:
    payload = {
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": chat_id, "is_bot": False},
        **message,
    }
    if text is not None:
        payload["text"] = text
    return {"update_id": update_id, "message": payload}


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path / "telegram_ai_agent")


def _cycle(telegram, provider, *, brand: str = "foton", token: str = "secret-token", state=None):
    state = _dialogue_state(brand) if state is None else state
    if agent._state_offset(state) is None:
        agent._save_checkpoint(state, next_offset=0)
    agent.run_cycle(brand=brand, token=token, provider=provider, state=state)
    return state


def _dialogue_state(brand: str):
    return agent.dialogue_state(brand)


def _offset(brand: str) -> int | None:
    return agent._state_offset(_dialogue_state(brand))


def _set_offset(brand: str, value: int) -> None:
    agent._save_checkpoint(_dialogue_state(brand), next_offset=value)


def test_private_text_reaches_provider_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Сколько стоит год?")])
    provider = FakeProvider(_result("Годовой курс стоит 37 000 ₽."))
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, provider)

    assert provider.calls[0]["client_message"] == "Сколько стоит год?"
    assert provider.calls[0]["context"]["active_brand"] == "foton"
    assert provider.calls[0]["context"]["public_pilot_mode"]["sends_client_replies"] is True
    assert telegram.sent == [{"chat_id": "555", "text": "Годовой курс стоит 37 000 ₽."}]
    assert telegram.actions == [{"chat_id": "555", "action": "typing"}]
    assert _offset("foton") == 101


def test_fallback_uses_only_verified_contact_of_active_brand() -> None:
    foton = agent.fallback_text("foton")
    unpk = agent.fallback_text("unpk")

    assert "8 (495) 500-25-88" in foton
    assert "+7 (495) 150-81-51" not in foton
    assert "+7 (495) 150-81-51" in unpk
    assert "8 (495) 500-25-88" not in unpk


def test_brands_keep_separate_tokens_and_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    foton = FakeTelegram([_update(100)])
    unpk = FakeTelegram([_update(700)])

    monkeypatch.setattr(agent, "_api", foton)
    _cycle(foton, FakeProvider(), brand="foton", token="foton-token")
    monkeypatch.setattr(agent, "_api", unpk)
    _cycle(unpk, FakeProvider(), brand="unpk", token="unpk-token")

    assert set(foton.tokens) == {"foton-token"}
    assert set(unpk.tokens) == {"unpk-token"}
    assert _offset("foton") == 101
    assert _offset("unpk") == 701
    assert agent.BRAND_TOKEN_ENV == {
        "foton": "MANGO_TELEGRAM_FOTON_BOT_TOKEN",
        "unpk": "MANGO_TELEGRAM_UNPK_BOT_TOKEN",
    }


@pytest.mark.parametrize(
    ("brand", "marker"),
    [("foton", "«Фотон»"), ("unpk", "УНПК МФТИ")],
)
def test_start_introduces_ai_assistant_of_its_own_brand(brand: str, marker: str, monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="/start")])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    state = _cycle(telegram, provider, brand=brand)

    reply = telegram.sent[0]["text"]
    assert "ИИ-помощница" in reply
    assert marker in reply
    other = "УНПК МФТИ" if brand == "foton" else "«Фотон»"
    assert other not in reply
    assert provider.calls == []
    assert telegram.actions == []
    assert state.dialogue_memory_for(agent.DraftLoopKey(brand, "555")) == {}
    assert (brand, "555", "100") in state.processed_keys()


def test_start_bypasses_slow_model_message_in_same_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([
        _update(100, text="Сколько стоит год?", chat_id="slow-client"),
        _update(101, text="/start", chat_id="new-client"),
    ])

    class Provider(FakeProvider):
        def build_draft(self, client_message, *, context=None):
            assert telegram.sent == [{
                "chat_id": "new-client",
                "text": "Здравствуйте! Я — ИИ-помощница учебного центра «Фотон». Подскажу по курсам, ценам, расписанию и записи. Что вас интересует?",
            }]
            return super().build_draft(client_message, context=context)

    provider = Provider(_result("Годовой курс стоит 37 000 ₽."))
    monkeypatch.setattr(agent, "_api", telegram)

    state = _cycle(telegram, provider)

    assert [item["chat_id"] for item in telegram.sent] == ["new-client", "slow-client"]
    assert provider.calls[0]["client_message"] == "Сколько стоит год?"
    assert state.processed_keys() >= {
        ("foton", "slow-client", "100"),
        ("foton", "new-client", "101"),
    }
    assert _offset("foton") == 102


def test_start_does_not_advance_past_earlier_message_when_cycle_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    updates = [
        _update(100, text="Сколько стоит год?", chat_id="earlier-client"),
        _update(101, text="/start", chat_id="new-client"),
    ]
    first = FakeTelegram(updates)
    monkeypatch.setattr(agent, "_api", first)
    state = _dialogue_state("foton")
    agent._save_checkpoint(state, next_offset=0)

    with pytest.raises(RuntimeError, match="model unavailable"):
        agent.run_cycle(
            brand="foton",
            token="token",
            provider=FakeProvider(raises=RuntimeError("model unavailable")),
            state=state,
        )

    persisted = _dialogue_state("foton")
    assert agent._state_offset(persisted) == 0
    assert ("foton", "new-client", "101") in persisted.processed_keys()

    retry = FakeTelegram(updates)
    provider = FakeProvider(_result("Годовой курс стоит 37 000 ₽."))
    monkeypatch.setattr(agent, "_api", retry)
    agent.run_cycle(brand="foton", token="token", provider=provider, state=persisted)

    assert provider.calls[0]["client_message"] == "Сколько стоит год?"
    assert retry.sent == [{"chat_id": "earlier-client", "text": "Годовой курс стоит 37 000 ₽."}]
    assert _offset("foton") == 102


def test_group_channel_edited_and_bot_updates_are_ignored_but_offset_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    group = {"update_id": 100, "message": {"chat": {"id": "1", "type": "group"}, "from": {"id": "1"}, "text": "Цена?"}}
    edited = {"update_id": 101, "edited_message": {"chat": {"id": "2", "type": "private"}, "text": "Цена?"}}
    callback = {"update_id": 102, "callback_query": {"id": "cb", "data": "x"}}
    from_bot = {
        "update_id": 103,
        "message": {"chat": {"id": "3", "type": "private"}, "from": {"id": "3", "is_bot": True}, "text": "Цена?"},
    }
    telegram = FakeTelegram([group, edited, callback, from_bot])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, provider)

    assert telegram.sent == []
    assert provider.calls == []
    assert _offset("foton") == 104


def test_attachment_gets_fixed_text_and_never_calls_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text=None, photo=[{"file_id": "abc"}])])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    state = _cycle(telegram, provider)

    assert telegram.sent == [{"chat_id": "555", "text": "Пока я понимаю только текст. Напишите вопрос текстом, пожалуйста."}]
    assert provider.calls == []
    assert state.dialogue_memory_for(agent.DraftLoopKey("foton", "555")) == {}
    assert ("foton", "555", "100") in state.processed_keys()


def test_provider_programming_error_does_not_advance_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100)])
    crashing = FakeProvider(raises=RuntimeError("codex died mid-answer"))
    monkeypatch.setattr(agent, "_api", telegram)

    _set_offset("foton", 0)
    with pytest.raises(RuntimeError, match="codex died"):
        agent.run_cycle(brand="foton", token="token", provider=crashing, state=_dialogue_state("foton"))
    assert _offset("foton") == 0
    assert telegram.sent == []
    assert _dialogue_state("foton").payload["dialogue_memory"] == {}
    assert _dialogue_state("foton").processed_keys() == set()


def test_typing_failure_never_blocks_model_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100)])

    def api(token, method, payload):
        if method == "sendChatAction":
            raise ValueError("broken optional response")
        return telegram(token, method, payload)

    provider = FakeProvider(_result("Полезный ответ"))
    monkeypatch.setattr(agent, "_api", api)

    _cycle(telegram, provider)

    assert len(provider.calls) == 1
    assert telegram.sent == [{"chat_id": "555", "text": "Полезный ответ"}]


def test_safe_error_code_never_logs_arbitrary_exception_text() -> None:
    assert agent._safe_error_code(RuntimeError("telegram_getUpdates_transport_error")) == "telegram_getUpdates_transport_error"
    assert agent._safe_error_code(RuntimeError("telegram_sendMessage_http_500")) == "telegram_sendMessage_http_500"
    assert agent._safe_error_code(RuntimeError("client text: secret")) == "RuntimeError"
    assert agent._safe_error_code(RuntimeError("79991234567")) == "RuntimeError"
    assert agent._safe_error_code(RuntimeError("customer_12345")) == "RuntimeError"
    assert agent._safe_error_code(RuntimeError("telegram_sendMessage_http_79991234567")) == "RuntimeError"


def test_blocked_chat_does_not_stall_other_clients_or_save_unsent_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    updates = [
        _update(100, text="Первый вопрос", chat_id="blocked"),
        _update(101, text="Второй вопрос", chat_id="active"),
    ]
    telegram = FakeTelegram(updates)
    sent_attempts = 0

    def api(token, method, payload):
        nonlocal sent_attempts
        if method == "sendMessage":
            sent_attempts += 1
            if sent_attempts == 1:
                raise RuntimeError("telegram_sendMessage_http_403")
        return telegram(token, method, payload)

    provider = FakeProvider(_result("Полезный ответ"))
    monkeypatch.setattr(agent, "_api", api)

    state = _cycle(telegram, provider)

    assert len(provider.calls) == 2
    assert telegram.sent == [{"chat_id": "active", "text": "Полезный ответ"}]
    assert state.dialogue_memory_for(agent.DraftLoopKey("foton", "blocked")) == {}
    assert len(state.dialogue_memory_for(agent.DraftLoopKey("foton", "active"))["turns"]) == 2
    assert _offset("foton") == 102


def test_http_500_retries_saved_reply_without_second_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailsOnce(FakeTelegram):
        def __call__(self, token, method, payload):
            if method == "sendMessage" and not self.sent:
                self.sent.append({"failed": True})
                raise RuntimeError("telegram_sendMessage_http_500")
            return super().__call__(token, method, payload)

    telegram = FailsOnce([_update(100)])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)
    _set_offset("foton", 0)
    with pytest.raises(RuntimeError):
        agent.run_cycle(brand="foton", token="token", provider=provider, state=_dialogue_state("foton"))
    assert _offset("foton") == 0
    assert _dialogue_state("foton").payload["dialogue_memory"] == {}
    assert _dialogue_state("foton").processed_keys() == set()

    agent.run_cycle(brand="foton", token="token", provider=provider, state=_dialogue_state("foton"))
    assert _offset("foton") == 101
    assert telegram.sent[-1]["text"] == "Годовой курс стоит 37 000 ₽."
    assert len(provider.calls) == 1


def test_failed_delivery_recovery_is_never_retried_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _dialogue_state("foton")
    agent._save_delivery_intent(
        state,
        marker=("foton", "555", "100"),
        next_offset=101,
        key=None,
        memory=None,
        reply_text="Сохранённый ответ",
    )
    send_attempts = 0

    def api(_token, method, _payload):
        nonlocal send_attempts
        if method == "sendMessage":
            send_attempts += 1
            raise RuntimeError("telegram_sendMessage_transport_error")
        return {"ok": True, "result": []}

    monkeypatch.setattr(agent, "_api", api)
    provider = FakeProvider()
    with pytest.raises(RuntimeError, match="transport_error"):
        agent.run_cycle(brand="foton", token="token", provider=provider, state=state)

    agent.run_cycle(brand="foton", token="token", provider=provider, state=state)

    assert send_attempts == 1
    assert provider.calls == []
    assert agent._state_offset(state) == 101


def test_corrupt_pending_delivery_is_dropped_without_stopping_bot() -> None:
    state = _dialogue_state("foton")
    state.payload["telegram_pending_delivery"] = {"profile_id": "foton", "chat_id": "555"}
    state.save()

    assert agent._finish_delivery_intent(state) is False
    assert "telegram_pending_delivery" not in state.payload


def test_first_start_discards_old_backlog_and_then_answers_new_update(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Старый вопрос")])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    agent.run_cycle(brand="foton", token="token", provider=provider, state=_dialogue_state("foton"))
    assert telegram.sent == []
    assert provider.calls == []
    assert _offset("foton") == 101

    telegram.updates.append(_update(101, text="Новый вопрос"))
    agent.run_cycle(brand="foton", token="token", provider=provider, state=_dialogue_state("foton"))
    assert telegram.sent == [{"chat_id": "555", "text": "Годовой курс стоит 37 000 ₽."}]


def test_corrupt_offset_stops_instead_of_replaying_backlog() -> None:
    path = agent.STATE_DIR / "foton_dialogue_state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"dialogue_memory": {}, "next_offset": "bad"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="telegram_offset_corrupt"):
        _offset("foton")


def test_corrupt_dialogue_state_stops_instead_of_answering_without_memory() -> None:
    path = agent.STATE_DIR / "foton_dialogue_state.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DraftLoopConfigError, match="Invalid draft loop state JSON"):
        _dialogue_state("foton")


def test_structurally_corrupt_dialogue_state_stops_instead_of_losing_history() -> None:
    path = agent.STATE_DIR / "foton_dialogue_state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"dialogue_memory": {"foton\t555": {"turns": "BROKEN"}}}), encoding="utf-8")
    with pytest.raises(DraftLoopConfigError, match="Invalid Telegram dialogue memory state"):
        _dialogue_state("foton")


def test_second_process_for_the_same_brand_is_rejected() -> None:
    first = agent.acquire_lock("foton")
    try:
        with pytest.raises(SystemExit, match="уже запущен"):
            agent.acquire_lock("foton")
    finally:
        first.close()


def test_main_enables_and_checks_canonical_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram()
    calls = []
    provider_kwargs = {}
    monkeypatch.setattr(agent, "_api", telegram)
    monkeypatch.setattr(
        agent,
        "SubscriptionLlmDraftProvider",
        lambda **kwargs: provider_kwargs.update(kwargs) or FakeProvider(),
    )
    monkeypatch.setattr(agent.profile, "ensure_canonical_pilot_profile", lambda **kwargs: calls.append("ensure"))
    monkeypatch.setattr(agent.profile, "pilot_profile_selfcheck", lambda **kwargs: calls.append("check") or object())
    monkeypatch.setattr(agent.profile, "raise_for_failed_selfcheck", lambda check: calls.append("raise"))
    monkeypatch.setenv(agent.BRAND_TOKEN_ENV["foton"], "token")
    monkeypatch.delenv(agent.profile.ENFORCE_CANONICAL_PROFILE_ENV, raising=False)

    assert agent.main(["--brand", "foton", "--once"]) == 0
    assert agent.os.environ[agent.profile.ENFORCE_CANONICAL_PROFILE_ENV] == "1"
    assert calls == ["ensure", "check", "raise"]
    assert provider_kwargs["codex_isolated"] is True


@pytest.mark.parametrize(
    ("bad_result", "reason"),
    [
        (_result(error="timeout"), "provider error"),
        (_result(action="downgrade_keep_text"), "gate downgraded"),
        (_result(action="annotate"), "gate annotated"),
        (_result(checked=False), "gate not checked"),
        (_result(verifier_checked=False), "semantic verifier not checked"),
        (SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="37 000 ₽", metadata={"authoritative_output_gate": {"checked": True, "action": "pass"}}), "gate without canonical schema"),
        (_result("Автономный ответ не требуется, безопасный вариант ниже."), "internal manager draft"),
        (_result(SAFE_FALLBACK_DRAFT_TEXT), "manager promise stub"),
        (_result("Приняли обращение. Передам вопрос менеджеру, он ответит вам.", verifier_action="downgrade_keep_text"), "unavailable public handoff"),
    ],
)
def test_unsafe_model_output_never_reaches_the_client(
    bad_result: SubscriptionDraftResult, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    telegram = FakeTelegram([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, FakeProvider(bad_result))

    sent = telegram.sent[0]["text"]
    assert sent == agent.fallback_text("foton"), reason
    assert "менеджер" not in sent.casefold()
    assert sent != bad_result.draft_text


@pytest.mark.parametrize("route", ["draft_for_manager", "manager_only"])
def test_client_receives_safe_model_text_after_authoritative_gate(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, FakeProvider(_result("Уточните, пожалуйста, удобный формат: очно или онлайн?", route=route)))

    assert telegram.sent == [{"chat_id": "555", "text": "Уточните, пожалуйста, удобный формат: очно или онлайн?"}]


@pytest.mark.parametrize(
    ("model_text", "expected"),
    [
        ("Годовой курс стоит 37 000 ₽. [source_id=kb_release_v3_snapshot fact:price_year]", "Годовой курс стоит 37 000 ₽."),
        ("DEBUG: route=bot_answer_self_for_pilot\nЗдравствуйте!", "Здравствуйте!"),
        ('{"safety_flags":["pass"]}\nЗдравствуйте!', "Здравствуйте!"),
        ("Менеджеру: проверить этот текст перед отправкой клиенту.", agent.fallback_text("foton")),
        ("manager_checklist: проверить цену перед отправкой.", agent.fallback_text("foton")),
        ("provider_error: timeout; показать менеджеру.", agent.fallback_text("foton")),
        ("model=gpt-5.5", agent.fallback_text("foton")),
    ],
)
def test_internal_service_markers_are_stripped_from_the_client_text(
    model_text: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    telegram = FakeTelegram([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, FakeProvider(_result(model_text)))

    sent = telegram.sent[0]["text"]
    assert sent == expected
    assert "source_id" not in sent
    assert "kb_release" not in sent


def test_token_leaks_neither_into_the_raised_error_nor_into_the_state(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "7654321:AAH-super-secret-bot-token"

    class Response:
        status_code = 401

        @staticmethod
        def json():  # pragma: no cover - не должен вызываться на 401
            return {"ok": False}

    monkeypatch.setattr(agent.requests, "post", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError) as excinfo:
        agent._api(token, "getUpdates", {"offset": 0})
    assert token not in str(excinfo.value)
    assert str(excinfo.value) == "telegram_getUpdates_http_401"

    _set_offset("foton", 101)
    state = (agent.STATE_DIR / "foton_dialogue_state.json").read_text(encoding="utf-8")
    assert token not in state
    assert json.loads(state)["next_offset"] == 101


def test_transport_error_hides_token(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "7654321:AAH-super-secret-bot-token"
    monkeypatch.setattr(agent.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(agent.requests.ConnectionError(args[0])))

    with pytest.raises(RuntimeError) as excinfo:
        agent._api(token, "getWebhookInfo", {})

    assert str(excinfo.value) == "telegram_getWebhookInfo_transport_error"
    assert token not in str(excinfo.value)


def test_dialogue_memory_survives_restart_and_reaches_next_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Ребёнок закончил 8 класс, нужна физика онлайн")])
    first = FakeProvider(_result("Подскажу по онлайн-физике для 9 класса."))
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, first)
    restarted_state = _dialogue_state("foton")
    stored = restarted_state.dialogue_memory_for(agent.DraftLoopKey("foton", "555"))
    assert [turn["text"] for turn in stored["turns"]] == [
        "Ребёнок закончил 8 класс, нужна физика онлайн",
        "Подскажу по онлайн-физике для 9 класса.",
    ]
    assert stored["known_slots"]["subject"]["value"] == "физика"

    telegram.updates.append(_update(101, text="А сколько это стоит?"))
    second = FakeProvider(_result("Проверю актуальную стоимость этого формата."))
    _cycle(telegram, second, state=restarted_state)

    prompt_memory = second.calls[0]["context"]["dialogue_memory_view"]
    assert [turn["text"] for turn in prompt_memory["recent_turns"]][-3:] == [
        "Ребёнок закончил 8 класс, нужна физика онлайн",
        "Подскажу по онлайн-физике для 9 класса.",
        "А сколько это стоит?",
    ]
    assert second.calls[0]["context"]["dialogue_memory_view"]["known_slots"]["subject"] == "физика"


def test_rolling_summary_and_seven_latest_turns_reach_final_direct_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _dialogue_state("foton")
    key = agent.DraftLoopKey("foton", "555")
    state.set_dialogue_memory(
        key,
        {
            "active_brand": "foton",
            "turns": [
                {"role": "client" if index % 2 == 0 else "bot", "text": f"старая реплика {index}"}
                for index in range(12)
            ],
            "conversation_summary_short": "Ребёнок закончил 8 класс, интересует физика онлайн.",
        },
    )
    agent._save_checkpoint(state, next_offset=101)
    telegram = FakeTelegram([_update(101, text="А сколько это стоит?")])
    provider = FakeProvider(_result("Стоимость зависит от выбранного формата."))
    monkeypatch.setattr(agent, "_api", telegram)

    agent.run_cycle(brand="foton", token="token", provider=provider, state=state)

    context = provider.calls[0]["context"]
    assert len(context["recent_messages"]) == 8
    assert context["recent_messages"][0].startswith("Ранее в диалоге:")
    prompt = _build_direct_path_prompt("А сколько это стоит?", context=context)
    assert "Ребёнок закончил 8 класс, интересует физика онлайн." in prompt
    assert "старая реплика 11" in prompt


def test_fallback_is_remembered_but_does_not_close_client_question(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Когда проходят занятия?")])
    monkeypatch.setattr(agent, "_api", telegram)

    state = _cycle(telegram, FakeProvider(_result(route="blocked")))

    memory = state.dialogue_memory_for(agent.DraftLoopKey("foton", "555"))
    assert memory["turns"][-1]["text"] == agent.fallback_text("foton")
    assert memory["open_question"]["text"] == "Когда проходят занятия?"
    assert memory["open_question"]["answered"] is False


def test_restart_after_uncertain_sent_reply_retries_once_without_second_model(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Нужна физика онлайн")])
    provider = FakeProvider(_result("Подскажу по физике онлайн."))
    state = _dialogue_state("foton")
    agent._save_checkpoint(state, next_offset=0)
    real_save = state.save
    attempts = 0

    def fail_after_send():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("temporary disk error")
        real_save()

    monkeypatch.setattr(state, "save", fail_after_send)
    monkeypatch.setattr(agent, "_api", telegram)
    with pytest.raises(OSError, match="temporary disk error"):
        agent.run_cycle(brand="foton", token="token", provider=provider, state=state)
    assert len(provider.calls) == 1
    assert len(telegram.sent) == 1
    persisted_after_crash = _dialogue_state("foton")
    assert agent._state_offset(persisted_after_crash) == 0
    assert persisted_after_crash.payload["telegram_pending_delivery"]["message_id"] == "100"

    agent.run_cycle(brand="foton", token="token", provider=provider, state=persisted_after_crash)

    assert len(provider.calls) == 1
    assert len(telegram.sent) == 2
    persisted = _dialogue_state("foton")
    assert agent._state_offset(persisted) == 101
    assert len(persisted.dialogue_memory_for(agent.DraftLoopKey("foton", "555"))["turns"]) == 2


def test_missing_offset_recovers_from_durable_processed_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeTelegram([_update(100, text="Нужна физика")])
    monkeypatch.setattr(agent, "_api", first)
    _cycle(first, FakeProvider(_result("Подскажу по физике.")))
    path = agent.STATE_DIR / "foton_dialogue_state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("next_offset")
    path.write_text(json.dumps(payload), encoding="utf-8")

    second = FakeTelegram([_update(101, text="А онлайн есть?")])
    provider = FakeProvider(_result("Да, онлайн-формат есть."))
    monkeypatch.setattr(agent, "_api", second)
    agent.run_cycle(brand="foton", token="token", provider=provider, state=_dialogue_state("foton"))

    assert len(provider.calls) == 1
    assert second.sent == [{"chat_id": "555", "text": "Да, онлайн-формат есть."}]
    assert _offset("foton") == 102


def test_processed_update_is_not_generated_or_sent_twice_after_offset_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    update = _update(100, text="Нужна физика онлайн")
    first_telegram = FakeTelegram([update])
    monkeypatch.setattr(agent, "_api", first_telegram)
    _cycle(first_telegram, FakeProvider(_result("Подскажу по физике онлайн.")))

    _set_offset("foton", 0)
    replay_telegram = FakeTelegram([update])
    replay_provider = FakeProvider(_result("Повторный ответ"))
    monkeypatch.setattr(agent, "_api", replay_telegram)
    state = _cycle(replay_telegram, replay_provider, state=_dialogue_state("foton"))

    assert replay_provider.calls == []
    assert replay_telegram.sent == []
    assert len(state.dialogue_memory_for(agent.DraftLoopKey("foton", "555"))["turns"]) == 2
    assert _offset("foton") == 101


def test_uncertain_send_error_retries_saved_reply_once_without_second_model(monkeypatch: pytest.MonkeyPatch) -> None:
    update = _update(100, text="Нужна физика онлайн")
    telegram = FakeTelegram([update])
    provider = FakeProvider(_result("Подскажу по физике онлайн."))
    real_api = telegram.__call__

    def uncertain_api(token, method, payload):
        if method == "sendMessage":
            telegram.sent.append(dict(payload))
            raise RuntimeError("telegram_sendMessage_transport_error")
        return real_api(token, method, payload)

    monkeypatch.setattr(agent, "_api", uncertain_api)
    state = _dialogue_state("foton")
    agent._save_checkpoint(state, next_offset=0)

    with pytest.raises(RuntimeError, match="telegram_sendMessage_transport_error"):
        agent.run_cycle(brand="foton", token="token", provider=provider, state=state)

    persisted = _dialogue_state("foton")
    assert persisted.payload["telegram_pending_delivery"]["message_id"] == "100"

    retry = FakeTelegram([update])
    retry_provider = FakeProvider(_result("Повторный ответ"))
    monkeypatch.setattr(agent, "_api", retry)
    agent.run_cycle(brand="foton", token="token", provider=retry_provider, state=persisted)

    assert retry_provider.calls == []
    assert retry.sent == [{"chat_id": "555", "text": "Подскажу по физике онлайн."}]
    assert _offset("foton") == 101


def test_dialogue_memory_isolated_by_chat_and_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    foton = FakeTelegram([_update(100, text="Нужна математика", chat_id="111")])
    monkeypatch.setattr(agent, "_api", foton)
    _cycle(foton, FakeProvider(_result("Расскажу о математике.")), brand="foton")

    foton.updates.append(_update(101, text="Что можете предложить?", chat_id="222"))
    other_chat = FakeProvider(_result("Уточните класс, пожалуйста."))
    _cycle(foton, other_chat, brand="foton")
    other_turns = other_chat.calls[0]["context"]["dialogue_memory_view"]["recent_turns"]
    assert all("математик" not in str(item).casefold() for item in other_turns)

    unpk = FakeTelegram([_update(700, text="Что можете предложить?", chat_id="111")])
    monkeypatch.setattr(agent, "_api", unpk)
    other_brand = FakeProvider(_result("Уточните класс, пожалуйста."))
    _cycle(unpk, other_brand, brand="unpk")
    assert other_brand.calls[0]["context"]["dialogue_memory_view"]["recent_turns"][-1]["text"] == "Что можете предложить?"
