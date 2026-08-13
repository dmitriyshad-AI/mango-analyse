from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_telegram_ai_agent as agent
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt
from mango_mvp.channels.subscription_llm import AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, SAFE_FALLBACK_DRAFT_TEXT, SubscriptionDraftResult
from mango_mvp.integrations.draft_loop import DraftLoopConfigError


class FakeTelegram:
    """Bot API поверх памяти: getUpdates честно фильтрует по offset, как настоящий."""

    def __init__(self, updates=()) -> None:
        self.updates = list(updates)
        self.sent: list[dict] = []
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
    error: str | None = None,
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=text,
        error=error,
        metadata={"authoritative_output_gate": {"schema_version": AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, "checked": checked, "action": action}},
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
    assert _offset("foton") == 101


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
    assert state.dialogue_memory_for(agent.DraftLoopKey(brand, "555")) == {}
    assert (brand, "555", "100") in state.processed_keys()


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


def test_send_failure_does_not_advance_offset_and_restart_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailsOnce(FakeTelegram):
        def __call__(self, token, method, payload):
            if method == "sendMessage" and not self.sent:
                self.sent.append({"failed": True})
                raise RuntimeError("telegram_sendMessage_http_500")
            return super().__call__(token, method, payload)

    telegram = FailsOnce([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)
    _set_offset("foton", 0)
    with pytest.raises(RuntimeError):
        agent.run_cycle(brand="foton", token="token", provider=FakeProvider(), state=_dialogue_state("foton"))
    assert _offset("foton") == 0
    assert _dialogue_state("foton").payload["dialogue_memory"] == {}
    assert _dialogue_state("foton").processed_keys() == set()

    agent.run_cycle(brand="foton", token="token", provider=FakeProvider(), state=_dialogue_state("foton"))
    assert _offset("foton") == 101
    assert telegram.sent[-1]["text"] == "Годовой курс стоит 37 000 ₽."


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
        (_result(route="manager_only"), "manager route"),
        (_result(route="draft_for_manager"), "downgraded route"),
        (_result(action="downgrade_keep_text"), "gate downgraded"),
        (_result(action="annotate"), "gate annotated"),
        (_result(checked=False), "gate not checked"),
        (SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="37 000 ₽", metadata={"authoritative_output_gate": {"checked": True, "action": "pass"}}), "gate without canonical schema"),
        (_result("Автономный ответ не требуется, безопасный вариант ниже."), "internal manager draft"),
        (_result(SAFE_FALLBACK_DRAFT_TEXT), "manager promise stub"),
    ],
)
def test_unsafe_model_output_never_reaches_the_client(
    bad_result: SubscriptionDraftResult, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    telegram = FakeTelegram([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, FakeProvider(bad_result))

    sent = telegram.sent[0]["text"]
    assert sent == agent.FALLBACK_TEXT, reason
    assert "менеджер" not in sent.casefold()
    assert sent != bad_result.draft_text


@pytest.mark.parametrize(
    ("model_text", "expected"),
    [
        ("Годовой курс стоит 37 000 ₽. [source_id=kb_release_v3_snapshot fact:price_year]", "Годовой курс стоит 37 000 ₽."),
        ("DEBUG: route=bot_answer_self_for_pilot\nЗдравствуйте!", "Здравствуйте!"),
        ('{"safety_flags":["pass"]}\nЗдравствуйте!', "Здравствуйте!"),
        ("Менеджеру: проверить этот текст перед отправкой клиенту.", agent.FALLBACK_TEXT),
        ("manager_checklist: проверить цену перед отправкой.", agent.FALLBACK_TEXT),
        ("provider_error: timeout; показать менеджеру.", agent.FALLBACK_TEXT),
        ("model=gpt-5.5", agent.FALLBACK_TEXT),
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

    state = _cycle(telegram, FakeProvider(_result(route="manager_only")))

    memory = state.dialogue_memory_for(agent.DraftLoopKey("foton", "555"))
    assert memory["turns"][-1]["text"] == agent.FALLBACK_TEXT
    assert memory["open_question"]["text"] == "Когда проходят занятия?"
    assert memory["open_question"]["answered"] is False


def test_transient_checkpoint_failure_retries_state_without_second_model_or_send(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Нужна физика онлайн")])
    provider = FakeProvider(_result("Подскажу по физике онлайн."))
    state = _dialogue_state("foton")
    agent._save_checkpoint(state, next_offset=0)
    real_save = state.save
    attempts = 0

    def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary disk error")
        real_save()

    monkeypatch.setattr(state, "save", fail_once)
    monkeypatch.setattr(agent, "_api", telegram)
    with pytest.raises(OSError, match="temporary disk error"):
        agent.run_cycle(brand="foton", token="token", provider=provider, state=state)
    assert len(provider.calls) == 1
    assert len(telegram.sent) == 1
    assert agent._state_offset(state) == 0

    agent.run_cycle(brand="foton", token="token", provider=provider, state=state)

    assert len(provider.calls) == 1
    assert len(telegram.sent) == 1
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
