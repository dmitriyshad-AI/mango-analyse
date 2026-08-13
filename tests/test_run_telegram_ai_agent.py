from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_telegram_ai_agent as agent
from mango_mvp.channels.subscription_llm import AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, SAFE_FALLBACK_DRAFT_TEXT, SubscriptionDraftResult


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


def _cycle(telegram, provider, *, brand: str = "foton", token: str = "secret-token", memory=None) -> dict:
    memory = {} if memory is None else memory
    if agent.load_offset(brand) is None:
        agent.save_offset(brand, 0)
    agent.run_cycle(brand=brand, token=token, provider=provider, memory=memory)
    return memory


def test_private_text_reaches_provider_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Сколько стоит год?")])
    provider = FakeProvider(_result("Годовой курс стоит 37 000 ₽."))
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, provider)

    assert provider.calls[0]["client_message"] == "Сколько стоит год?"
    assert provider.calls[0]["context"]["active_brand"] == "foton"
    assert provider.calls[0]["context"]["public_pilot_mode"]["sends_client_replies"] is True
    assert telegram.sent == [{"chat_id": "555", "text": "Годовой курс стоит 37 000 ₽."}]
    assert agent.load_offset("foton") == 101


def test_brands_keep_separate_tokens_and_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    foton = FakeTelegram([_update(100)])
    unpk = FakeTelegram([_update(700)])

    monkeypatch.setattr(agent, "_api", foton)
    _cycle(foton, FakeProvider(), brand="foton", token="foton-token")
    monkeypatch.setattr(agent, "_api", unpk)
    _cycle(unpk, FakeProvider(), brand="unpk", token="unpk-token")

    assert set(foton.tokens) == {"foton-token"}
    assert set(unpk.tokens) == {"unpk-token"}
    assert agent.load_offset("foton") == 101
    assert agent.load_offset("unpk") == 701
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

    _cycle(telegram, provider, brand=brand)

    reply = telegram.sent[0]["text"]
    assert "ИИ-помощница" in reply
    assert marker in reply
    other = "УНПК МФТИ" if brand == "foton" else "«Фотон»"
    assert other not in reply
    assert provider.calls == []


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
    assert agent.load_offset("foton") == 104


def test_attachment_gets_fixed_text_and_never_calls_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text=None, photo=[{"file_id": "abc"}])])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    _cycle(telegram, provider)

    assert telegram.sent == [{"chat_id": "555", "text": "Пока я понимаю только текст. Напишите вопрос текстом, пожалуйста."}]
    assert provider.calls == []


def test_provider_programming_error_does_not_advance_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100)])
    crashing = FakeProvider(raises=RuntimeError("codex died mid-answer"))
    monkeypatch.setattr(agent, "_api", telegram)

    agent.save_offset("foton", 0)
    with pytest.raises(RuntimeError, match="codex died"):
        agent.run_cycle(brand="foton", token="token", provider=crashing, memory={})
    assert agent.load_offset("foton") == 0
    assert telegram.sent == []


def test_send_failure_does_not_advance_offset_and_restart_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailsOnce(FakeTelegram):
        def __call__(self, token, method, payload):
            if method == "sendMessage" and not self.sent:
                self.sent.append({"failed": True})
                raise RuntimeError("telegram_sendMessage_http_500")
            return super().__call__(token, method, payload)

    telegram = FailsOnce([_update(100)])
    monkeypatch.setattr(agent, "_api", telegram)
    agent.save_offset("foton", 0)
    with pytest.raises(RuntimeError):
        agent.run_cycle(brand="foton", token="token", provider=FakeProvider(), memory={})
    assert agent.load_offset("foton") == 0

    agent.run_cycle(brand="foton", token="token", provider=FakeProvider(), memory={})
    assert agent.load_offset("foton") == 101
    assert telegram.sent[-1]["text"] == "Годовой курс стоит 37 000 ₽."


def test_first_start_discards_old_backlog_and_then_answers_new_update(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram = FakeTelegram([_update(100, text="Старый вопрос")])
    provider = FakeProvider()
    monkeypatch.setattr(agent, "_api", telegram)

    agent.run_cycle(brand="foton", token="token", provider=provider, memory={})
    assert telegram.sent == []
    assert provider.calls == []
    assert agent.load_offset("foton") == 101

    telegram.updates.append(_update(101, text="Новый вопрос"))
    agent.run_cycle(brand="foton", token="token", provider=provider, memory={})
    assert telegram.sent == [{"chat_id": "555", "text": "Годовой курс стоит 37 000 ₽."}]


def test_corrupt_offset_stops_instead_of_replaying_backlog() -> None:
    path = agent.STATE_DIR / "foton_offset.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="telegram_offset_corrupt"):
        agent.load_offset("foton")


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
    monkeypatch.setattr(agent, "_api", telegram)
    monkeypatch.setattr(agent, "SubscriptionLlmDraftProvider", lambda **kwargs: FakeProvider())
    monkeypatch.setattr(agent.profile, "ensure_canonical_pilot_profile", lambda **kwargs: calls.append("ensure"))
    monkeypatch.setattr(agent.profile, "pilot_profile_selfcheck", lambda **kwargs: calls.append("check") or object())
    monkeypatch.setattr(agent.profile, "raise_for_failed_selfcheck", lambda check: calls.append("raise"))
    monkeypatch.setenv(agent.BRAND_TOKEN_ENV["foton"], "token")
    monkeypatch.delenv(agent.profile.ENFORCE_CANONICAL_PROFILE_ENV, raising=False)

    assert agent.main(["--brand", "foton", "--once"]) == 0
    assert agent.os.environ[agent.profile.ENFORCE_CANONICAL_PROFILE_ENV] == "1"
    assert calls == ["ensure", "check", "raise"]


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

    agent.save_offset("foton", 101)
    state = (agent.STATE_DIR / "foton_offset.json").read_text(encoding="utf-8")
    assert token not in state
    assert json.loads(state) == {"next_offset": 101}


def test_transport_error_hides_token(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "7654321:AAH-super-secret-bot-token"
    monkeypatch.setattr(agent.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(agent.requests.ConnectionError(args[0])))

    with pytest.raises(RuntimeError) as excinfo:
        agent._api(token, "getWebhookInfo", {})

    assert str(excinfo.value) == "telegram_getWebhookInfo_transport_error"
    assert token not in str(excinfo.value)
