from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path

import scripts.run_amo_wappi_draft_loop as runner
from mango_mvp.channels.pilot_profile_runtime import DIRECT_PATH_PILOT_CONFIG_ENV, ENFORCE_CANONICAL_PROFILE_ENV
from mango_mvp.channels.subscription_llm_parts.direct_path import _direct_path_recent_messages
from mango_mvp.integrations.amo_wappi_transport import TransportDenied
from mango_mvp.integrations.draft_loop import DraftLoopConfig, DraftLoopKey, DraftLoopPair, DraftLoopProfile, WappiHistoryMessage
from tests.test_bot_safe_runtime_context import _seed_bot_safe_timeline


def test_build_config_loads_profiles_pairs_and_keeps_state_outside_repo(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    pairs = tmp_path / "pairs.json"
    local_dir = tmp_path / ".mango_local" / "draft_loop"
    profiles.write_text(
        json.dumps([{"profile_id": "profile-foton", "brand": "foton", "channel": "telegram"}]),
        encoding="utf-8",
    )
    pairs.write_text(
        json.dumps(
            [
                {
                    "profile_id": "profile-foton",
                    "chat_id": "chat-1",
                    "lead_id": "49832125",
                    "expected_brand": "foton",
                }
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        profiles_file=profiles,
        pairs_file=pairs,
        phase1_config=tmp_path / "missing.json",
        local_dir=local_dir,
        stop_file=tmp_path / "STOP",
        manager_outgoing_visible="unknown",
    )

    config = runner.build_config(args)

    key = DraftLoopKey("profile-foton", "chat-1")
    assert config.brand_for_profile("profile-foton") == "foton"
    assert config.pair_for(key).lead_id == "49832125"
    assert config.state_path == local_dir / "state.json"
    assert config.journal_path == local_dir / "journal.jsonl"
    assert config.heartbeat_path == local_dir / "heartbeat.json"
    assert config.allowed_test_lead_ids == frozenset({"49832125"})


def test_context_builder_marks_draft_loop_as_not_sending_clients(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "test", "facts": [], "chunks": []}), encoding="utf-8")
    build_context = runner.build_context_builder(snapshot)

    context = build_context(DraftLoopKey("profile-foton", "chat-1"), ("Клиент: 9 класс",), "Цена?", "foton")

    assert context["client_identity"]["channel"] == "wappi_telegram"
    assert context["public_pilot_mode"]["sends_client_replies"] is False
    assert context["public_pilot_mode"]["no_crm_tallanto_write"] is True

    max_context = build_context(DraftLoopKey("profile-foton-max", "chat-1"), (), "Цена?", "foton", channel="max")
    assert max_context["client_identity"]["channel"] == "wappi_max"


def test_context_builder_keeps_wappi_summary_and_15_raw_messages(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "test", "facts": [], "chunks": []}), encoding="utf-8")
    build_context = runner.build_context_builder(snapshot)
    older_summary = "Ранее в диалоге: Клиент: сын в 7 классе, интересует физика онлайн"
    raw_messages = tuple(f"Клиент: последняя строка {idx}" for idx in range(20))

    context = build_context(DraftLoopKey("profile-foton", "chat-1"), (older_summary, *raw_messages), "Продолжим?", "foton")

    recent = tuple(context["recent_messages"])
    assert recent[0] == older_summary
    assert recent[1:] == raw_messages[-15:]
    assert context["public_pilot_mode"]["sends_client_replies"] is False


def test_direct_path_wappi_recent_messages_keeps_summary_and_15_raw_messages() -> None:
    older_summary = "Ранее в диалоге: Клиент: сын в 7 классе, интересует физика онлайн"
    raw_messages = tuple(f"Клиент: последняя строка {idx}" for idx in range(20))
    context = {
        "client_identity": {"channel": "wappi_telegram"},
        "recent_messages": (older_summary, *raw_messages),
    }

    recent = _direct_path_recent_messages(context, limit=8)

    assert recent[0] == older_summary
    assert recent[1:] == raw_messages[-15:]
    assert _direct_path_recent_messages({"client_identity": {"channel": "telegram"}, "recent_messages": raw_messages}, limit=8) == raw_messages[-8:]


def test_context_builder_injects_only_bot_safe_crm_context_when_enabled(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "test", "facts": [], "chunks": []}), encoding="utf-8")
    timeline_db, customer_id = _seed_bot_safe_timeline(tmp_path)
    key = DraftLoopKey("profile-foton", "chat-1")
    config = DraftLoopConfig(
        profiles={"profile-foton": DraftLoopProfile("profile-foton", "foton")},
        pairs={
            key: DraftLoopPair(
                key=key,
                lead_id="5001",
                contact_id="7001",
                expected_brand="foton",
            )
        },
    )
    monkeypatch.setenv("TELEGRAM_BOT_SAFE_CRM_CONTEXT", "1")
    build_context = runner.build_context_builder(
        snapshot,
        draft_config=config,
        customer_timeline_db=timeline_db,
        customer_timeline_allowed_root=tmp_path,
    )

    context = build_context(key, ("Клиент: что по расписанию?",), "Что дальше?", "foton")

    raw = json.dumps(context.get("read_only_customer_context"), ensure_ascii=False)
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in raw
    assert "УНПК: клиент интересовался выездной школой" not in raw
    assert "next_step_status" in raw
    assert customer_id not in raw
    assert "botsafe:" not in raw
    assert context["read_only_customer_context"]["timeline_context"]["safety"]["customer_profile_included"] is False


def test_context_builder_keeps_bot_safe_crm_context_off_by_default(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "test", "facts": [], "chunks": []}), encoding="utf-8")
    timeline_db, _customer_id = _seed_bot_safe_timeline(tmp_path)
    key = DraftLoopKey("profile-foton", "chat-1")
    config = DraftLoopConfig(
        profiles={"profile-foton": DraftLoopProfile("profile-foton", "foton")},
        pairs={key: DraftLoopPair(key=key, lead_id="5001", expected_brand="foton")},
    )
    monkeypatch.delenv("TELEGRAM_BOT_SAFE_CRM_CONTEXT", raising=False)
    build_context = runner.build_context_builder(
        snapshot,
        draft_config=config,
        customer_timeline_db=timeline_db,
        customer_timeline_allowed_root=tmp_path,
    )

    context = build_context(key, (), "Что дальше?", "foton")

    assert "read_only_customer_context" not in context


def test_safe_transport_blocks_unlisted_wappi_get() -> None:
    ai_office_config = runner.AiOfficeClientConfig(base_url="https://api.fotonai.online", api_key="key")
    wappi_config = runner.WappiClientConfig(base_url="https://wappi.pro", telegram_token="token")
    transport = runner.build_safe_transport(ai_office_config, wappi_config)

    try:
        transport(method="GET", url="https://wappi.pro/tapi/profile/queue/purge?profile_id=p")
    except TransportDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("queue purge must be denied")

    try:
        transport(method="POST", url="https://educent.amocrm.ru/api/v4/leads/49832125/notes")
    except TransportDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("direct amoCRM note writes must be denied")


def test_safe_transport_allows_amo_events_get_only() -> None:
    calls = []
    transport = runner.DefaultDenyTransport(
        lambda **kwargs: calls.append(kwargs) or {"ok": True},
        policy=runner.SafeTransportPolicy(
            wappi_hosts=frozenset({"wappi.pro"}),
            amo_read_hosts=frozenset({"educent.amocrm.ru"}),
            ai_office_hosts=frozenset({"api.fotonai.online"}),
        ),
    )

    assert transport(method="GET", url="https://educent.amocrm.ru/api/v4/events?filter[type][]=incoming_chat_message") == {"ok": True}

    try:
        transport(method="POST", url="https://educent.amocrm.ru/api/v4/events")
    except TransportDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("AMO events POST must be denied")


def test_build_runner_uses_gated_canonical_profile_helper(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DIRECT_PATH_PILOT_CONFIG_ENV, raising=False)
    monkeypatch.delenv(ENFORCE_CANONICAL_PROFILE_ENV, raising=False)
    monkeypatch.setattr(runner, "load_env_file", lambda _path: {})
    monkeypatch.setattr(runner, "build_config", lambda _args: SimpleNamespace(state_path=tmp_path / "state.json"))
    monkeypatch.setattr(runner.AiOfficeClientConfig, "from_env", staticmethod(lambda: SimpleNamespace(base_url="https://api.fotonai.online")))
    monkeypatch.setattr(runner.WappiClientConfig, "from_env", staticmethod(lambda: SimpleNamespace(base_url="https://wappi.pro")))
    monkeypatch.setattr(runner, "build_safe_transport", lambda _ai, _wappi: object())
    monkeypatch.setattr(runner, "SubscriptionLlmDraftProvider", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "WappiPhase1Client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "AiOfficeAmoNoteClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "build_context_builder", lambda _snapshot, **_kwargs: object())
    monkeypatch.setattr(runner, "DraftLoopState", lambda _path: object())
    monkeypatch.setattr(runner, "build_auto_resolver", lambda _args: None)
    monkeypatch.setattr(runner, "AmoWappiDraftLoop", lambda **kwargs: kwargs)
    args = argparse.Namespace(
        env_file=tmp_path / ".env",
        ai_office_env_file=tmp_path / ".env.ai",
        model="gpt-5.5",
        reasoning="xhigh",
        timeout_sec=240,
        snapshot=tmp_path / "snapshot.json",
        customer_timeline_db=None,
        customer_timeline_allowed_root=None,
        customer_timeline_tenant="mango",
    )

    runner.build_runner(args)
    assert DIRECT_PATH_PILOT_CONFIG_ENV not in os.environ

    monkeypatch.setenv(ENFORCE_CANONICAL_PROFILE_ENV, "1")
    runner.build_runner(args)
    assert os.environ[DIRECT_PATH_PILOT_CONFIG_ENV] == "pilot_gold_v1"
    os.environ.pop(DIRECT_PATH_PILOT_CONFIG_ENV, None)


class FakeMcp:
    def __init__(self, contacts=None, leads=None, events=None) -> None:
        self.contacts = contacts or []
        self.leads = {str(item["id"]): item for item in (leads or [])}
        self.events = events or []
        self.calls = []

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls.append({"path": path, "params": params or {}, "limit": limit})
        if path == "events":
            return {"_embedded": {"events": self.events}}
        if path == "contacts":
            query = str((params or {}).get("query") or "")
            contacts = []
            for contact in self.contacts:
                haystack = json.dumps(contact, ensure_ascii=False)
                if query in haystack:
                    contacts.append(contact)
            return {"_embedded": {"contacts": contacts}}
        if path.startswith("contacts/"):
            contact_id = path.split("/", 1)[1]
            return next((item for item in self.contacts if str(item.get("id")) == contact_id), {})
        if path.startswith("leads/"):
            lead_id = path.split("/", 1)[1]
            return self.leads.get(lead_id, {})
        raise AssertionError(path)


class EventFailMcp(FakeMcp):
    def amo_api_get(self, *, path, params=None, limit=50):
        if path == "events":
            raise RuntimeError("events unavailable")
        return super().amo_api_get(path=path, params=params, limit=limit)


def _contact(contact_id="111", *, telegram_id="", phone="", leads=("49762441",)):
    fields = []
    if telegram_id:
        fields.append({"field_name": "Telegram ID", "values": [{"value": telegram_id}]})
    if phone:
        fields.append({"field_code": "PHONE", "field_name": "Телефон", "values": [{"value": phone}]})
    return {
        "id": contact_id,
        "custom_fields_values": fields,
        "_embedded": {"leads": [{"id": int(item)} for item in leads]},
    }


def _lead(lead_id="49762441", *, status_id=123, closed_at=None, deleted=False, org="", contacts=()):
    fields = []
    if org:
        fields.append({"field_name": "Организация", "values": [{"value": org}]})
    return {
        "id": int(lead_id),
        "status_id": status_id,
        "closed_at": closed_at,
        "is_deleted": deleted,
        "pipeline_id": 999,
        "custom_fields_values": fields,
        "_embedded": {"contacts": [{"id": int(item)} for item in contacts]},
    }


def _event(
    *,
    event_id="evt-1",
    event_type="incoming_chat_message",
    ts=1004,
    lead_id="50101349",
    contact_id="77345755",
    talk_id="3040",
    origin="pro.wappi.tg",
):
    return {
        "id": event_id,
        "type": event_type,
        "entity_type": "lead",
        "entity_id": int(lead_id),
        "created_at": ts,
        "value_after": [{"message": {"origin": origin, "talk_id": int(talk_id), "id": f"msg-{event_id}"}}],
        "_embedded": {"entity": {"linked_talk_contact_id": int(contact_id)}},
    }


def _wappi_message(*, profile_id="profile-foton", chat_id="758394977", message_id="15623", ts=1000, from_me=False):
    return WappiHistoryMessage(
        profile_id=profile_id,
        chat_id=chat_id,
        message_id=message_id,
        text="Есть ли места?",
        message_type="text",
        timestamp=ts,
        from_me=from_me,
    )


def _resolver(*, contacts=None, leads=None, events=None, stoplist=None, stoplist_error=""):
    return runner.AmoAutoResolver(
        client=FakeMcp(contacts=contacts, leads=leads, events=events),
        shared_phone_stoplist=set(stoplist or ()),
        stoplist_error=stoplist_error,
    )


def test_auto_resolver_prefers_unique_amo_chat_event_over_contact_search() -> None:
    profile = DraftLoopProfile("profile-unpk", "unpk", "telegram")
    key = DraftLoopKey("profile-unpk", "758394977")
    current = _wappi_message(profile_id=profile.profile_id)
    previous_outgoing = _wappi_message(profile_id=profile.profile_id, message_id="15624", ts=1006, from_me=True)
    resolver = _resolver(
        contacts=[],
        leads=[_lead("50101349", contacts=("77345755",), org="")],
        events=[
            _event(),
            _event(event_id="evt-2", event_type="outgoing_chat_message", ts=1006),
        ],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[current, previous_outgoing], message=current)

    assert result["status"] == "matched"
    assert result["lead_id"] == "50101349"
    assert result["contact_id"] == "77345755"
    assert result["match_key"] == "amo_chat_event"
    assert result["match_value"] == "talk:3040"
    assert result["amo_talk_id"] == "3040"
    assert result["amo_sequence_match_count"] == 2
    assert resolver.client.calls[0]["path"] == "events"
    assert not any(call["path"] == "contacts" for call in resolver.client.calls)


def test_auto_resolver_rejects_ambiguous_amo_chat_event_without_fallback() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    current = _wappi_message(profile_id=profile.profile_id, chat_id="123456")
    previous_outgoing = _wappi_message(profile_id=profile.profile_id, chat_id="123456", message_id="prev", ts=990, from_me=True)
    resolver = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", contacts=("111",))],
        events=[
            _event(event_id="evt-1", lead_id="1", contact_id="111", talk_id="10"),
            _event(event_id="evt-1-out", event_type="outgoing_chat_message", ts=990, lead_id="1", contact_id="111", talk_id="10"),
            _event(event_id="evt-2", lead_id="2", contact_id="222", talk_id="20", ts=1005),
            _event(event_id="evt-2-out", event_type="outgoing_chat_message", ts=990, lead_id="2", contact_id="222", talk_id="20"),
        ],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[previous_outgoing, current], message=current)

    assert result["status"] == "rejected"
    assert result["reason"] == "amo_chat_event_ambiguous"
    assert result["candidate_count"] == 2
    assert not any(call["path"] == "contacts" for call in resolver.client.calls)


def test_auto_resolver_rejects_single_unconfirmed_amo_chat_event_without_fallback() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    current = _wappi_message(profile_id=profile.profile_id, chat_id="123456")
    resolver = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", contacts=("111",))],
        events=[_event(event_id="evt-1", lead_id="1", contact_id="111", talk_id="10")],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[current], message=current)

    assert result["status"] == "rejected"
    assert result["reason"] == "amo_chat_event_sequence_unconfirmed"
    assert result["sequence_points_count"] == 1
    assert not any(call["path"] == "contacts" for call in resolver.client.calls)


def test_auto_resolver_rejects_event_contact_not_linked_to_lead() -> None:
    profile = DraftLoopProfile("profile-unpk", "unpk", "telegram")
    key = DraftLoopKey("profile-unpk", "758394977")
    current = _wappi_message(profile_id=profile.profile_id)
    previous_outgoing = _wappi_message(profile_id=profile.profile_id, message_id="15624", ts=1006, from_me=True)
    resolver = _resolver(
        leads=[_lead("50101349", contacts=("111111",), org="")],
        events=[
            _event(),
            _event(event_id="evt-2", event_type="outgoing_chat_message", ts=1006),
        ],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[current, previous_outgoing], message=current)

    assert result["status"] == "rejected"
    assert result["reason"] == "event_contact_not_linked_to_lead"


def test_auto_resolver_rejects_event_lead_without_contact_readback() -> None:
    profile = DraftLoopProfile("profile-unpk", "unpk", "telegram")
    key = DraftLoopKey("profile-unpk", "758394977")
    current = _wappi_message(profile_id=profile.profile_id)
    previous_outgoing = _wappi_message(profile_id=profile.profile_id, message_id="15624", ts=1006, from_me=True)
    resolver = _resolver(
        leads=[_lead("50101349", contacts=(), org="")],
        events=[
            _event(),
            _event(event_id="evt-2", event_type="outgoing_chat_message", ts=1006),
        ],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[current, previous_outgoing], message=current)

    assert result["status"] == "rejected"
    assert result["reason"] == "event_lead_contacts_missing"


def test_auto_resolver_rejects_when_amo_events_unavailable_without_fallback() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    resolver = runner.AmoAutoResolver(
        client=EventFailMcp(contacts=[_contact(telegram_id="123456", leads=("1",))], leads=[_lead("1", contacts=("111",))]),
        shared_phone_stoplist=set(),
    )

    result = resolver(
        key=key,
        profile=profile,
        dialog={},
        messages=[_wappi_message(profile_id=profile.profile_id, chat_id="123456")],
        message=_wappi_message(profile_id=profile.profile_id, chat_id="123456"),
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "amo_chat_event_unavailable"
    assert not any(call["path"] == "contacts" for call in resolver.client.calls)


def test_auto_resolver_falls_back_to_exact_telegram_when_event_is_absent() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    resolver = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", contacts=("111",))],
        events=[],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[], message=_wappi_message(profile_id=profile.profile_id, chat_id="123456"))

    assert result["status"] == "matched"
    assert result["lead_id"] == "1"
    assert result["match_key"] == "Telegram ID"
    assert [call["path"] for call in resolver.client.calls[:2]] == ["events", "contacts"]


def test_auto_resolver_rejects_closed_deleted_zero_and_multi_active_leads() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")

    closed = _resolver(contacts=[_contact(telegram_id="123456", leads=("49804475",))], leads=[_lead("49804475", status_id=143, closed_at=1)])
    assert closed(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "closed_lead"

    deleted = _resolver(contacts=[_contact(telegram_id="123456", leads=("111",))], leads=[_lead("111", deleted=True)])
    assert deleted(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "deleted_lead"

    zero = _resolver(contacts=[_contact(telegram_id="123456", leads=())], leads=[])
    assert zero(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "no_active_lead"

    multi = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1", "2"))],
        leads=[_lead("1"), _lead("2")],
    )
    assert multi(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "multi_active_lead"


def test_auto_resolver_rejects_username_only_and_duplicate_telegram_contacts() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    resolver = _resolver(contacts=[], leads=[])

    assert resolver(key=DraftLoopKey("profile-foton", "top_coach"), profile=profile, dialog={}, messages=[], message=None)["reason"] == "username_only"

    resolver = _resolver(
        contacts=[_contact("1", telegram_id="123456"), _contact("2", telegram_id="123456")],
        leads=[_lead()],
    )
    assert resolver(key=DraftLoopKey("profile-foton", "123456"), profile=profile, dialog={}, messages=[], message=None)["reason"] == "multi_contact"


def test_auto_resolver_rejects_max_numeric_id_shared_phone_text_phone_and_multi_contact() -> None:
    profile = DraftLoopProfile("profile-max", "foton", "max")
    key = DraftLoopKey("profile-max", "123456")
    resolver = _resolver(
        contacts=[_contact("1", telegram_id="123456", phone="+79990000000")],
        leads=[_lead()],
        stoplist={"79990000000"},
    )

    assert resolver(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "max_phone_missing"
    assert resolver(key=key, profile=profile, dialog={"last_message": {"body": "+7 999 000-00-00"}}, messages=[], message=None)["reason"] == "max_phone_missing"
    assert resolver(key=key, profile=profile, dialog={"phone": "+7 999 000-00-00"}, messages=[], message=None)["reason"] == "shared_phone"

    resolver = _resolver(
        contacts=[_contact("1", phone="+79990000001"), _contact("2", phone="+79990000001")],
        leads=[_lead()],
    )
    assert resolver(key=key, profile=profile, dialog={"phone": "+7 999 000-00-01"}, messages=[], message=None)["reason"] == "multi_contact"


def test_auto_resolver_rejects_brand_mismatch_and_accepts_empty_organization() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    mismatch = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", org="УНПК МФТИ")],
    )
    assert mismatch(key=key, profile=profile, dialog={}, messages=[], message=None)["reason"] == "brand_mismatch"

    ok = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", org="")],
    )
    result = ok(key=key, profile=profile, dialog={}, messages=[], message=None)
    assert result["status"] == "matched"
    assert result["lead_id"] == "1"


def test_auto_resolver_includes_organization_snapshot_for_review() -> None:
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    resolver = _resolver(
        contacts=[_contact(telegram_id="123456", leads=("1",))],
        leads=[_lead("1", org="Фотон")],
    )

    result = resolver(key=key, profile=profile, dialog={}, messages=[], message=None)

    assert result["status"] == "matched"
    assert result["lead_snapshot"]["organization_brand"] == "foton"
    assert result["lead_snapshot"]["organization_values"] == ["Фотон"]


def test_load_phone_stoplist_uses_plural_default_and_legacy_fallback(tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    secrets = fake_home / ".mango_secrets"
    secrets.mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_STOPLIST_PATH", secrets / "shared_phones_stoplist.json")
    monkeypatch.setattr(runner, "LEGACY_STOPLIST_PATH", secrets / "shared_phone_stoplist.json")

    (secrets / "shared_phone_stoplist.json").write_text(json.dumps({"phones": ["+7 999 000-00-00"]}), encoding="utf-8")
    phones, error = runner._load_phone_stoplist(runner.DEFAULT_STOPLIST_PATH)
    assert phones == {"+79990000000"}
    assert error == ""

    (secrets / "shared_phones_stoplist.json").write_text(json.dumps({"phones": ["+7 999 000-00-01"]}), encoding="utf-8")
    phones, error = runner._load_phone_stoplist(runner.DEFAULT_STOPLIST_PATH)
    assert phones == {"+79990000001"}
    assert error == ""
