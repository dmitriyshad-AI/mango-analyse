from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.run_amo_wappi_draft_loop as runner
from mango_mvp.integrations.amo_wappi_transport import TransportDenied
from mango_mvp.integrations.draft_loop import DraftLoopKey, DraftLoopProfile


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


class FakeMcp:
    def __init__(self, contacts=None, leads=None) -> None:
        self.contacts = contacts or []
        self.leads = {str(item["id"]): item for item in (leads or [])}
        self.calls = []

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls.append({"path": path, "params": params or {}, "limit": limit})
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


def _lead(lead_id="49762441", *, status_id=123, closed_at=None, deleted=False, org=""):
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
    }


def _resolver(
    *,
    contacts=None,
    leads=None,
    stoplist=None,
    stoplist_error="",
    cache=None,
    allowed_channels=frozenset({"telegram", "max"}),
    throttle_sec=0.0,
    retry_after_sec=0.0,
    sleep_fn=None,
):
    return runner.AmoAutoResolver(
        client=FakeMcp(contacts=contacts, leads=leads),
        shared_phone_stoplist=set(stoplist or ()),
        stoplist_error=stoplist_error,
        cache=cache,
        allowed_channels=frozenset(allowed_channels),
        throttle_sec=throttle_sec,
        retry_after_sec=retry_after_sec,
        sleep_fn=sleep_fn or (lambda seconds: None),
    )


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


def test_auto_resolver_cache_hit_revalidates_open_lead_without_contact_search(tmp_path: Path) -> None:
    cache = runner.AutoResolverCache(tmp_path / "cache.json", ttl_sec=3600, time_fn=lambda: 1000)
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    cache.put(
        key,
        profile=profile,
        candidate={"lead_id": "1", "contact_id": "111", "match_key": "Telegram ID", "match_value": "123456"},
    )
    resolver = _resolver(contacts=[], leads=[_lead("1", org="Фотон")], cache=cache)

    result = resolver(key=key, profile=profile, dialog={}, messages=[], message=None)

    assert result["status"] == "matched"
    assert result["cache"] == "hit"
    assert result["lead_id"] == "1"
    assert [call["path"] for call in resolver.client.calls] == ["leads/1"]


def test_auto_resolver_cache_does_not_return_closed_stale_lead(tmp_path: Path) -> None:
    cache = runner.AutoResolverCache(tmp_path / "cache.json", ttl_sec=3600, time_fn=lambda: 1000)
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")
    key = DraftLoopKey("profile-foton", "123456")
    cache.put(
        key,
        profile=profile,
        candidate={"lead_id": "1", "contact_id": "111", "match_key": "Telegram ID", "match_value": "123456"},
    )
    resolver = _resolver(contacts=[], leads=[_lead("1", status_id=143, closed_at=1)], cache=cache)

    result = resolver(key=key, profile=profile, dialog={}, messages=[], message=None)

    assert result["status"] == "rejected"
    assert result["reason"] == "username_only"
    assert cache.get(key) is None


def test_auto_resolver_can_disable_max_channel() -> None:
    profile = DraftLoopProfile("profile-max", "foton", "max")
    resolver = _resolver(contacts=[_contact(phone="+79990000000")], leads=[_lead()], allowed_channels={"telegram"})

    result = resolver(key=DraftLoopKey("profile-max", "max-chat-1"), profile=profile, dialog={"phone": "+7 999 000-00-00"}, messages=[], message=None)

    assert result["status"] == "rejected"
    assert result["reason"] == "auto_resolver_channel_disabled"


def test_auto_resolver_retries_rate_limit_once() -> None:
    class RateLimitOnceMcp(FakeMcp):
        def __init__(self) -> None:
            super().__init__(contacts=[_contact(telegram_id="123456", leads=("1",))], leads=[_lead("1")])
            self.failed = False

        def amo_api_get(self, *, path, params=None, limit=50):
            if path == "contacts" and not self.failed:
                self.failed = True
                raise RuntimeError("HTTP 429: Tallanto rate limit")
            return super().amo_api_get(path=path, params=params, limit=limit)

    sleeps: list[float] = []
    resolver = runner.AmoAutoResolver(
        client=RateLimitOnceMcp(),
        shared_phone_stoplist=set(),
        throttle_sec=0.0,
        retry_after_sec=2.5,
        sleep_fn=sleeps.append,
    )
    profile = DraftLoopProfile("profile-foton", "foton", "telegram")

    result = resolver(key=DraftLoopKey("profile-foton", "123456"), profile=profile, dialog={}, messages=[], message=None)

    assert result["status"] == "matched"
    assert sleeps == [2.5]


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
