from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.integrations.amo_wappi_phase1 import (
    AMO_WAPPI_CONFIG_PATH_ENV,
    DRAFT_NOTE_MARKER,
    AmoClientConfig,
    AmoPhase1Client,
    AmoWappiConfigError,
    AmoWappiPhase1Config,
    AmoWappiWriteBlocked,
    ManagerEditLogRecord,
    WappiClientConfig,
    WappiPhase1Client,
    append_manager_edit_log,
    build_draft_note_text,
    load_env_file,
)


def test_amo_wappi_env_file_loads_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "amo_wappi.env"
    env_file.write_text(
        "\n".join(
            [
                "AMOCRM_BASE_URL=https://educent.amocrm.ru",
                "AMOCRM_ACCESS_TOKEN=amo-token",
                "WAPPI_TELEGRAM_TOKEN=wappi-token",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("AMOCRM_BASE_URL", "AMOCRM_ACCESS_TOKEN", "WAPPI_TELEGRAM_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_env_file(env_file)

    assert loaded["AMOCRM_BASE_URL"] == "https://educent.amocrm.ru"
    assert AmoClientConfig.from_env().access_token == "amo-token"
    assert WappiClientConfig.from_env().telegram_token == "wappi-token"


def test_phase1_config_maps_profile_to_brand_and_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "amo_wappi_phase1.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "tg-foton": {"brand": "foton", "channel": "telegram"},
                    "max-unpk": {"brand": "unpk", "channel": "max"},
                },
                "allowed_test_lead_ids": ["123"],
                "manager_edit_log_path": str(tmp_path / "edits.jsonl"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(AMO_WAPPI_CONFIG_PATH_ENV, str(config_path))

    config = AmoWappiPhase1Config.from_file()

    assert config.brand_for_profile("tg-foton") == "foton"
    assert config.brand_for_profile("max-unpk") == "unpk"
    assert config.require_note_allowed("123") == "123"
    with pytest.raises(AmoWappiConfigError):
        config.brand_for_profile("unknown")


def test_wappi_profile_list_uses_readonly_profile_endpoints() -> None:
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"profiles": [{"id": "profile-1", "name": "Foton TG"}]}

    client = WappiPhase1Client(
        WappiClientConfig(base_url="https://wappi.pro", telegram_token="tg-token", max_token=""),
        transport=transport,
    )

    profiles = client.list_profiles("telegram")

    assert profiles == [{"id": "profile-1", "name": "Foton TG", "profile_id": "profile-1", "channel": "telegram"}]
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://wappi.pro/tapi/profile/all/get"
    assert calls[0]["headers"]["Authorization"] == "tg-token"


def test_amo_client_reads_pipelines_leads_and_contacts() -> None:
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    client = AmoPhase1Client(
        AmoClientConfig(base_url="https://educent.amocrm.ru", access_token="token"),
        transport=transport,
    )

    client.list_pipelines()
    client.get_lead("123")
    client.list_contacts(query="+79991234567")
    client.get_contact(456)

    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://educent.amocrm.ru/api/v4/leads/pipelines?with=statuses"
    assert calls[1]["url"] == "https://educent.amocrm.ru/api/v4/leads/123?with=contacts"
    assert calls[2]["url"].startswith("https://educent.amocrm.ru/api/v4/contacts?")
    assert "query=%2B79991234567" in calls[2]["url"]
    assert calls[3]["url"] == "https://educent.amocrm.ru/api/v4/contacts/456?with=leads"


def test_draft_note_write_allowed_only_for_test_lead() -> None:
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    config = AmoWappiPhase1Config(
        profile_brand_map={"profile-1": "foton"},
        allowed_test_lead_ids=frozenset({"123"}),
    )
    client = AmoPhase1Client(
        AmoClientConfig(base_url="https://educent.amocrm.ru", access_token="token"),
        transport=transport,
    )
    created_at = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)

    response = client.add_draft_note_to_test_lead(
        "123",
        config=config,
        draft_text="Подскажите, пожалуйста, удобное время для звонка.",
        brand="foton",
        profile_id="profile-1",
        created_at=created_at,
    )

    assert response == {"status": "ok"}
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://educent.amocrm.ru/api/v4/leads/123/notes"
    note_text = calls[0]["json_body"][0]["params"]["text"]
    assert DRAFT_NOTE_MARKER in note_text
    assert "Бренд: foton" in note_text
    assert "Wappi profile_id: profile-1" in note_text
    assert "Подскажите, пожалуйста" in note_text


def test_draft_note_write_outside_allowlist_is_blocked_before_http() -> None:
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"status": "should-not-happen"}

    config = AmoWappiPhase1Config(allowed_test_lead_ids=frozenset({"123"}))
    client = AmoPhase1Client(
        AmoClientConfig(base_url="https://educent.amocrm.ru", access_token="token"),
        transport=transport,
    )

    with pytest.raises(AmoWappiWriteBlocked):
        client.add_draft_note_to_test_lead(
            "999",
            config=config,
            draft_text="Черновик",
            brand="foton",
        )

    assert calls == []


def test_draft_note_text_requires_known_brand() -> None:
    with pytest.raises(AmoWappiConfigError):
        build_draft_note_text(draft_text="Черновик", brand="other")


def test_manager_edit_log_keeps_bot_and_manager_texts_side_by_side(tmp_path: Path) -> None:
    log_path = tmp_path / "manager_edits.jsonl"
    append_manager_edit_log(
        log_path,
        ManagerEditLogRecord(
            lead_id="123",
            brand="unpk",
            profile_id="profile-unpk",
            bot_draft_text="Черновик бота",
            manager_sent_text="Финальный текст менеджера",
            reason_codes=("edited_tone",),
            created_at="2026-06-09T12:00:00+00:00",
        ),
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "schema_version": "amo_wappi_manager_edit_log_v1_2026_06_09",
            "created_at": "2026-06-09T12:00:00+00:00",
            "lead_id": "123",
            "brand": "unpk",
            "profile_id": "profile-unpk",
            "bot_draft_text": "Черновик бота",
            "manager_sent_text": "Финальный текст менеджера",
            "reason_codes": ["edited_tone"],
        }
    ]
