from __future__ import annotations

import json
from pathlib import Path

from scripts.run_telegram_public_pilot_bots import (
    BrandBotConfig,
    build_server_amo_context_readonly,
    build_timeline_hint_from_local_context,
    configs_from_env,
    debug_client_label,
    debug_customer_summary,
    load_debug_clients,
    normalize_phone,
    parse_debug_phone_command,
    public_reply_text,
)
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult


def test_parse_debug_phone_command_without_payload() -> None:
    command = parse_debug_phone_command("Представь, что я пишу с номера 79092009933")

    assert command.matched is True
    assert command.phone == "79092009933"
    assert command.rest == ""


def test_parse_debug_phone_command_with_payload() -> None:
    command = parse_debug_phone_command('«Представь, что я пишу с номера +7 (909) 200-99-33»: какая цена?')

    assert command.matched is True
    assert command.phone == "79092009933"
    assert command.rest == "какая цена?"


def test_non_debug_message_not_matched() -> None:
    command = parse_debug_phone_command("Подскажите стоимость обучения")

    assert command.matched is False


def test_normalize_phone_handles_russian_formats() -> None:
    assert normalize_phone("8 (909) 200-99-33") == "79092009933"
    assert normalize_phone("9092009933") == "79092009933"
    assert normalize_phone("+7 909 200 99 33") == "79092009933"


def test_load_debug_clients_from_json() -> None:
    payload = {
        "79092009933": {
            "student_name": "Колосов Даниил Максимович",
            "parent_name": "Ананьевская Анна Георгиевна",
        }
    }
    clients = load_debug_clients({"MANGO_TELEGRAM_DEBUG_CLIENTS_JSON": json.dumps(payload, ensure_ascii=False)})

    assert clients["79092009933"]["student_name"] == "Колосов Даниил Максимович"
    assert debug_client_label(clients["79092009933"]) == (
        "Ананьевская Анна Георгиевна, ученик Колосов Даниил Максимович"
    )


def test_debug_customer_summary_is_explicitly_test_mode() -> None:
    summary = debug_customer_summary(
        "79092009933",
        {
            "student_name": "Колосов Даниил Максимович",
            "parent_name": "Ананьевская Анна Георгиевна",
        },
    )

    assert "Тестовый режим сотрудника" in summary
    assert "79092009933" in summary
    assert "Ананьевская Анна Георгиевна" in summary


def test_configs_from_env_builds_two_brand_isolated_configs(tmp_path: Path) -> None:
    snapshot = tmp_path / "kb_release_v3_snapshot.json"
    env = {
        "MANGO_TELEGRAM_FOTON_BOT_TOKEN": "foton-token",
        "MANGO_TELEGRAM_UNPK_BOT_TOKEN": "unpk-token",
        "MANGO_TELEGRAM_KB_SNAPSHOT": str(snapshot),
        "MANGO_TELEGRAM_CRM_READ_MODE": "server",
        "MANGO_TELEGRAM_CRM_ENV_FILE": str(tmp_path / ".env.private"),
        "MANGO_CRM_SERVER_URL": "https://api.fotonai.online",
        "MANGO_CRM_SERVER_API_KEY": "test-key",
    }

    configs = configs_from_env(env, brand="all")

    assert [config.brand for config in configs] == ["foton", "unpk"]
    assert [config.display_name for config in configs] == ["Фотон", "УНПК МФТИ"]
    assert all(isinstance(config, BrandBotConfig) for config in configs)
    assert all(config.snapshot_path == snapshot for config in configs)
    assert all(config.crm_read_mode == "server" for config in configs)
    assert all(config.crm_env_file == tmp_path / ".env.private" for config in configs)
    assert all(config.crm_server_url == "https://api.fotonai.online" for config in configs)
    assert all(config.crm_server_api_key == "test-key" for config in configs)


def test_public_reply_text_strips_internal_markers() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="[source=source:local_docx:test; freshness=unknown] Здравствуйте! Подскажите класс ребёнка.",
    )

    assert public_reply_text(result) == "Здравствуйте! Подскажите класс ребёнка."


def test_public_reply_text_falls_back_when_empty() -> None:
    result = SubscriptionDraftResult(route="manager_only", draft_text="")

    assert "Передам вопрос менеджеру" in public_reply_text(result)


def test_timeline_hint_from_local_context_is_read_only() -> None:
    context = build_timeline_hint_from_local_context(
        {
            "status": "ok",
            "history_summary": "Клиент интересовался курсом математики.",
            "last_call_at": "2026-05-20",
            "call_count": 3,
        }
    )

    assert context["found"] is True
    assert context["read_only"] is True
    assert context["call_count"] == 3


def test_crm_mode_rejects_unknown_value() -> None:
    try:
        BrandBotConfig(
            brand="foton",
            token="token",
            display_name="Фотон",
            snapshot_path=Path("snapshot.json"),
            crm_read_mode="write",
        )
    except ValueError as exc:
        assert "crm_read_mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_server_amo_context_compacts_read_only_payload(monkeypatch) -> None:
    def fake_server_json_request(**kwargs):
        assert kwargs["path"] == "/api/integrations/amocrm/leads/by-phone"
        return {
            "status": "matched",
            "contact_count": 1,
            "lead_count": 1,
            "contacts": [{"id": 1, "name": "Тестовый контакт", "custom_fields_values": [{"secret": "not exposed"}]}],
            "leads": [{"id": 2, "name": "Тестовая сделка", "price": 1000, "custom_fields_values": [{"secret": "not exposed"}]}],
        }

    monkeypatch.setattr("scripts.run_telegram_public_pilot_bots.server_json_request", fake_server_json_request)

    context = build_server_amo_context_readonly(
        "79092009933",
        server_url="https://api.fotonai.online",
        api_key="test-key",
    )

    assert context["source"] == "amo_server"
    assert context["read_only"] is True
    assert context["contacts_found"] == 1
    assert context["leads_found"] == 1
    assert "custom_fields_values" not in context["contacts"][0]
