from __future__ import annotations

import json
from pathlib import Path

from scripts.run_telegram_public_pilot_bots import (
    apply_public_autonomy_kill_switch,
    BrandBotConfig,
    build_server_amo_context_readonly,
    ChatSession,
    context_flags_for_report,
    build_timeline_hint_from_local_context,
    configs_from_env,
    debug_client_label,
    debug_customer_summary,
    known_client_fields_for_session,
    known_dialog_fields_from_messages,
    load_debug_clients,
    knowledge_base_version_for_store,
    normalize_phone,
    parse_debug_phone_command,
    public_reply_text,
    PublicPilotBotRuntime,
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


def test_known_dialog_fields_extract_grade_subject_and_format() -> None:
    fields = known_dialog_fields_from_messages(
        ["Клиент: 9 класс, предмет физика", "Клиент: онлайн подходит"],
        active_brand="unpk",
    )

    assert fields["grade"] == "9"
    assert fields["subject"] == "физика"
    assert fields["format"] == "онлайн"
    assert fields["active_brand"] == "unpk"


def test_known_dialog_fields_ignore_bot_answers_to_avoid_self_pollution() -> None:
    fields = known_dialog_fields_from_messages(
        [
            "Клиент: 9 класс, предмет физика",
            "Ответ: Вы писали, что онлайн подходит и важен формат 1 на 1.",
        ],
        active_brand="unpk",
    )

    assert fields["grade"] == "9"
    assert fields["subject"] == "физика"
    assert "format" not in fields


def test_known_client_fields_include_debug_client_and_phone() -> None:
    session = ChatSession(
        debug_phone="79092009933",
        debug_client={
            "student_name": "Колосов Даниил Максимович",
            "parent_name": "Ананьевская Анна Георгиевна",
        },
    )

    fields = known_client_fields_for_session(session=session, crm_context={"amo_context": {"status": "ok"}})

    assert fields["phone"] == "79092009933"
    assert fields["student_name"] == "Колосов Даниил Максимович"
    assert fields["parent_name"] == "Ананьевская Анна Георгиевна"
    assert fields["amo_context"] == "found"


def test_context_flags_for_report_is_compact() -> None:
    flags = context_flags_for_report(
        {
            "read_only_customer_context": {"summary": "ok"},
            "known_client_fields": {"student_name": "Даниил"},
            "context_quality": {"multiple_students": True},
        }
    )

    assert flags["crm_context"] is True
    assert flags["known_client_fields"] is True
    assert flags["multiple_students"] is True


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
        "MANGO_TELEGRAM_PILOT_STORE_PATH": str(tmp_path / "pilot.sqlite"),
        "MANGO_TELEGRAM_PILOT_STORE_ENABLED": "1",
        "TELEGRAM_PILOT_AUTONOMY_ENABLED": "0",
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
    assert all(config.store_path == tmp_path / "pilot.sqlite" for config in configs)
    assert all(config.store_enabled is True for config in configs)
    assert all(config.autonomy_enabled is False for config in configs)


def test_public_reply_text_strips_internal_markers() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="[source=source:local_docx:test; freshness=unknown] Здравствуйте! Подскажите класс ребёнка.",
    )

    assert public_reply_text(result) == "Здравствуйте! Подскажите класс ребёнка."


def test_public_reply_text_falls_back_when_empty() -> None:
    result = SubscriptionDraftResult(route="manager_only", draft_text="")

    assert "Передам вопрос менеджеру" in public_reply_text(result)


def test_autonomy_kill_switch_downgrades_autonomous_route() -> None:
    result = SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, курс есть.")

    downgraded = apply_public_autonomy_kill_switch(result, autonomy_enabled=False)

    assert downgraded.route == "draft_for_manager"
    assert "autonomy_kill_switch_applied" in downgraded.safety_flags
    assert downgraded.metadata["original_route_before_autonomy_kill_switch"] == "bot_answer_self_for_pilot"


def test_runtime_persists_pilot_decision_to_local_store(tmp_path: Path) -> None:
    class FakeMessage:
        message_id = 42
        date = None

    config = BrandBotConfig(
        brand="foton",
        token="token",
        display_name="Фотон",
        snapshot_path=tmp_path / "snapshot.json",
        store_path=tmp_path / "telegram_pilot.sqlite",
        p0_register_path=tmp_path / "p0.csv",
    )
    runtime = PublicPilotBotRuntime(config, debug_clients={})
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, в Фотоне есть курс.",
        topic_id="theme:016_program",
        safety_flags=("client_safe_fact_verified",),
    )

    runtime.persist_pilot_decision(
        message=FakeMessage(),
        chat_id=123,
        input_text="Есть курс?",
        answer_text="Да, в Фотоне есть курс.",
        context={"knowledge_base_version": "kb-test", "active_brand": "foton"},
        result=result,
        latency_seconds=1.2,
        request_started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    summary = runtime.store.summary() if runtime.store is not None else {}
    runtime.close()

    assert summary["messages"] == 1
    assert summary["drafts"] == 1
    assert knowledge_base_version_for_store({"knowledge_base_version": "kb-test"}, tmp_path / "snapshot.json") == "kb-test"


def test_runtime_persists_funnel_and_manager_summary_metadata(tmp_path: Path) -> None:
    class FakeMessage:
        message_id = 43
        date = None

    config = BrandBotConfig(
        brand="unpk",
        token="token",
        display_name="УНПК МФТИ",
        snapshot_path=tmp_path / "snapshot.json",
        store_path=tmp_path / "telegram_pilot.sqlite",
        p0_register_path=tmp_path / "p0.csv",
    )
    runtime = PublicPilotBotRuntime(config, debug_clients={})
    session = ChatSession()
    context = {"knowledge_base_version": "kb-test", "active_brand": "unpk"}
    funnel = runtime.build_funnel_state(
        chat_id=123,
        session=session,
        current_text="9 класс, физика. Когда расписание?",
        context=context,
    )
    context = runtime.attach_funnel_state_to_context(context, funnel)
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Поняла: 9 класс, физика. Расписание проверит менеджер.",
        topic_id="theme:013_schedule",
        risk_level="low",
        missing_facts=("точное расписание",),
    )
    manager_summary = runtime.build_manager_summary(
        input_text="9 класс, физика. Когда расписание?",
        answer_text=result.draft_text,
        result=result,
        funnel_state=funnel,
        context=context,
    )

    runtime.persist_pilot_decision(
        message=FakeMessage(),
        chat_id=123,
        input_text="9 класс, физика. Когда расписание?",
        answer_text=result.draft_text,
        context=context,
        result=result,
        funnel_state=funnel,
        manager_summary=manager_summary,
        latency_seconds=1.4,
        request_started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    drafts = runtime.store.list_drafts() if runtime.store is not None else ()
    runtime.close()

    assert len(drafts) == 1
    metadata = drafts[0]["metadata"]
    assert metadata["lead_stage"] == funnel.lead_stage
    assert metadata["next_step_type"] == funnel.next_step_type
    assert metadata["known_slots"]["grade"] == "9"
    assert "manager_summary" in metadata
    assert "AMO" not in metadata["manager_summary"]
    assert "Tallanto" not in metadata["manager_summary"]
    assert "manager_summary" not in drafts[0]["draft_text"]


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
