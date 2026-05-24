from __future__ import annotations

from mango_mvp.channels.telegram_pilot_context_builder import (
    NO_KNOWLEDGE_SNAPSHOT_VERSION,
    build_telegram_pilot_context,
)


def test_builder_wires_fresh_snapshot_into_pilot_context() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_night_20260517_v1",
        "sources": [
            {
                "source_id": "source:price_2026",
                "title": "Стоимость 2026/2027",
                "fact_types": ["price"],
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
            }
        ],
        "facts": [
            {
                "fact_id": "fact:price_grade_10",
                "fact_type": "price",
                "client_safe_text": "Стоимость курса для 10 класса: 120 000 рублей.",
                "source_id": "source:price_2026",
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk:price_10",
                "source_id": "source:price_2026",
                "title": "Стоимость обучения",
                "text": "Стоимость курса для 10 класса: 120 000 рублей. Источник проверен на 2026/2027 год.",
                "fact_types": ["price"],
                "freshness_status": "fresh_verified",
            }
        ],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит курс для 10 класса?",
        theme={"topic_id": "theme:001_pricing", "topic_name": "Стоимость"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["knowledge_base_version"] == "kb_night_20260517_v1"
    assert payload["facts_context"]["knowledge_base_version"] == "kb_night_20260517_v1"
    assert payload["facts_context"]["fresh"] is True
    assert payload["facts_context"]["client_safe_fact_verified"] is True
    assert payload["rop_policy"]["autonomy_policy"]["allow_autonomous"] is True
    assert payload["facts_context"]["facts_missing"] is False
    assert payload["confirmed_facts"]["fact:price_grade_10"] == "Стоимость курса для 10 класса: 120 000 рублей."
    assert payload["knowledge_snippets"]
    assert "Стоимость обучения" in payload["knowledge_snippets"][0]
    assert "source=" not in payload["knowledge_snippets"][0]
    assert "freshness=" not in payload["knowledge_snippets"][0]
    assert "missing_facts" not in payload
    assert payload["answer_contract"]["schema_version"] == "answer_contract_v2_2026_05_24"
    assert payload["answer_contract"]["required_fact_ids"] == ["fact:price_grade_10"]
    assert payload["answer_contract"]["facts_resolved_by_intent"] == ["fact:price_grade_10"]
    assert payload["answer_contract"]["route"]
    assert payload["answer_contract"]["route_reason"]


def test_builder_uses_known_slots_to_retrieve_followup_price_fact() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_followup_price_test",
        "facts": [
            {
                "fact_id": "fact:foton_offline_8_year",
                "fact_type": "price",
                "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
                "brand": "foton",
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Это цена сейчас? Можно зафиксировать годовую?",
        active_brand="foton",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
        known_slots={"grade": "8", "subject": "информатика", "format": "очно"},
    )
    payload = context.to_prompt_context()

    assert payload["facts_context"]["client_safe_fact_verified"] is True
    assert payload["confirmed_facts"]["fact:foton_offline_8_year"].endswith("74 500 ₽.")
    assert payload["dialogue_memory_view"]["known_slots"]["grade"] == "8"
    assert payload["dialogue_memory_view"]["known_slots"]["subject"] == "информатика"
    assert payload["dialogue_memory_view"]["open_question"]["kind"] == "price_fix"


def test_builder_dialogue_memory_uses_recent_client_context_without_reasking() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_dialogue_memory_test",
        "facts": [
            {
                "fact_id": "fact:foton_online_price",
                "fact_type": "price",
                "client_safe_text": "Фотон: 5-11 класс, онлайн, год — 47 250 ₽.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "А это цена на сейчас?",
        active_brand="foton",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
        recent_messages=["Клиент: 8 класс физика онлайн", "Ответ: Сейчас сориентирую."],
        session_id="ctx-memory",
    )
    payload = context.to_prompt_context()

    memory = payload["dialogue_memory_view"]
    assert memory["session_id"] == "ctx-memory"
    assert memory["known_slots"]["grade"] == "8"
    assert memory["known_slots"]["subject"] == "физика"
    assert memory["known_slots"]["format"] == "онлайн"
    assert "grade" in memory["do_not_ask_again"]


def test_builder_adds_conversation_intent_plan_to_prompt_context() -> None:
    context = build_telegram_pilot_context(
        "Можно закрепить место на ЛВШ?",
        active_brand="foton",
        recent_messages=["Клиент: интересует лагерь в Менделеево для 8 класса"],
        kc_snapshot={"schema_version": "kc_knowledge_snapshot_v1", "run_id": "empty", "facts": [], "chunks": []},
    )
    payload = context.to_prompt_context()

    plan = payload["conversation_intent_plan"]
    assert plan["primary_intent"] == "live_availability"
    assert plan["topic_id"] == "theme:026_camp_general"
    assert plan["answer_policy"] == "answer_safe_parts_then_manager_live_check"
    assert "availability.current" in plan["required_fact_keys"]


def test_builder_uses_intent_plan_to_retrieve_foton_online_price_not_offline_or_camp() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_foton_online_context_test",
        "facts": [
            {
                "fact_id": "fact:foton_online_semester",
                "fact_type": "price",
                "client_safe_text": "Фотон: 5-11 класс, онлайн, семестр — 29 750 ₽.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            },
            {
                "fact_id": "fact:foton_online_year",
                "fact_type": "price",
                "client_safe_text": "Фотон: 5-11 класс, онлайн, год — 47 250 ₽.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            },
            {
                "fact_id": "fact:foton_offline_year",
                "fact_type": "price",
                "client_safe_text": "Фотон: 5-11 класс, очно, год — 74 500 ₽.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            },
            {
                "fact_id": "fact:foton_lvsh_price",
                "fact_type": "price",
                "client_safe_text": "Фотон: ЛВШ Менделеево сейчас стоит 93 100 ₽.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:026_camp_general"],
            },
            {
                "fact_id": "fact:unpk_price",
                "fact_type": "price",
                "client_safe_text": "УНПК: очный курс стоит 82 000 ₽.",
                "brand": "unpk",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            },
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит?",
        active_brand="foton",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
        recent_messages=["Клиент: 8 класс физика онлайн"],
        known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
    )
    payload = context.to_prompt_context()
    facts = " ".join(payload["confirmed_facts"].values())

    assert "29 750" in facts
    assert "47 250" in facts
    assert "74 500" not in facts
    assert "93 100" not in facts
    assert "82 000" not in facts
    assert payload["conversation_intent_plan"]["primary_intent"] == "pricing"
    assert payload["conversation_intent_plan"]["known_slots"]["format"] == "онлайн"


def test_builder_filters_snapshot_by_active_brand() -> None:
    snapshot = {
        "schema_version": "kb_release_v2_snapshot_2026_05_17",
        "run_id": "kb_release_v2_test",
        "facts": [
            {
                "fact_id": "fact:foton_installment",
                "fact_type": "installment",
                "client_safe_text": "Для Фотона можно обсудить рассрочку.",
                "brand": "foton",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:006_installment"],
            },
            {
                "fact_id": "fact:unpk_installment",
                "fact_type": "installment",
                "client_safe_text": "Для УНПК МФТИ другой порядок оплаты.",
                "brand": "unpk",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:006_installment"],
            },
        ],
        "chunks": [
            {
                "chunk_id": "chunk:foton",
                "source_id": "source:foton",
                "title": "Фотон рассрочка",
                "text": "Для Фотона можно обсудить рассрочку.",
                "fact_types": ["installment"],
                "freshness_status": "document_verified",
                "brand": "foton",
            },
            {
                "chunk_id": "chunk:unpk",
                "source_id": "source:unpk",
                "title": "УНПК рассрочка",
                "text": "Для УНПК МФТИ другой порядок оплаты.",
                "fact_types": ["installment"],
                "freshness_status": "document_verified",
                "brand": "unpk",
            },
        ],
    }

    context = build_telegram_pilot_context(
        "Можно оплатить частями?",
        active_brand="foton",
        theme={"topic_id": "theme:006_installment"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["installment_terms.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["active_brand"] == "foton"
    assert list(payload["confirmed_facts"]) == ["fact:foton_installment"]
    assert "Фотона" in payload["knowledge_snippets"][0]
    assert "УНПК" not in " ".join(payload["knowledge_snippets"])


def test_builder_adds_related_discount_fact_for_installment_query() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test_installment_discount",
        "facts": [
            {
                "fact_id": "fact:unpk_installment_base",
                "fact_type": "installment",
                "client_safe_text": "В УНПК можно платить помесячно, за семестр или за год.",
                "brand": "unpk",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:006_installment"],
            },
            {
                "fact_id": "fact:unpk_year_discount",
                "fact_type": "course_parameter",
                "client_safe_text": "При оплате за семестр действует скидка 10%, за год — 14%.",
                "brand": "unpk",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:006_installment"],
            },
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Можно оплатить помесячно или за год?",
        active_brand="unpk",
        theme={"topic_id": "theme:006_installment"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["installment_terms.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert "fact:unpk_installment_base" in payload["confirmed_facts"]
    assert "fact:unpk_year_discount" in payload["confirmed_facts"]
    assert "14%" in " ".join(payload["confirmed_facts"].values())


def test_builder_uses_safe_fallback_when_snapshot_missing() -> None:
    context = build_telegram_pilot_context(
        "Какая цена?",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=None,
    )
    payload = context.to_prompt_context()

    assert payload["knowledge_base_version"] == NO_KNOWLEDGE_SNAPSHOT_VERSION
    assert payload["facts_context"]["snapshot_found"] is False
    assert payload["facts_context"]["fresh"] is False
    assert payload["facts_context"]["facts_missing"] is True
    assert payload["missing_facts"] == ["prices.current"]
    assert "knowledge_snapshot_missing" in payload["context_warnings"]
    assert "precise_answer_blocked" in payload["context_warnings"]
    assert "knowledge_snippets" not in payload


def test_builder_keeps_metadata_only_price_as_missing_fact() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_night_20260517_v1",
        "sources": [
            {
                "source_id": "source:price_metadata_only",
                "title": "Стоимость 2026/2027",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
                "usable_for_precise_answer": False,
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk:price_exact_metadata_only",
                "source_id": "source:price_metadata_only",
                "title": "Стоимость обучения",
                "text": "Стоимость курса 120 000 рублей, но документ не прочитан и не подтвержден.",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
            },
            {
                "chunk_id": "chunk:price_warning",
                "source_id": "source:price_metadata_only",
                "title": "Стоимость обучения требует проверки",
                "text": "Документ по стоимости зарегистрирован, но точные цены требуют проверки менеджером.",
                "fact_types": ["price"],
                "freshness_status": "metadata_only",
            }
        ],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит обучение?",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["facts_context"]["snapshot_found"] is True
    assert payload["facts_context"]["fresh"] is False
    assert payload["facts_context"]["facts_missing"] is True
    assert payload["missing_facts"] == ["prices.current"]
    assert "facts_stale" in payload["context_warnings"]
    assert "Стоимость обучения требует проверки" in payload["knowledge_snippets"][0]
    assert "120 000" not in " ".join(payload["knowledge_snippets"])
    assert "confirmed_facts" not in payload


def test_builder_does_not_mark_usable_fact_without_client_permission_as_verified() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test",
        "facts": [
            {
                "fact_id": "fact:internal_price",
                "fact_type": "price",
                "client_safe_text": "Стоимость курса: 120 000 рублей.",
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": False,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "related_theme_ids": ["theme:001_pricing"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит курс?",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert "confirmed_facts" not in payload
    assert payload["facts_context"]["client_safe_fact_verified"] is False
    assert payload["facts_context"]["facts_missing"] is True
    assert payload["rop_policy"]["autonomy_policy"]["allow_autonomous"] is False


def test_builder_prefers_matching_class_and_format_price_fact() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test",
        "facts": [
            {
                "fact_id": "fact:online_1_4",
                "fact_type": "price",
                "client_safe_text": "Фотон: 1-4 класс, онлайн, семестр — 19 000 ₽.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:001_pricing"],
            },
            {
                "fact_id": "fact:offline_5_11",
                "fact_type": "price",
                "client_safe_text": "Фотон: 5-11 класс, очно, семестр — 44 600 ₽.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:001_pricing"],
            },
            {
                "fact_id": "fact:camp_price",
                "fact_type": "price",
                "client_safe_text": "Фотон: городской летний лагерь, базовый вариант — 34 300 ₽.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:001_pricing"],
            },
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит очный курс для 5 класса?",
        active_brand="foton",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert list(payload["confirmed_facts"])[0] == "fact:offline_5_11"
    assert "44 600" in next(iter(payload["confirmed_facts"].values()))
    assert "fact:camp_price" not in payload["confirmed_facts"]


def test_builder_does_not_use_lvsh_price_for_online_course_price_gap() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test_lvsh_scope",
        "facts": [
            {
                "fact_id": "fact:unpk_lvsh_price",
                "fact_type": "price",
                "client_safe_text": "УНПК: ЛВШ Менделеево, текущая цена — 114 000 ₽.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "unpk",
                "related_theme_ids": ["theme:001_pricing", "theme:026_camp_general"],
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk:unpk_lvsh_price",
                "title": "ЛВШ Менделеево",
                "text": "УНПК: ЛВШ Менделеево, текущая цена — 114 000 ₽.",
                "fact_types": ["price"],
                "freshness_status": "document_verified",
                "brand": "unpk",
            }
        ],
    }

    context = build_telegram_pilot_context(
        "Сколько стоит онлайн 5 класс математика?",
        active_brand="unpk",
        theme={"topic_id": "theme:001_pricing"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["prices.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert "confirmed_facts" not in payload
    assert payload["facts_context"]["client_safe_fact_verified"] is False
    assert "prices.current" in payload["missing_facts"]
    combined_context = " ".join(payload.get("knowledge_snippets", ()))
    assert "ЛВШ" not in combined_context
    assert "Менделеево" not in combined_context
    assert "114 000" not in combined_context


def test_builder_treats_outbound_camp_as_lvsh_context() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test_outbound_camp_scope",
        "facts": [
            {
                "fact_id": "fact:foton_lvsh_living",
                "fact_type": "camp_lvsh",
                "client_safe_text": "Фотон: ЛВШ Менделеево — проживание, 5-разовое питание и трансфер из Москвы.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:027_camp_living_conditions"],
            },
            {
                "fact_id": "fact:foton_city_camp",
                "fact_type": "camp_city",
                "client_safe_text": "Фотон: городской летний лагерь — без проживания.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:027_camp_living_conditions"],
            },
        ],
        "chunks": [
            {
                "chunk_id": "chunk:foton_lvsh_living",
                "title": "ЛВШ Менделеево",
                "text": "Фотон: ЛВШ Менделеево — проживание, 5-разовое питание и трансфер из Москвы.",
                "fact_types": ["camp_lvsh"],
                "freshness_status": "document_verified",
                "brand": "foton",
            },
            {
                "chunk_id": "chunk:foton_city_camp",
                "title": "Городской лагерь",
                "text": "Фотон: городской летний лагерь — без проживания.",
                "fact_types": ["camp_city"],
                "freshness_status": "document_verified",
                "brand": "foton",
            },
        ],
    }

    context = build_telegram_pilot_context(
        "Что входит в выездной лагерь: проживание, питание и трансфер?",
        active_brand="foton",
        theme={"topic_id": "theme:027_camp_living_conditions"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["programs.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    combined_facts = " ".join(payload["confirmed_facts"].values())
    combined_snippets = " ".join(payload.get("knowledge_snippets", ()))
    assert "ЛВШ Менделеево" in combined_facts
    assert "трансфер" in combined_snippets
    assert "городской летний лагерь" not in combined_facts.casefold()


def test_builder_uses_recent_lvsh_context_for_transport_followup() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test_transport_followup_scope",
        "facts": [
            {
                "fact_id": "fact:foton_lvsh_transfer",
                "fact_type": "camp_lvsh",
                "client_safe_text": "Трансфер до ЛВШ Фотона включён в стоимость; ориентир места сбора — метро Ховрино.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "foton",
                "related_theme_ids": ["theme:028_transport_logistics"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "А трансфер из Москвы есть?",
        active_brand="foton",
        theme={"topic_id": "theme:028_transport_logistics"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["transport.current"]},
        kc_snapshot=snapshot,
        recent_messages=[
            "Клиент: расскажите про выездной лагерь",
            "Бот: ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽.",
        ],
    )
    payload = context.to_prompt_context()

    assert payload["facts_context"]["client_safe_fact_verified"] is True
    assert "fact:foton_lvsh_transfer" in payload["confirmed_facts"]
    assert "Ховрино" in next(iter(payload["confirmed_facts"].values()))


def test_builder_prefers_deadline_fact_for_when_question() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test",
        "facts": [
            {
                "fact_id": "fact:camp_address",
                "fact_type": "location",
                "client_safe_text": "УНПК: адрес ЛВШ Менделеево.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "unpk",
                "related_theme_ids": ["theme:026_camp_general"],
            },
            {
                "fact_id": "fact:camp_dates",
                "fact_type": "deadline",
                "client_safe_text": "УНПК: ЛВШ Менделеево проходит 18-26 июля.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "unpk",
                "related_theme_ids": ["theme:013_schedule"],
            },
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Когда проходит ЛВШ Менделеево?",
        active_brand="unpk",
        theme={"topic_id": "theme:013_schedule"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["schedule.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert list(payload["confirmed_facts"])[0] == "fact:camp_dates"
    assert "18-26 июля" in next(iter(payload["confirmed_facts"].values()))


def test_builder_does_not_use_unrelated_city_camp_dates_for_lvsh_when_question() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test",
        "facts": [
            {
                "fact_id": "fact:city_camp_dates",
                "fact_type": "deadline",
                "client_safe_text": "УНПК: городской летний лагерь, Долгопрудный — 20-31 июля.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "unpk",
                "related_theme_ids": ["theme:013_schedule"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Когда проходит ЛВШ Менделеево?",
        active_brand="unpk",
        theme={"topic_id": "theme:013_schedule"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["schedule.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert "confirmed_facts" not in payload
    assert payload["facts_context"]["client_safe_fact_verified"] is False
    assert payload["missing_facts"] == ["schedule.current"]


def test_builder_uses_waitlist_fact_for_zvsh_when_dates_unknown() -> None:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "kb_test",
        "facts": [
            {
                "fact_id": "fact:zvsh_waitlist",
                "fact_type": "deadline",
                "client_safe_text": "УНПК: даты ЗВШ Менделеево пока не определены; можно записаться в лист ожидания.",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": True,
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": False,
                "forbidden_for_client": False,
                "brand": "unpk",
                "related_theme_ids": ["theme:013_schedule"],
            }
        ],
        "chunks": [],
    }

    context = build_telegram_pilot_context(
        "Когда будет ЗВШ Менделеево?",
        active_brand="unpk",
        theme={"topic_id": "theme:013_schedule"},
        rop_policy={"bot_permission": "allowed_after_fact_check", "required_fact_keys": ["schedule.current"]},
        kc_snapshot=snapshot,
    )
    payload = context.to_prompt_context()

    assert payload["confirmed_facts"] == {
        "fact:zvsh_waitlist": "УНПК: даты ЗВШ Менделеево пока не определены; можно записаться в лист ожидания."
    }
    assert payload["facts_context"]["client_safe_fact_verified"] is True
