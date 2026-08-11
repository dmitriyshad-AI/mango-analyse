from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.channels.dialogue_memory import (
    DialogueMemory,
    DialogueSlot,
    DialogueTurn,
    SLOTS_GSF_KNOWN_MERGE_ENV,
    SLOTS_REASK_ENV,
    build_dialogue_memory,
    dialogue_memory_from_mapping,
    safe_next_action,
    update_dialogue_memory_after_answer,
)
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt, _direct_path_prompt_known_slots
from mango_mvp.channels.subscription_llm_parts.support import DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SEMANTIC_READING_CLASSES_ENV,
    SEMANTIC_READING_SLOT_SOURCE,
    SemanticReading,
    enabled_classes,
    off_topic_reading_decision,
    reading_class_enabled,
    sense_seats_reading_decision,
    slot_candidates_from_reading,
    slots_reading_candidates,
)


def test_semantic_reading_classes_are_default_off(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    monkeypatch.delenv(DIRECT_PATH_PILOT_CONFIG_ENV, raising=False)

    assert enabled_classes({}) == frozenset()
    assert reading_class_enabled({}, "off_topic") is False

    context = {
        SEMANTIC_READING_CLASSES_ENV: (
            "off_topic,slots_gsf,intent_actions,route_templates,rewrite_quality,post_semantics,"
            "live_status_read,reask_read,roles_read,fact_select_read,unknown"
        )
    }
    assert enabled_classes(context) == frozenset(
        {
            "slots_gsf",
            "live_status_read",
            "fact_select_read",
        }
    )
    assert reading_class_enabled(context, "off_topic") is False
    assert reading_class_enabled(context, "sense_seats") is False
    assert reading_class_enabled(context, "intent_actions") is False
    assert reading_class_enabled(context, "route_templates") is False
    assert reading_class_enabled(context, "rewrite_quality") is False
    assert reading_class_enabled(context, "post_semantics") is False
    assert reading_class_enabled(context, "live_status_read") is True
    assert reading_class_enabled(context, "reask_read") is False
    assert reading_class_enabled(context, "roles_read") is False
    assert reading_class_enabled(context, "fact_select_read") is True


def test_semantic_reading_classes_profile_default_and_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)

    assert enabled_classes({}) == frozenset(
        {
            "sense_seats",
            "slots_gsf",
            "live_status_read",
            "fact_select_read",
        }
    )
    assert reading_class_enabled(None, "slots_gsf") is True
    assert reading_class_enabled(None, "off_topic") is False
    assert reading_class_enabled(None, "intent_actions") is False
    assert reading_class_enabled(None, "route_templates") is False
    assert reading_class_enabled(None, "rewrite_quality") is False
    assert reading_class_enabled(None, "post_semantics") is False
    assert reading_class_enabled(None, "live_status_read") is True
    assert reading_class_enabled(None, "reask_read") is False
    assert reading_class_enabled(None, "roles_read") is False
    assert reading_class_enabled(None, "fact_select_read") is True

    assert enabled_classes({SEMANTIC_READING_CLASSES_ENV: ""}) == frozenset()
    assert reading_class_enabled({SEMANTIC_READING_CLASSES_ENV: ""}, "slots_gsf") is False
    monkeypatch.setenv(SEMANTIC_READING_CLASSES_ENV, "")
    assert enabled_classes({}) == frozenset()
    monkeypatch.setenv(SEMANTIC_READING_CLASSES_ENV, "off_topic")
    assert enabled_classes({}) == frozenset()


def test_semantic_reading_reads_inline_frame_without_exposing_p0_fields() -> None:
    result = SubscriptionDraftResult(
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "off_topic",
                "sense": "other",
                "scope": "crypto",
                "confidence": 0.91,
            },
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "requested_product": {"grade": "9 класс", "subject": "физика", "format": "онлайн"},
                "risk_class": "p0",
                "must_handoff": True,
                "confidence": 0.84,
            },
        }
    )

    reading = SemanticReading.from_result(result)

    assert reading is not None
    assert reading.source == "inline"
    assert reading.primary_intent == "off_topic"
    assert reading.product_grade == "9 класс"
    assert reading.frame_confidence == 0.84
    assert not hasattr(reading, "risk_class")
    assert not hasattr(reading, "must_handoff")


def test_semantic_reading_infers_posthoc_source_from_status() -> None:
    result = SubscriptionDraftResult(
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "requested_action": "answer_question",
                "requested_product": {"subject": "математика"},
                "confidence": 0.9,
            },
        }
    )

    reading = SemanticReading.from_result(result)

    assert reading is not None
    assert reading.source == "posthoc"


def test_slot_candidates_from_reading_are_llm_sourced_and_history_checked() -> None:
    reading = SemanticReading(
        source="inline",
        product_grade="9 класс",
        product_subject="физика",
        product_format="онлайн",
        product_raw_text="9 класс, физика онлайн",
        frame_confidence=0.82,
    )

    slots = slot_candidates_from_reading(reading, history_texts=("Нужна физика онлайн для 9 класса",))

    assert slots["grade"]["value"] == "9"
    assert slots["grade"]["source_name"] == SEMANTIC_READING_SLOT_SOURCE
    assert slots["subject"]["value"] == "физика"
    assert slots["format"]["value"] == "онлайн"
    assert slot_candidates_from_reading(reading, history_texts=("Нужен курс",), confidence_threshold=0.7) == {}
    assert slot_candidates_from_reading(SemanticReading(source="posthoc", product_grade="9", frame_confidence=0.9)) == {}


def test_slot_candidates_reject_grade_from_dates_multi_children_and_transitions() -> None:
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="6-17 июля", frame_confidence=0.94),
            history_texts=("6-17 июля есть места? адрес?",),
        )
        == {}
    )
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="9", product_subject="математика", frame_confidence=0.91),
            history_texts=("У меня двое: 9 и 7 класс, обоим математика онлайн.",),
        ).get("grade")
        is None
    )
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="6", product_subject="физика", frame_confidence=0.91),
            history_texts=("Сын 6-й закончил, что есть по физике на осень?",),
        ).get("grade")
        is None
    )


def test_slot_candidates_accept_grade_near_non_class_price_number() -> None:
    slots = slot_candidates_from_reading(
        SemanticReading(source="inline", product_grade="8 класс", frame_confidence=0.91),
        history_texts=("Клиент: У нас 8 класс, стоимость 9 000 ₽ подходит.",),
    )

    assert slots["grade"]["value"] == "8"
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="9 класс", frame_confidence=0.91),
            history_texts=("Клиент: У нас 8 класс, стоимость 9 000 ₽ подходит.",),
        ).get("grade")
        is None
    )


def test_slot_candidates_accept_current_transition_into_grade_but_not_finished_grade() -> None:
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="6 класс", frame_confidence=0.91),
            history_texts=("Клиент: Ребенок перешел в 6 класс, интересует смена.",),
        ).get("grade", {}).get("value")
        == "6"
    )
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_grade="6 класс", frame_confidence=0.91),
            history_texts=("Клиент: Закончил 6 класс.",),
        ).get("grade")
        is None
    )


def test_slot_candidates_reject_tz147_kb_copied_grade_and_format_when_client_only_p0_context() -> None:
    reading = SemanticReading(
        source="inline",
        product_grade="11 класс",
        product_format="онлайн",
        product_raw_text="УНПК: ЛВШ для 11 класса онлайн",
        frame_confidence=0.93,
    )

    assert (
        slot_candidates_from_reading(
            reading,
            history_texts=("Клиент: Оплатили, а занятие не назначили.", "Клиент: Сегодня уже должно было быть."),
        )
        == {}
    )


def test_slot_candidates_accept_closed_dictionary_values_spoken_by_client() -> None:
    slots = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_grade="1го класса",
            product_subject="программирование",
            product_format="из дома",
            frame_confidence=0.91,
        ),
        history_texts=("Расписание для 1го класса, программирование из дома.",),
    )

    assert slots["grade"]["value"] == "1"
    assert slots["subject"]["value"] == "информатика"
    assert slots["format"]["value"] == "онлайн"


def test_slot_candidates_accept_it_alias_without_broad_substring_match() -> None:
    slots = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_subject="ИТ",
            frame_confidence=0.91,
        ),
        history_texts=("Интересует ИТ для 8 класса.",),
    )

    assert slots["subject"]["value"] == "информатика"
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_subject="гуманитарное направление", frame_confidence=0.91),
            history_texts=("Есть курсы очные для гуманитарного направления?",),
        ).get("subject")
        is None
    )


def test_semantic_reading_slots_hidden_storage_does_not_leak_to_behavior(monkeypatch) -> None:
    monkeypatch.setenv(SEMANTIC_READING_CLASSES_ENV, "slots_gsf")
    memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        turns=(DialogueTurn("client", "Нужна физика онлайн для 9 класса."),),
    )
    semantic_reading = SemanticReading(
        source="inline",
        product_grade="9 класс",
        product_subject="физика",
        product_format="онлайн",
        product_raw_text="9 класс, физика онлайн",
        frame_confidence=0.91,
    ).to_memory_dict()

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=semantic_reading,
    )
    payload = updated.to_json_dict()
    roundtrip = dialogue_memory_from_mapping(payload)

    assert payload["semantic_reading_slots"]["grade"]["value"] == "9"
    assert roundtrip.semantic_reading_slots["subject"]["value"] == "физика"
    assert "semantic_reading_slots" not in updated.to_prompt_view()
    assert updated.known_slots == {}
    assert updated.client_confirmed_slots == {}
    assert updated.do_not_reask_slots == ()
    assert updated.topic_focus == {}
    assert safe_next_action(updated) == {}
    assert "grade" not in _direct_path_prompt_known_slots({"dialogue_memory_view": updated.to_prompt_view()})


def test_semantic_reading_slots_profile_default_writes_hidden_storage(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)

    updated = update_dialogue_memory_after_answer(
        DialogueMemory(
            session_id="s1",
            active_brand="foton",
            turns=(DialogueTurn("client", "Нужна физика онлайн для 9 класса."),),
        ),
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=SemanticReading(
            source="inline",
            product_grade="9 класс",
            product_subject="физика",
            frame_confidence=0.91,
        ).to_memory_dict(),
    )

    assert updated.semantic_reading_slots["grade"]["value"] == "9"
    assert updated.semantic_reading_slots["subject"]["value"] == "физика"


def test_semantic_reading_slots_are_not_written_when_mask_off(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    updated = update_dialogue_memory_after_answer(
        DialogueMemory(
            session_id="s1",
            active_brand="foton",
            turns=(DialogueTurn("client", "Хочу подобрать занятия."),),
        ),
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=SemanticReading(
            source="inline",
            product_grade="9 класс",
            frame_confidence=0.91,
        ).to_memory_dict(),
    )

    assert updated.semantic_reading_slots == {}
    assert "semantic_reading_slots" not in updated.to_json_dict()


def test_slots_reask_does_not_create_hidden_slots_without_slots_gsf(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    followup = build_dialogue_memory(
        current_message="А сколько стоит?",
        active_brand="foton",
        previous_memory={"session_id": "s1", "active_brand": "foton"},
    )

    assert followup.do_not_reask_slots == ()


def test_slots_reask_with_semantic_payload_is_noop_when_slots_gsf_is_off(monkeypatch) -> None:
    monkeypatch.setenv(SEMANTIC_READING_CLASSES_ENV, "")
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    neutral_memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        turns=(DialogueTurn("client", "Хочу подобрать занятия."),),
    )
    updated = update_dialogue_memory_after_answer(
        neutral_memory,
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=SemanticReading(
            source="inline",
            product_grade="9 класс",
            product_subject="физика",
            product_format="онлайн",
            frame_confidence=0.91,
        ).to_memory_dict(),
    )
    followup = build_dialogue_memory(
        current_message="А расписание есть?",
        active_brand="foton",
        previous_memory=updated.to_json_dict(),
    )

    assert updated.semantic_reading_slots == {}
    assert "semantic_reading_slots" not in updated.to_json_dict()
    assert followup.do_not_reask_slots == ()


def test_slots_reask_reads_previous_hidden_slot_names_without_prompt_value_leak(monkeypatch) -> None:
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    previous = {
        "session_id": "s1",
        "active_brand": "foton",
        "semantic_reading_slots": {
            "grade": {"value": "9#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
            "subject": {"value": "химия#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
            "format": {"value": "онлайн#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
        },
    }
    followup = build_dialogue_memory(
        current_message="А расписание есть?",
        active_brand="foton",
        previous_memory=previous,
    )
    prompt_view_text = json.dumps(followup.to_prompt_view(), ensure_ascii=False, sort_keys=True)

    assert set(followup.do_not_reask_slots) >= {"grade", "subject", "format"}
    assert "semantic_reading_slots" not in followup.to_prompt_view()
    assert followup.to_prompt_view()["known_slots"] == {}
    assert followup.to_prompt_view()["client_confirmed_slots"] == {}
    assert "#SENTINEL#" not in prompt_view_text


def test_semantic_hidden_slots_do_not_enter_direct_path_prompt_as_confirmed_values(monkeypatch) -> None:
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    previous = {
        "session_id": "s1",
        "active_brand": "foton",
        "semantic_reading_slots": {
            "grade": {"value": "9#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
            "subject": {"value": "физика#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
        },
    }
    followup = build_dialogue_memory(
        current_message="А сколько стоит?",
        active_brand="foton",
        previous_memory=previous,
    )

    prompt = _build_direct_path_prompt(
        "А сколько стоит?",
        context={"active_brand": "foton", "dialogue_memory_view": followup.to_prompt_view()},
        facts={},
    )

    assert "#SENTINEL#" not in prompt
    assert "semantic_reading_slots" not in prompt
    assert '"client_confirmed_slots": {}' in prompt
    assert '"known_slots": {}' in prompt
    assert "grade" not in _direct_path_prompt_known_slots({"dialogue_memory_view": followup.to_prompt_view()})


def test_slots_gsf_known_merge_is_default_off(monkeypatch) -> None:
    monkeypatch.delenv(SLOTS_GSF_KNOWN_MERGE_ENV, raising=False)
    memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        semantic_reading_slots={
            "grade": {"value": "9", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
            "subject": {"value": "физика", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
        },
    )

    view = memory.to_prompt_view()

    assert view["known_slots"] == {}
    assert view["semantic_inferred_slots"] == {}
    assert view["slots_merge_trace"] == []


def test_slots_gsf_known_merge_adds_only_empty_gsf_without_confirmation(monkeypatch) -> None:
    monkeypatch.setenv(SLOTS_GSF_KNOWN_MERGE_ENV, "1")
    memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        known_slots={"subject": DialogueSlot("математика", "memory_provenance", 1.0, quote="математика")},
        client_confirmed_slots={"subject": "математика"},
        semantic_reading_slots={
            "grade": {"value": "9", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
            "subject": {"value": "физика", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
            "format": {"value": "онлайн", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.91},
            "child_name": {"value": "Максим", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.99},
        },
    )

    view = memory.to_prompt_view()

    assert view["known_slots"]["grade"] == "9"
    assert view["known_slots"]["subject"] == "математика"
    assert view["known_slots"]["format"] == "онлайн"
    assert "child_name" not in view["known_slots"]
    assert view["slot_sources"]["grade"] == SEMANTIC_READING_SLOT_SOURCE
    assert view["slot_sources"]["format"] == SEMANTIC_READING_SLOT_SOURCE
    assert view["client_confirmed_slots"] == {"subject": "математика"}
    assert "grade" not in view["client_confirmed_slots"]
    assert "format" not in view["client_confirmed_slots"]
    assert view["do_not_ask_again"] == ["subject"]
    assert view["semantic_inferred_slots"]["grade"]["status"] == "llm_inferred_not_client_confirmed"
    assert any(item["slot"] == "subject" and item["status"] == "kept_existing" and item["conflict"] for item in view["slots_merge_trace"])


def test_slots_gsf_known_merge_prompt_marks_llm_slots_as_inferred(monkeypatch) -> None:
    monkeypatch.setenv(SLOTS_GSF_KNOWN_MERGE_ENV, "1")
    memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        semantic_reading_slots={
            "grade": {"value": "9", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
            "subject": {"value": "физика", "source_name": SEMANTIC_READING_SLOT_SOURCE, "confidence": 0.94},
        },
    )
    view = memory.to_prompt_view()

    prompt = _build_direct_path_prompt(
        "Сколько стоит?",
        context={
            "active_brand": "foton",
            "dialogue_memory_view": view,
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        },
        facts={},
    )

    assert "модель вывела из реплики" in prompt
    assert "НЕ подтверждение клиента" in prompt
    assert "клиент уже назвал — НЕ переспрашивай: класс: 9" not in prompt
    assert _direct_path_prompt_known_slots({"dialogue_memory_view": view})["grade"] == "9"


def test_slots_reask_ignores_empty_hidden_values_and_never_leaks_sentinels(monkeypatch) -> None:
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    previous = {
        "session_id": "s1",
        "active_brand": "foton",
        "semantic_reading_slots": {
            "grade": {"value": "", "source_name": SEMANTIC_READING_SLOT_SOURCE},
            "subject": {"value": "химия#SENTINEL#", "source_name": SEMANTIC_READING_SLOT_SOURCE},
        },
    }

    followup = build_dialogue_memory(
        current_message="А расписание есть?",
        active_brand="foton",
        previous_memory=previous,
    )
    prompt_view_text = json.dumps(followup.to_prompt_view(), ensure_ascii=False, sort_keys=True)

    assert "subject" in followup.do_not_reask_slots
    assert "grade" not in followup.do_not_reask_slots
    assert "химия#SENTINEL#" not in prompt_view_text
    assert "semantic_reading_slots" not in prompt_view_text


def test_slots_reask_survives_memory_llm_update_without_value_leak(monkeypatch) -> None:
    monkeypatch.setenv(SEMANTIC_READING_CLASSES_ENV, "slots_gsf")
    monkeypatch.setenv(SLOTS_REASK_ENV, "1")
    memory = DialogueMemory(
        session_id="s1",
        active_brand="foton",
        turns=(DialogueTurn("client", "Нужна физика онлайн для 9 класса."),),
    )
    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=SemanticReading(
            source="inline",
            product_grade="9 класс",
            product_subject="физика",
            product_format="онлайн",
            frame_confidence=0.91,
        ).to_memory_dict(),
        memory_llm_fn=lambda _prompt: {"slots": {}, "topic": {}, "open_question": {}, "commitments": [], "summary": ""},
    )
    prompt_view_text = json.dumps(updated.to_prompt_view(), ensure_ascii=False, sort_keys=True)

    assert set(updated.do_not_reask_slots) >= {"grade", "subject", "format"}
    assert "semantic_reading_slots" not in updated.to_prompt_view()
    assert "физика" not in updated.to_prompt_view()["known_slots"].values()
    assert "semantic_reading_slots" not in prompt_view_text


def test_slot_candidates_reject_multi_subject_and_format_choice() -> None:
    multi_subject = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_subject="математика",
            frame_confidence=0.91,
        ),
        history_texts=("Нужны математика и физика для 8 класса.",),
    )
    comma_subject = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_subject="математика",
            frame_confidence=0.91,
        ),
        history_texts=("Интересует математика, физика для 8 класса.",),
    )
    format_choice = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_format="онлайн",
            frame_confidence=0.91,
        ),
        history_texts=("У вас онлайн или очно для 7 класса?",),
    )
    slash_format_choice = slot_candidates_from_reading(
        SemanticReading(
            source="inline",
            product_format="очно",
            frame_confidence=0.91,
        ),
        history_texts=("Формат очно/онлайн ещё выбираем.",),
    )

    assert "subject" not in multi_subject
    assert "subject" not in comma_subject
    assert "format" not in format_choice
    assert "format" not in slash_format_choice


def test_slot_candidates_do_not_treat_address_or_venue_as_offline_format() -> None:
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_format="адрес", frame_confidence=0.91),
            history_texts=("Подскажите адрес занятий для 8 класса.",),
        ).get("format")
        is None
    )
    assert (
        slot_candidates_from_reading(
            SemanticReading(source="inline", product_format="площадка", frame_confidence=0.91),
            history_texts=("Какая площадка у занятий?",),
        ).get("format")
        is None
    )


def test_slot_candidates_floor_uses_client_authored_lines_only() -> None:
    reading = SemanticReading(
        source="inline",
        product_subject="физика",
        product_format="онлайн",
        frame_confidence=0.91,
    )

    assert (
        slot_candidates_from_reading(
            reading,
            history_texts=("Ответ: Есть физика онлайн для 8 класса.", "Клиент: Подскажите адрес."),
        )
        == {}
    )
    slots = slot_candidates_from_reading(
        reading,
        history_texts=("Клиент: Нужна физика онлайн.",),
    )
    assert slots["subject"]["value"] == "физика"
    assert slots["format"]["value"] == "онлайн"


def test_dialogue_memory_persists_semantic_reading_without_client_confirmation() -> None:
    memory = build_dialogue_memory(
        current_message="Здравствуйте",
        active_brand="foton",
    )
    reading = SemanticReading(
        source="inline",
        primary_intent="schedule",
        product_grade="9 класс",
        product_subject="физика",
        product_format="онлайн",
        frame_confidence=0.82,
    )

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Передам менеджеру.",
        route="draft_for_manager",
        semantic_reading=reading.to_memory_dict(),
    )
    payload = updated.to_json_dict()
    rebuilt = dialogue_memory_from_mapping(payload)

    assert payload["last_semantic_reading"]["source"] == "inline"
    assert rebuilt.last_semantic_reading["product_subject"] == "физика"
    assert "grade" not in rebuilt.client_confirmed_slots
    assert "last_semantic_reading" not in rebuilt.to_prompt_view()


def test_semantic_reading_pure_decisions_do_not_switch_routes() -> None:
    seats = SemanticReading(source="inline", primary_intent="live_availability", sense="seats", intent_confidence=0.82)
    venue = SemanticReading(source="inline", primary_intent="address", sense="venue", intent_confidence=0.83)
    off_topic = SemanticReading(source="inline", primary_intent="off_topic", intent_confidence=0.91)
    low_confidence = SemanticReading(source="inline", primary_intent="off_topic", intent_confidence=0.3)

    assert sense_seats_reading_decision(seats, "Есть места?") == "seats"
    assert sense_seats_reading_decision(venue, "Подскажите место занятий") == "not_seats"
    assert off_topic_reading_decision(off_topic) == "off_topic"
    assert off_topic_reading_decision(low_confidence) == ""


def test_slots_reading_candidates_wraps_floor_with_same_contract() -> None:
    reading = SemanticReading(
        source="inline",
        product_grade="8 класс",
        product_subject="физика",
        frame_confidence=0.88,
    )

    slots = slots_reading_candidates(reading, ("Клиент: Нужна физика для 8 класса.",))

    assert slots["grade"]["value"] == "8"
    assert slots["subject"]["source_name"] == SEMANTIC_READING_SLOT_SOURCE


def test_slot_gold_19_floor_has_no_false_writes_and_known_fixture_gaps_are_explicit() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures/adr003_slot_gold_19_machine_readable.json"
    rows = json.loads(fixture.read_text(encoding="utf-8"))["rows"]
    known_fixture_or_policy_gaps = {
        "wappi_pair_missing_72h_004",
        "wappi_pair_missing_72h_012",
        "wappi_pair_missing_72h_019",
        "wappi_pair_missing_72h_020",
    }
    mismatches: list[tuple[str, str, str, str]] = []
    for row in rows:
        if row.get("unresolved") or row.get("field") != "grade":
            continue
        reading = SemanticReading(source="inline", product_grade=str(row.get("frame_value") or ""), frame_confidence=0.91)
        slots = slot_candidates_from_reading(reading, history_texts=tuple(row.get("client_quotes") or ()))
        got = "yes" if slots.get("grade") else "none"
        expected = str(row.get("expected_slot_write") or "")
        dialog_id = str(row.get("dialog_id") or "")
        if got != expected:
            mismatches.append((dialog_id, str(row.get("source_class") or ""), expected, got))
    assert mismatches == [
        ("wappi_pair_missing_72h_004", "client_history", "yes", "none"),
        ("wappi_pair_missing_72h_012", "client_history", "yes", "none"),
        ("wappi_pair_missing_72h_019", "client_history", "yes", "none"),
        ("wappi_pair_missing_72h_020", "client_history", "yes", "none"),
    ]
    assert {item[0] for item in mismatches} == known_fixture_or_policy_gaps
