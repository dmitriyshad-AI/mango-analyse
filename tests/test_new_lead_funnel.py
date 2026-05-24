from __future__ import annotations

from mango_mvp.channels.new_lead_funnel import build_lead_funnel_state


def test_extracts_known_slots_without_reasking_first_price_message() -> None:
    state = build_lead_funnel_state(
        "7 класс, математика, хотим очно в Москве. Сколько стоит?",
        active_brand="foton",
        topic_id="theme:001_pricing",
        context={"client_identity": {"channel_user_id": "tg-1"}},
    )

    payload = state.to_json_dict()

    assert payload["client_segment"] == "new_lead"
    assert payload["filled_slots"]["grade"] == "7"
    assert payload["filled_slots"]["subject"] == "математика"
    assert payload["filled_slots"]["format"] == "offline"
    assert payload["filled_slots"]["city"] == "Москва"
    assert "grade" not in payload["missing_slots"]
    assert "subject" not in payload["missing_slots"]
    assert payload["next_step_type"] in {"ask_goal", "offer_group_check"}


def test_format_extraction_does_not_treat_tochnoy_as_offline() -> None:
    state = build_lead_funnel_state(
        "29 750 это семестр, а 47 250 год, правильно? и без точной даты повышения?",
        active_brand="foton",
        topic_id="theme:001_pricing",
        recent_messages=["Клиент: онлайн 8 класс физика, без воды"],
    )

    payload = state.to_json_dict()

    assert payload["filled_slots"]["format"] == "online"
    assert payload["filled_slots"]["grade"] == "8"
    assert payload["filled_slots"]["subject"] == "физика"


def test_missing_grade_becomes_next_best_question() -> None:
    state = build_lead_funnel_state(
        "Нужна онлайн-физика для подготовки к олимпиадам",
        active_brand="unpk",
        topic_id="theme:016_program",
    )

    assert state.next_step_type == "ask_grade"
    assert "классе" in state.next_best_question
    assert state.missing_slots == ("grade",)


def test_camp_flow_asks_class_not_age_and_never_promises_seats() -> None:
    state = build_lead_funnel_state(
        "Хотим в летний лагерь в Менделеево, есть места?",
        active_brand="foton",
        topic_id="theme:026_camp_general",
    )

    assert state.product_scope == "lvsh"
    assert state.next_step_type == "ask_camp_class"
    assert "классе" in state.next_best_question
    assert "возраст" not in state.next_best_question.casefold()


def test_funnel_slots_ignore_bot_answers_to_avoid_self_pollution() -> None:
    state = build_lead_funnel_state(
        "а места на выездную еще есть?",
        active_brand="foton",
        topic_id="theme:026_camp_general",
        recent_messages=[
            "Клиент: а выездная смена есть? сколько она стоит?",
            "Ответ: В ЛВШ есть физика, математика, Python и проектная деятельность.",
        ],
    )

    assert state.known_slots.subject == ""
    assert "программ" not in state.to_json_dict()["filled_slots"].get("subject", "")


def test_funnel_does_not_treat_program_word_as_programming_subject() -> None:
    state = build_lead_funnel_state(
        "8 класс информатика очно, без подбора программы",
        active_brand="foton",
        topic_id="service:S5_general_consultation",
    )

    assert state.known_slots.subject == "информатика"
    assert "программирование" not in state.to_json_dict()["filled_slots"].get("subject", "")


def test_p0_composite_stops_qualification() -> None:
    state = build_lead_funnel_state(
        "Сколько стоит курс и как вернуть деньги за прошлый месяц?",
        active_brand="unpk",
        topic_id="theme:001_pricing",
    )

    assert state.lead_stage == "p0_manager_only"
    assert state.next_step_type == "manager_only_p0"
    assert state.missing_slots == ()
    assert "p0_blocks_qualification" in state.semantic_flags


def test_known_customer_does_not_request_phone_or_student_name() -> None:
    state = build_lead_funnel_state(
        "Можно забронировать смену 17-28 августа?",
        active_brand="unpk",
        topic_id="theme:026_camp_general",
        context={
            "known_client_fields": {
                "student_name": "Колосов Даниил Максимович",
                "parent_name": "Ананьевская Анна Георгиевна",
                "phone": "79092009933",
                "grade": "9",
            },
            "client_identity": {"debug_impersonation": True, "phone": "79092009933"},
        },
    )

    assert state.client_segment == "staff_test"
    assert state.known_slots.student_name == "Колосов Даниил Максимович"
    assert state.known_slots.phone_known is True
    assert state.next_step_type == "offer_manager_seat_check"
    assert "student_known_do_not_reask_name" in state.semantic_flags
