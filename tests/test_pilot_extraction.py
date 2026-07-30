from __future__ import annotations

import json

import pytest

from mango_mvp.insights.pilot_extraction import (
    build_llm_input,
    choose_customer_question_or_need,
    choose_manager_answer,
    extract_sales_moment,
    ideal_reaction_for_signal,
    infer_customer_signal,
    infer_hidden_sales_stage,
    parse_role_blocks,
    sales_moment_exclusion_reason,
    score_manager_response,
    select_calls_for_client,
    split_sentences,
)


def _confirmed_variants() -> str:
    return json.dumps(
        {
            "call_topology": "simple_two_party",
            "role_mapping": {
                "manager_quality_allowed": True,
                "confirmed": True,
                "topology": "simple_two_party",
            },
            "manager": {"physical_channel": "left"},
            "client": {"physical_channel": "right"},
        }
    )


def test_parse_role_blocks_extracts_manager_and_client() -> None:
    roles = parse_role_blocks("MANAGER:\nЗдравствуйте, можем отправить расписание.\n\nCLIENT:\nСколько стоит курс?")

    assert "отправить расписание" in roles["manager"]
    assert "Сколько стоит" in roles["client"]


def test_split_sentences_chunks_long_asr_text() -> None:
    sentences = split_sentences(" ".join(["слово"] * 120))

    assert len(sentences) > 1
    assert all(len(item) <= 260 for item in sentences)


def test_select_calls_keeps_first_last_and_high_signal_calls() -> None:
    chain = {"extraction_use_case": "reactivation_revenue"}
    calls = [
        {
            "source_filename": "first.mp3",
            "started_at": "2025-01-01",
            "contentful": "True",
            "call_type": "sales_call",
            "history_summary": "first",
        },
        {
            "source_filename": "middle.mp3",
            "started_at": "2025-01-02",
            "contentful": "True",
            "call_type": "sales_call",
            "next_step": "Перезвонить",
            "objections": "цена",
            "lead_priority": "warm",
            "history_summary": "middle " * 40,
        },
        {
            "source_filename": "last.mp3",
            "started_at": "2025-01-03",
            "contentful": "True",
            "call_type": "service_call",
            "history_summary": "last",
        },
    ]

    selected = select_calls_for_client(chain, calls, 3)

    assert [row["source_filename"] for row in selected] == ["first.mp3", "middle.mp3", "last.mp3"]


def test_question_and_answer_fallback_to_structured_fields() -> None:
    structured = {
        "interests": {"products": ["годовые курсы"], "subjects": ["математика"]},
        "objections": ["цена"],
        "next_step": {"action": "Отправить материалы"},
    }

    question = choose_customer_question_or_need("", "", structured, {})
    answer = choose_manager_answer("", "", structured, {})

    assert "годовые курсы" in question
    assert "цена" in question
    assert "Отправить материалы" in answer


def test_signal_stage_and_quality_for_price_objection() -> None:
    chain = {
        "final_outcome_label": "open_sales_potential",
        "extraction_use_case": "open_pipeline_learning",
        "touch_count": "5",
    }
    call = {
        "call_type": "sales_call",
        "lead_priority": "warm",
        "next_step": "Отправить ссылку на оплату",
        "objections": "цена",
        "_sequence_position": "2",
    }

    signal, evidence = infer_customer_signal(chain, call, "Сколько стоит курс и есть ли скидка?", "", "")
    stage = infer_hidden_sales_stage(chain, call, signal)
    score, band, reasons = score_manager_response(
        chain,
        call,
        signal,
        "Сколько стоит курс?",
        "Менеджер объяснил стоимость курса, доступную скидку, порядок оплаты и отправил ссылку на оплату.",
    )

    assert signal == "price_or_payment"
    assert "сто" in evidence.lower()
    assert stage == "objection_handling"
    assert score >= 75
    assert band == "high"
    assert "addresses_customer_signal" in reasons


def test_ideal_reaction_for_next_year_interest_mentions_follow_up() -> None:
    reaction, template = ideal_reaction_for_signal("next_year_interest", {"products_top": "годовые курсы: 3"}, {})

    assert "следующий год" in reaction
    assert "следующий учебный год" in template


def test_sales_moment_exclusion_filters_no_live_technical_call() -> None:
    call = {
        "source_filename": "no_live.mp3",
        "phone": "79990000000",
        "started_at": "2026-04-14 16:11:16",
        "manager_name": "Менеджер",
        "call_type": "technical_call",
        "history_summary": "Абонент сейчас не может ответить на звонок, живого диалога не было.",
    }
    db_record = {
        "duration_sec": 14.4,
        "transcript_text": (
            "MANAGER: Продолжение следует...\n"
            "CLIENT: Абонент сейчас не может ответить на ваш звонок. Его телефон занят. "
            "Попробуйте перезвонить позднее."
        ),
        "analysis_json": "{}",
        "transcript_variants_json": _confirmed_variants(),
    }

    exclusion = sales_moment_exclusion_reason({"final_outcome_label": "open_sales_potential"}, call, db_record)

    assert exclusion is not None
    assert exclusion["exclusion_reason"] == "no_live_or_voicemail_not_safe_for_sales_kb"


def test_sales_moment_exclusion_keeps_bridge_artifact_with_live_dialogue() -> None:
    call = {
        "source_filename": "bridge_live.mp3",
        "phone": "79990000000",
        "started_at": "2026-03-28 12:38:50",
        "manager_name": "Менеджер",
        "call_type": "sales_call",
        "history_summary": "Клиент спросил про курс, формат, расписание и стоимость; менеджер предложил варианты.",
        "products": "летняя школа",
        "subjects": "математика",
    }
    db_record = {
        "duration_sec": 180,
        "transcript_text": (
            "MANAGER: Продолжаем дозваниваться. Оставайтесь на линии. Алло, добрый день, учебный центр.\n"
            "CLIENT: Здравствуйте, интересует летняя школа по математике, какой формат и сколько стоит?\n"
            "MANAGER: Расскажу варианты, расписание и стоимость, затем отправлю материалы."
        ),
        "analysis_json": "{}",
        "transcript_variants_json": _confirmed_variants(),
    }

    assert sales_moment_exclusion_reason({"final_outcome_label": "open_sales_potential"}, call, db_record) is None


def test_manager_quality_requires_explicit_confirmed_role_gate() -> None:
    call = {"source_filename": "call.mp3", "call_type": "sales_call"}
    db_record = {
        "transcript_text": "MANAGER: Ответ. CLIENT: Вопрос.",
        "analysis_json": "{}",
        "transcript_variants_json": json.dumps(
            {"role_mapping": {"manager_quality_allowed": False}}
        ),
    }
    exclusion = sales_moment_exclusion_reason({}, call, db_record)
    assert exclusion is not None
    assert exclusion["exclusion_reason"] == "unconfirmed_roles_not_safe_for_manager_quality"
    with pytest.raises(ValueError, match="explicit confirmed"):
        extract_sales_moment(1, {}, call, db_record)
    with pytest.raises(ValueError, match="explicit confirmed"):
        build_llm_input({}, {}, call, db_record, 1000)


def test_contradictory_true_role_flag_is_still_blocked() -> None:
    call = {"source_filename": "call.mp3", "call_type": "sales_call"}
    db_record = {
        "transcript_text": "MANAGER: Ответ. CLIENT: Вопрос.",
        "analysis_json": "{}",
        "transcript_variants_json": json.dumps(
            {
                "call_topology": "echo_or_duplicate_channels",
                "role_mapping": {
                    "manager_quality_allowed": True,
                    "confirmed": False,
                    "topology": "echo_or_duplicate_channels",
                },
            }
        ),
    }
    exclusion = sales_moment_exclusion_reason({}, call, db_record)
    assert exclusion is not None
    assert exclusion["exclusion_reason"] == "unconfirmed_roles_not_safe_for_manager_quality"


@pytest.mark.parametrize("channel_blocks", [{}, {"manager": {"physical_channel": "right"}, "client": {"physical_channel": "right"}}])
def test_missing_or_duplicate_physical_channels_are_blocked(channel_blocks) -> None:
    payload = json.loads(_confirmed_variants())
    payload.pop("manager")
    payload.pop("client")
    payload.update(channel_blocks)
    db_record = {
        "analysis_json": "{}",
        "transcript_text": "MANAGER: Ответ. CLIENT: Вопрос.",
        "transcript_variants_json": json.dumps(payload),
    }
    exclusion = sales_moment_exclusion_reason({}, {"source_filename": "call.mp3"}, db_record)
    assert exclusion is not None
    assert exclusion["exclusion_reason"] == "unconfirmed_roles_not_safe_for_manager_quality"
