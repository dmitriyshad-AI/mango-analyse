from __future__ import annotations

from mango_mvp.insights.pilot_extraction import (
    choose_customer_question_or_need,
    choose_manager_answer,
    ideal_reaction_for_signal,
    infer_customer_signal,
    infer_hidden_sales_stage,
    parse_role_blocks,
    score_manager_response,
    select_calls_for_client,
    split_sentences,
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
