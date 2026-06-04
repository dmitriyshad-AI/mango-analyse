from __future__ import annotations

from mango_mvp.insights.tone_score import score_tone, summarize_tone_scores


def test_tone_score_penalizes_bureaucratic_office_style() -> None:
    result = score_tone(
        "В рамках текущего учебного центра обучение осуществляется очно. "
        "По вашему обращению менеджер уточнит ближайший шаг и детали."
    )

    assert result.tone_canc >= 3
    assert result.tone_warm == 0
    assert result.tone_score <= 25
    assert "canc:current_center_frame" in result.flags
    assert "canc:manager_clarifies_next_step" in result.flags


def test_tone_score_rewards_warm_direct_bot_answer() -> None:
    result = score_tone(
        "Да, пробное занятие есть. "
        "Понимаю, хочется сначала спокойно посмотреть формат; помогу подобрать удобное время под ваш класс."
    )

    assert result.tone_canc == 0
    assert result.tone_warm >= 3
    assert result.tone_score >= 90
    assert "warm:direct_first" in result.flags
    assert "warm:bot_step" in result.flags


def test_tone_score_handles_service_text_without_crashing() -> None:
    result = score_tone({"route": "draft_for_manager", "draft_text": "[manager] Передам менеджеру."})

    assert 0 <= result.tone_score <= 100
    assert result.tone_canc >= 0
    assert result.tone_warm >= 0


def test_tone_summary_returns_turn_level_and_aggregate_values() -> None:
    summary = summarize_tone_scores(
        [
            {
                "dialog_id": "tone_case",
                "brand": "foton",
                "turns": [
                    {"turn": 1, "bot_text": "В рамках текущего учебного центра услуга предоставляется."},
                    {"turn": 2, "bot_text": "Да, помогу сориентироваться по сути."},
                ],
            }
        ]
    )

    assert summary["turns_count"] == 2
    assert summary["tone_canc"] >= 2
    assert summary["tone_warm"] >= 2
    assert isinstance(summary["tone_score"], float)
    assert summary["turns"][0]["dialog_id"] == "tone_case"
    assert {"tone_canc", "tone_warm", "tone_score"} <= set(summary["turns"][0])

