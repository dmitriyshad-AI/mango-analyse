from __future__ import annotations

from mango_mvp.channels.manager_handoff_summary import build_manager_handoff_summary
from mango_mvp.channels.new_lead_funnel import build_lead_funnel_state


def test_manager_summary_contains_required_blocks() -> None:
    funnel = build_lead_funnel_state(
        "9 класс, физика. Можно узнать расписание?",
        active_brand="unpk",
        topic_id="theme:013_schedule",
    )

    summary = build_manager_handoff_summary(
        brand="unpk",
        client_message="9 класс, физика. Можно узнать расписание?",
        answer_text="Поняла: 9 класс, физика. Расписание лучше проверить по группе.",
        route="draft_for_manager",
        topic_id="theme:013_schedule",
        risk_level="low",
        missing_facts=("точное расписание",),
        manager_checklist=("Проверить группу по физике для 9 класса.",),
        funnel_state=funnel,
    )

    assert "Бренд:" in summary
    assert "Маршрут:" in summary
    assert "Вопрос клиента:" in summary
    assert "Что уже известно:" in summary
    assert "Что ответили клиенту:" in summary
    assert "Что нужно проверить:" in summary
    assert "Рекомендуемый следующий шаг:" in summary
    assert "Что нельзя обещать:" in summary
    assert "9" in summary
    assert "физика" in summary


def test_manager_summary_redacts_internal_sources() -> None:
    funnel = build_lead_funnel_state("Какая цена?", active_brand="foton", topic_id="theme:001_pricing")

    summary = build_manager_handoff_summary(
        brand="foton",
        client_message="Какая цена?",
        answer_text="Передам менеджеру.",
        route="manager_only",
        topic_id="theme:001_pricing",
        safety_flags=("source_id:abc",),
        missing_facts=("AMO lead_id 123 Tallanto token source_id",),
        manager_checklist=("CRM проверить",),
        funnel_state=funnel,
    )

    forbidden = ("AMO", "Tallanto", "CRM", "source_id", "lead_id", "token", "{", "}")
    assert not any(item in summary for item in forbidden)


def test_p0_summary_contains_zero_collect_warning() -> None:
    funnel = build_lead_funnel_state(
        "Хочу вернуть деньги",
        active_brand="unpk",
        topic_id="theme:009_refund",
    )

    summary = build_manager_handoff_summary(
        brand="unpk",
        client_message="Хочу вернуть деньги",
        answer_text="Передам вопрос ответственному сотруднику.",
        route="manager_only",
        topic_id="theme:009_refund",
        risk_level="high",
        safety_flags=("high_risk_manager_only",),
        funnel_state=funnel,
    )

    assert "Не просить" in summary
    assert "ФИО" in summary
    assert "номер договора" in summary
    assert "сумму" in summary
