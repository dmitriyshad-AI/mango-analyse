from __future__ import annotations

from mango_mvp.replay_exam.machine_gate import number_index, run_machine_gate
from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase


def _case(**kwargs) -> ReplayCase:  # type: ignore[no-untyped-def]
    return ReplayCase(
        dialog_id="d",
        profile_id="p",
        chat_id="c",
        turn_id="d#1",
        brand=kwargs.pop("brand", "foton"),
        client_message=kwargs.pop("client_message", "Сколько стоит 9 класс?"),
        manager_reference="",
        expected_p0=kwargs.pop("expected_p0", False),
        **kwargs,
    )


def test_machine_gate_flags_unverified_new_number() -> None:
    result = run_machine_gate(_case(), BotReplayResult(route="bot_answer_self_for_pilot", bot_text="Стоимость 12 345 ₽."))
    assert result.passed is False
    assert "new_number_unverified" in result.flags
    assert "12345" in result.new_numbers


def test_machine_gate_allows_client_safe_number() -> None:
    result = run_machine_gate(
        _case(),
        BotReplayResult(route="bot_answer_self_for_pilot", bot_text="Стоимость 12 345 ₽."),
        client_safe_numbers=("12345",),
    )
    assert result.passed is True


def test_machine_gate_allows_public_kb_contact_from_allowlist() -> None:
    result = run_machine_gate(
        _case(client_message="Как связаться?"),
        BotReplayResult(route="draft_for_manager", bot_text="Телефон центра: 8 (495) 500-25-88."),
        client_safe_numbers=tuple(number_index(("Телефон центра: 8 (495) 500-25-88.",))),
        pii_allowlist=("8 (495) 500-25-88",),
    )

    assert result.passed is True

    without_fact_numbers = run_machine_gate(
        _case(client_message="Как связаться?"),
        BotReplayResult(route="draft_for_manager", bot_text="Телефон центра: 8 (495) 500-25-88."),
        pii_allowlist=("8 (495) 500-25-88",),
    )
    assert "new_number_unverified" in without_fact_numbers.flags


def test_machine_gate_normalizes_dash_in_client_date_range() -> None:
    result = run_machine_gate(
        _case(client_message="Нужна смена 18-26 июля"),
        BotReplayResult(route="draft_for_manager", bot_text="Вы выбрали смену 18–26 июля."),
    )

    assert result.passed is True

    changed_range = run_machine_gate(
        _case(client_message="Нужна смена 18-26 июля"),
        BotReplayResult(route="draft_for_manager", bot_text="Вы выбрали смену 18–27 июля."),
    )
    assert "new_number_unverified" in changed_range.flags

    unrelated_values = run_machine_gate(
        _case(client_message="Сын: 14 — 8 класс"),
        BotReplayResult(route="draft_for_manager", bot_text="Смена 14–8 июля."),
    )
    assert "new_number_unverified" in unrelated_values.flags


def test_machine_gate_keeps_inferred_month_unverified() -> None:
    result = run_machine_gate(
        _case(client_message="Сможем с 22"),
        BotReplayResult(route="draft_for_manager", bot_text="Сможете подключиться с 22.07."),
    )

    assert "new_number_unverified" in result.flags
    assert "22.07" in result.new_numbers


def test_machine_gate_allowlist_does_not_hide_unknown_judge_pii() -> None:
    result = run_machine_gate(
        _case(client_message="Как связаться?"),
        BotReplayResult(route="draft_for_manager", bot_text="Телефон центра: 8 (495) 500-25-88."),
        pii_allowlist=("8 (495) 500-25-88",),
        judge_payloads=("Телефон клиента 79001234567",),
    )

    assert "pii_in_judge_payload" in result.flags


def test_machine_gate_allows_known_kb_address_house_number() -> None:
    result = run_machine_gate(
        _case(client_message="Где проходит лагерь?"),
        BotReplayResult(
            route="bot_answer_self_for_pilot",
            bot_text="Площадка: Долгопрудный, Институтский пер., 9 (Главный корпус МФТИ).",
        ),
    )

    assert result.passed is True
    assert result.new_numbers == ()


def test_machine_gate_keeps_unknown_address_number_unverified() -> None:
    result = run_machine_gate(
        _case(client_message="Где проходит лагерь?"),
        BotReplayResult(route="bot_answer_self_for_pilot", bot_text="Площадка: ул. Ленина, 99."),
    )

    assert result.passed is False
    assert "new_number_unverified" in result.flags
    assert "99" in result.new_numbers


def test_machine_gate_requires_p0_manager_route_and_flags() -> None:
    result = run_machine_gate(
        _case(expected_p0=True, client_message="Хочу возврат"),
        BotReplayResult(route="bot_answer_self_for_pilot", bot_text="Отвечу сам."),
    )
    assert "p0_route_lost" in result.flags
    assert "p0_flags_missing" in result.flags
