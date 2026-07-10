from __future__ import annotations

from mango_mvp.replay_exam.machine_gate import run_machine_gate
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
