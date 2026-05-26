from __future__ import annotations

from mango_mvp.channels.answer_plan import build_answer_plan
from mango_mvp.channels.held_state import HeldState, update_held
from mango_mvp.channels.semantic_roles import tag_message_roles


def _run_dialog(turns: list[str]) -> list[tuple[object, object, HeldState]]:
    held = HeldState()
    out: list[tuple[object, object, HeldState]] = []
    for text in turns:
        roles = tag_message_roles(text, context=held.tagger_context())
        plan = build_answer_plan(roles, external_p0=held.p0_latched)
        held = update_held(held, text, roles, p0_required=plan.p0_required)
        out.append((roles, plan, held))
    return out


def test_transfer_followup_inherits_group_context() -> None:
    steps = _run_dialog(["можно перевести ребёнка в другую группу, если сложно?", "то есть реально переводят?"])

    assert steps[0][0].transfer_sense == "group"
    assert steps[1][0].transfer_sense == "group"


def test_group_topic_resolves_bare_transfer_followup() -> None:
    steps = _run_dialog(["вы тестируете уровень перед группой или сразу берёте?", "а перевести потом можно?"])

    assert steps[0][2].group_topic_active is True
    assert steps[1][0].transfer_sense == "group"


def test_explicit_format_correction_overrides_held_state() -> None:
    steps = _run_dialog(["хочу заниматься онлайн", "ой нет, я как раз про очно"])

    assert steps[0][2].training_format == "online"
    assert steps[1][2].training_format == "ochno"


def test_negated_format_correction_does_not_keep_old_online() -> None:
    steps = _run_dialog(["хочу заниматься онлайн", "только не онлайн, передайте менеджеру очное пробное"])

    assert steps[0][2].training_format == "online"
    assert steps[1][2].training_format == "ochno"


def test_neutral_followup_keeps_previous_format() -> None:
    steps = _run_dialog(["сколько стоит онлайн 9 класс?", "а расписание какое?"])

    assert steps[1][2].training_format == "online"


def test_p0_latch_survives_later_safe_question() -> None:
    steps = _run_dialog(["сколько стоит очно 8 класс?", "я оплатил, а группу не открыли, верните деньги", "так что там по цене?"])

    assert steps[1][1].p0_required is True
    assert steps[2][1].p0_required is True
    assert steps[2][1].route == "manager_only"


def test_bare_transfer_without_context_stays_unresolved() -> None:
    roles = tag_message_roles("то есть реально переводят?")

    assert roles.transfer_sense == ""


def test_invoice_monthly_followup_does_not_latch_installment() -> None:
    steps = _run_dialog(
        [
            "Можно оплатить банковским переводом на счёт?",
            "а помесячно так можно?",
            "я про счёт каждый месяц, не рассрочку",
        ]
    )

    assert steps[1][0].payment_method == "invoice_monthly"
    assert "installment" not in steps[1][0].topics
    assert steps[2][0].payment_method == "invoice_monthly"
    assert "installment" not in steps[2][2].active_topics


def test_both_formats_survive_in_held_state() -> None:
    steps = _run_dialog(["хочу онлайн", "можно и очно, и онлайн, пусть оба варианта"])

    assert steps[1][2].training_format == ""
    assert set(steps[1][2].training_formats) == {"online", "ochno"}
