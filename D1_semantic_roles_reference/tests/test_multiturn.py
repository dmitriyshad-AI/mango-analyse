from __future__ import annotations

"""Многоходовые тесты held-состояния (без вызова модели).

Проверяем ровно классы удержания контекста:
  - наследование смысла «перевод» из held на follow-up (пункт 3 ТЗ);
  - класс C: явная поправка клиента перезаписывает held (онлайн↔очно);
  - append-only: нейтральный follow-up НЕ сбрасывает ранее названный слот;
  - P0-latch: спор держится на последующей мирной реплике.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)

from semantic_roles import tag_message_roles            # noqa: E402
from decision_policy import build_answer_plan            # noqa: E402
from held_state import HeldState, update_held            # noqa: E402

_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"[FAIL] {name}: {detail}")


def run_dialog(turns):
    """Прогнать список клиентских реплик, вернуть список (roles, plan, held)."""
    held = HeldState()
    out = []
    for text in turns:
        ctx = held.tagger_context()
        roles = tag_message_roles(text, context=ctx)
        plan = build_answer_plan(roles, external_p0=held.p0_latched)
        held = update_held(held, text, roles, p0_required=plan.p0_required)
        out.append((roles, plan, held))
    return out


def main() -> int:
    # 1. Наследование «перевода»: follow-up без соседа берёт group из held.
    steps = run_dialog(["можно перевести ребёнка в другую группу, если сложно?", "то есть реально переводят?"])
    check("transfer_turn0_group", steps[0][0].transfer_sense == "group", steps[0][0].transfer_sense)
    check("transfer_turn1_inherits_group", steps[1][0].transfer_sense == "group", steps[1][0].transfer_sense)

    # 1b. Групповой топик без явного перевода на t0, перевод-followup на t1.
    steps = run_dialog(["вы тестируете уровень перед группой или сразу берёте?", "а перевести потом можно?"])
    check("group_topic_active_after_t0", steps[0][2].group_topic_active is True, "")
    check("transfer_followup_to_group", steps[1][0].transfer_sense == "group", steps[1][0].transfer_sense)

    # 2. Класс C: явная поправка онлайн→очно перезаписывает held.
    steps = run_dialog(["хочу заниматься онлайн", "ой нет, я как раз про очно"])
    check("format_t0_online", steps[0][2].training_format == "online", steps[0][2].training_format)
    check("classC_override_to_ochno", steps[1][2].training_format == "ochno", steps[1][2].training_format)

    # 3. Append-only: нейтральный follow-up не сбрасывает формат.
    steps = run_dialog(["сколько стоит онлайн 9 класс?", "а расписание какое?"])
    check("format_held_after_neutral", steps[1][2].training_format == "online", steps[1][2].training_format)

    # 4. P0-latch: спор на t1 держится на мирной реплике t2.
    steps = run_dialog(["сколько стоит очно 8 класс?", "я оплатил, а группу не открыли, верните деньги", "так что там по цене?"])
    check("dispute_t1_p0", steps[1][1].p0_required is True, "")
    check("p0_latched_t2", steps[2][1].p0_required is True, f"p0={steps[2][1].p0_required}")
    check("p0_route_t2_manager", steps[2][1].route == "manager_only", steps[2][1].route)

    # 5. Контроль: без контекста голый follow-up перевода остаётся неразрешённым.
    r = tag_message_roles("то есть реально переводят?")
    check("no_context_transfer_unknown", r.transfer_sense == "", r.transfer_sense)

    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for f in _FAILURES:
        print(f)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
