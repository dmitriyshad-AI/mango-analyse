from __future__ import annotations

"""Тесты слоя человечности. Кейсы взяты из реальных провалов round-5 (101407).
Запуск: python3 D1_semantic_roles_reference/tests/test_humanity_guards.py
"""

from mango_mvp.channels.humanity_guards import (  # noqa: E402
    is_near_repeat, has_meta_leak, meta_markers_present,
    should_answer_not_handoff, unanswered_direct_question, repeat_ratio,
    humanity_route_action,
)

_P = _F = 0
_FA: list[str] = []


def check(n, c, d=""):
    global _P, _F
    if c:
        _P += 1
    else:
        _F += 1
        _FA.append(f"[FAIL] {n}: {d}")


def main() -> int:
    global _P, _F, _FA
    _P = _F = 0
    _FA = []

    # 1. Повтор: диалог 02 ход1==ход2 дословно (реальный кейс)
    t1 = "Да, это можно уточнить заранее по 9 класс, физика, очно. Такой вопрос до оплаты не оформляю как жалобу или заявление на возврат. По смыслу: возможность есть."
    check("repeat_verbatim", is_near_repeat(t1, [t1]) is True, "дословный повтор не пойман")
    check("repeat_ratio_self", repeat_ratio(t1, t1) == 1.0)
    diff = "По физике очно 9 класс: семестр 44 600 ₽, год 74 500 ₽. Подскажу, как оформить."
    check("not_repeat_different", is_near_repeat(diff, [t1]) is False, "разные ответы посчитаны повтором")
    check("short_reply_not_repeat", is_near_repeat("Поняла, спасибо.", ["Поняла, спасибо."]) is False, "короткая реплика не должна латчиться как повтор")

    # 2. Мета-утечка: реальный текст из диалога 09
    leak = "Клиент понял условия и взял паузу. Автономный ответ не требуется. Если менеджер решит ответить, безопасный вариант: «Конечно, подумайте»."
    check("meta_leak_caught", has_meta_leak(leak) is True, f"мета не поймана; {meta_markers_present(leak)}")
    check("meta_leak_dl02", has_meta_leak(t1) is True, "«не оформляю как жалобу» — мета")
    normal = "По физике очно 9 класс семестр 44 600 ₽. Помогу оформить дистанционно."
    check("normal_no_meta", has_meta_leak(normal) is False, "нормальный ответ ложно помечен мета")
    fallback = "Передам вопрос менеджеру, он ответит по сути."
    check("meta_fallback_no_leak", has_meta_leak(fallback) is False, "аварийный fallback не должен быть мета-утечкой")
    for index, phrase in enumerate(
        (
            "В фактах нет информации по этому вопросу.",
            "В базе нет точного ответа.",
            "Нет в данных точного расписания.",
            "Цена не указана в фактах.",
            "Нет в фактах точного расписания.",
        ),
        start=1,
    ):
        check(f"meta_fact_phrase_pos_{index}", has_meta_leak(phrase) is True, phrase)
        check(f"meta_fact_phrase_marker_{index}", "fact_phrase_leak" in meta_markers_present(phrase), phrase)
    for index, phrase in enumerate(
        (
            "Точное время мы согласуем.",
            "Дату уточнит менеджер.",
            "Расписание уточняется.",
            "У нас нет занятий по выходным.",
            "Класс не указан в анкете.",
            "Класс не указан в договоре.",
            "В группе нет свободных мест.",
        ),
        start=1,
    ):
        check(f"meta_fact_phrase_neg_{index}", has_meta_leak(phrase) is False, phrase)

    # 3. Over-handoff: факт есть, P0 нет, ушёл в менеджера → флаг
    check("overhandoff_flag", should_answer_not_handoff(p0_required=False, has_retrieved_answer_fact=True, route="manager_only") is True)
    check("overhandoff_draft", should_answer_not_handoff(p0_required=False, has_retrieved_answer_fact=True, route="draft_for_manager") is True)
    check("p0_not_flagged", should_answer_not_handoff(p0_required=True, has_retrieved_answer_fact=True, route="manager_only") is False, "P0 нельзя ослаблять")
    check("no_fact_no_flag", should_answer_not_handoff(p0_required=False, has_retrieved_answer_fact=False, route="manager_only") is False)
    check("answer_route_ok", should_answer_not_handoff(p0_required=False, has_retrieved_answer_fact=True, route="bot_answer_self") is False)

    # 4. Игнор нового вопроса: клиент про «правила/договор» (document), черновик про возврат (refund)
    check("ignored_new_q", unanswered_direct_question("а где посмотреть эти правила, договор?", t1, client_topics=["document"], draft_topics=["refund_presale"]) is True)
    check("answered_q_ok", unanswered_direct_question("сколько стоит?", "Цена 44 600 ₽ за семестр", client_topics=["price"], draft_topics=["price"]) is False)
    check("no_question_no_flag", unanswered_direct_question("понятно, спасибо", "Рад помочь", client_topics=["price"], draft_topics=[]) is False)

    # 5. humanity_route_action — ДЕЙСТВЕННЫЙ override (не no-op)
    a = humanity_route_action(p0_required=False, has_retrieved_answer_fact=True, route="manager_only", message_type="question")
    check("action_flips_to_answer", a["route"] == "bot_answer_self" and a["regenerate"] is True, str(a))
    a = humanity_route_action(p0_required=True, has_retrieved_answer_fact=True, route="manager_only", message_type="question")
    check("action_p0_kept", a["route"] == "manager_only" and a["regenerate"] is False, str(a))
    a = humanity_route_action(p0_required=False, has_retrieved_answer_fact=True, route="manager_only", message_type="context_update")
    check("action_gratitude_kept", a["regenerate"] is False, str(a))
    a = humanity_route_action(p0_required=False, has_retrieved_answer_fact=False, route="bot_answer_self", message_type="question", direct_question_answered=False)
    check("action_regen_delta", a["regenerate"] is True and a["route"] == "bot_answer_self", str(a))

    print(f"PASS={_P}  FAIL={_F}")
    for f in _FA:
        print(f)
    return 1 if _F else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_humanity_guards_reference_cases() -> None:
    assert main() == 0
