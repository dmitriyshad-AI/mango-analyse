from __future__ import annotations

"""Приёмочный тест P0-перифраз (часть 1.2 ТЗ): детектор p0_recall_spec.codes_from_text
ДОЛЖЕН ловить реальные возвраты/споры в перифразах и НЕ ловить безобидные фразы.

СТАТУС НА 2026-05-25: приёмочный тест для расширенного P0-детектора.
Должен оставаться зелёным после правок REFUND_RE/PAYMENT_DISPUTE_RE/LEGAL_RE и
защиты от ложного срабатывания на «вернуться к теме цены».

Запуск (песочница): python3 D1_semantic_roles_reference/tests/test_p0_perifraz.py
В репо Codex: pytest подхватит через обычный импорт mango_mvp.channels.p0_recall_spec.
"""

import importlib.util
import os
import sys
import types

import pytest


def load_codes_from_text():
    # 1) обычный импорт (среда репо с зависимостями)
    try:
        from mango_mvp.channels.p0_recall_spec import codes_from_text  # type: ignore
        return codes_from_text
    except Exception:
        pass
    # 2) песочница: грузим по файлу в обход пакетного __init__ (тянет sqlalchemy)
    base = None
    for cand in ("src/mango_mvp/channels",
                 "/sessions/confident-sleepy-darwin/mnt/Mango analyse/src/mango_mvp/channels"):
        if os.path.isdir(cand):
            base = cand
            break
    if base is None:
        raise RuntimeError("не найден src/mango_mvp/channels")
    pm = types.ModuleType("mango_mvp"); pm.__path__ = []
    pc = types.ModuleType("mango_mvp.channels"); pc.__path__ = [base]
    sys.modules.setdefault("mango_mvp", pm)
    sys.modules.setdefault("mango_mvp.channels", pc)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"mango_mvp.channels.{name}", f"{base}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"mango_mvp.channels.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    for dep in ("text_signals", "fact_scope_spec", "new_lead_funnel", "semantic_roles"):
        try:
            load(dep)
        except Exception:
            pass
    return load("p0_recall_spec").codes_from_text


# Реальные P0 — ДОЛЖНЫ ловиться (непустой codes)
REAL_P0 = [
    "напишу в банк чарджбек",
    "оспорю операцию в банке",
    "отменю платёж через банк",
    "деньги ушли, а доступа нет",
    "оплатил, а курс так и не открыли",
    "не буду платить, это развод",
    "ребёнка унизили на занятии, я этого так не оставлю",
    "на ребёнка накричали при всей группе",
    "преподаватель высмеял сына при всех",
    "дочку довели до слёз на занятии, буду разбираться",
    "отдайте оплату обратно",
    "аннулировать договор",
    "нарушение моих прав потребителя",
    # контроли (уже должны ловиться — не сломать)
    "верните деньги за курс",
    "пойду в суд и прокуратуру",
    "буду жаловаться в роспотребнадзор",
]

# Безобидные — НЕ должны ловиться (пустой codes)
BENIGN = [
    "вернёмся к этому вопросу позже",
    "вернуться к теме цены",
    "верните меня в список рассылки",
    "у подруги был возврат, а у вас как с этим?",
    "а если передумаю до начала, деньги вернут?",
    "перед оплатой хочу понять условия возврата",
    "я не про возврат, где смотреть запись?",
    "сколько можно вернуть по налоговому вычету?",
    "ребёнок расстроился после занятия, как ему помочь?",
    "ребёнок стесняется отвечать при всех, что посоветуете?",
]


def main() -> int:
    cf = load_codes_from_text()
    p = f = 0
    fails = []
    for t in REAL_P0:
        if cf(t):
            p += 1
        else:
            f += 1
            fails.append(f"[FAIL] НЕ поймал реальный P0: «{t}»")
    for t in BENIGN:
        if not cf(t):
            p += 1
        else:
            f += 1
            fails.append(f"[FAIL] ложно поймал benign: «{t}» -> {cf(t)}")
    print(f"PASS={p}  FAIL={f}  (real_p0={len(REAL_P0)}, benign={len(BENIGN)})")
    for x in fails:
        print(x)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_p0_perifraz_reference_cases() -> None:
    assert main() == 0


def test_tz147_deep_payment_access_cases_are_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    from mango_mvp.channels.p0_recall_spec import PAYMENT_DISPUTE_DEEP_POSITIVE_CASES, codes_from_text

    monkeypatch.setenv("TELEGRAM_P0_DEEP_MATCH", "1")

    misses = [message for message in PAYMENT_DISPUTE_DEEP_POSITIVE_CASES if "payment_dispute" not in codes_from_text(message)]

    assert misses == []
    assert len(PAYMENT_DISPUTE_DEEP_POSITIVE_CASES) >= 12


@pytest.mark.parametrize(
    "message",
    (
        "оплату оформили, а доступ почему-то заблокирован",
        "оплату оформили, а доступ заблокирован",
    ),
)
def test_tz147_deep_payment_object_verb_order_is_flagged(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    from mango_mvp.channels.p0_recall_spec import codes_from_text

    monkeypatch.setenv("TELEGRAM_P0_DEEP_MATCH", "1")

    assert "payment_dispute" in codes_from_text(message)


def test_tz147_deep_payment_access_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from mango_mvp.channels.p0_recall_spec import PAYMENT_DISPUTE_DEEP_POSITIVE_CASES, codes_from_text

    monkeypatch.delenv("TELEGRAM_P0_DEEP_MATCH", raising=False)

    assert "payment_dispute" not in codes_from_text("Оплата прошла. Доступа нет.")
    assert any("payment_dispute" not in codes_from_text(message) for message in PAYMENT_DISPUTE_DEEP_POSITIVE_CASES)


@pytest.mark.parametrize(
    "message",
    (
        "Занятия завтра, в системе расписания пока нет.",
        "Оплату ещё не вносил, доступ не появился — так и должно быть?",
        "Подключили новую платформу, удобно?",
        "Ссылку выслали, спасибо!",
        "Активировали аккаунт, всё работает.",
        "Списать абонемент за пропуск — нормально?",
        "Онлайн - очень удобно.",
    ),
)
def test_tz147_deep_payment_negatives_stay_non_p0(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    from mango_mvp.channels.p0_recall_spec import codes_from_text

    monkeypatch.setenv("TELEGRAM_P0_DEEP_MATCH", "1")

    assert codes_from_text(message) == ()


def test_tz147_deep_payment_schedule_anchor_stays_non_p0(monkeypatch: pytest.MonkeyPatch) -> None:
    from mango_mvp.channels.p0_recall_spec import codes_from_text

    monkeypatch.setenv("TELEGRAM_P0_DEEP_MATCH", "1")

    assert codes_from_text("Оплатила вчера, занятия завтра — в системе пока нет") == ()
