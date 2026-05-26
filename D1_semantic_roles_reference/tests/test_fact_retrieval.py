from __future__ import annotations

"""Тесты слоя извлечения: факт-ответ ДОЛЖЕН попадать в confirmed_facts.

Кандидаты синтетические, но повторяют реальные провалы round-5 (recall=0/8):
адрес Москвы под locations_*.address, скидка «за год 14%» под payment_options...,
follow-up без маркеров, чужая область (выезд при городском) — исключается.

Запуск: python3 D1_semantic_roles_reference/tests/test_fact_retrieval.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)

from fact_retrieval import select_confirmed_facts, key_matches  # noqa: E402
from held_state import HeldState, update_held                    # noqa: E402
from semantic_roles import tag_message_roles                      # noqa: E402

_PASS = 0
_FAIL = 0
_FAILS: list[str] = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILS.append(f"[FAIL] {name}: {detail}")


def fk_in(result, needle):
    return any(needle in str(f.get("fact_key") or "") for f in result)


# Синтетическая база УНПК/Фотон (имитирует ключи рантайма)
UNPK = [
    {"fact_key": "locations_unpk.addresses.1.address", "brand": "unpk", "scopes": set(), "text": "УНПК: Сретенка, 20."},
    {"fact_key": "locations_unpk.addresses.0.address", "brand": "unpk", "scopes": set(), "text": "УНПК: Долгопрудный, Институтский 9."},
    {"fact_key": "payment_options.available_schedules.3.year.discount_extra", "brand": "unpk", "scopes": set(), "text": "УНПК: за год скидка 14%."},
    {"fact_key": "discounts_multichild_condition_client_text", "brand": "unpk", "scopes": {"discount_multichild"}, "text": "многодетным по удостоверению."},
    {"fact_key": "discounts_stacking_rule", "brand": "unpk", "scopes": {"discount_stacking"}, "text": "скидки не суммируются."},
    {"fact_key": "matkap.sfr_review_timing", "brand": "unpk", "scopes": {"matkap_process"}, "text": "СФР рассматривает до 10 рабочих дней + до 5."},
    # шум для проверки капа
    *[{"fact_key": f"noise.fact_{i}", "brand": "unpk", "scopes": set(), "text": f"шум {i}"} for i in range(15)],
]
FOTON_CAMP = [
    {"fact_key": "ls_city_2026_foton.schedule", "brand": "foton", "scopes": {"city_day_camp"}, "text": "городская летняя школа, без проживания, пн-пт."},
    {"fact_key": "lvsh_mendeleevo_foton.lodging", "brand": "foton", "scopes": {"residential_lvsh"}, "text": "выездная ЛВШ Менделеево, проживание, 5-раз питание."},
]


def main() -> int:
    # 1. Адрес Москвы: факт-ответ под locations_*.address должен попасть (был recall-провал)
    r = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["locations.current"], k=10)
    check("address_recall", fk_in(r, "locations_unpk.addresses.1.address"), f"в результате нет Сретенки; {[f['fact_key'] for f in r][:5]}")

    # 2. Скидка за год 14%: лежит под payment_options..., запрос discounts.current — алиас матчит
    check("alias_discount", key_matches("discounts.current", "payment_options.available_schedules.3.year.discount_extra"), "алиас discount не сработал")
    r = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["discounts.current"], k=10)
    check("year_discount_recall", fk_in(r, "year.discount_extra"), f"нет факта 14%; {[f['fact_key'] for f in r][:6]}")

    # 3. Кап 10 + шум: факт-ответ всё равно внутри (гарантия), даже если шум вытеснял бы
    check("answer_survives_cap", fk_in(r, "year.discount_extra"), "факт-ответ срезан капом")

    # 4. Факт без scope НЕ отбрасывается (Сретенка scopes=set()) — recall, не drop
    check("scopeless_not_dropped", fk_in(select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["locations.current"]), "addresses.1"), "scopeless факт отброшен")

    # 5. Чужой бренд никогда: фотоновский факт не попадёт в выдачу УНПК
    mixed = UNPK + [{"fact_key": "foton.price", "brand": "foton", "scopes": set(), "text": "Фотон цена"}]
    r = select_confirmed_facts(mixed, active_brand="unpk", required_fact_keys=["prices.current"])
    check("no_cross_brand", not any(f.get("brand") == "foton" for f in r), "просочился чужой бренд")

    # 6. Точность сохранена: при городском лагере выездной факт ИСКЛЮЧЁН (blocked)
    r = select_confirmed_facts(FOTON_CAMP, active_brand="foton", required_fact_keys=["programs.current"],
                               active_topics=["camp"], blocked_scopes=["residential_lvsh"])
    check("foreign_scope_blocked", not fk_in(r, "lvsh_mendeleevo"), "выездной факт просочился при городском вопросе")
    check("city_fact_kept", fk_in(r, "ls_city_2026_foton"), "городской факт потерян")

    # 7. Follow-up: тема извлечения держится через held (бид без маркеров не обнуляет тему)
    held = HeldState()
    roles0 = tag_message_roles("если оплатить за год, скидка будет?")
    held = update_held(held, "если оплатить за год, скидка будет?", roles0, p0_required=False,
                       required_fact_keys=("discounts.current",))
    roles1 = tag_message_roles("а за семестр?")  # bare follow-up, без 'скидк'
    held = update_held(held, "а за семестр?", roles1, p0_required=False,
                       required_fact_keys=tuple(r for r in () ))  # пусто на текущем ходу
    rc = held.retrieval_context()
    check("followup_keeps_keys", "discounts.current" in rc["required_fact_keys"], f"тема извлечения потеряна: {rc['required_fact_keys']}")
    r = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=rc["required_fact_keys"])
    check("followup_recall", fk_in(r, "year.discount_extra"), "на follow-up факт-ответ не извлёкся")

    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for f in _FAILS:
        print(f)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
