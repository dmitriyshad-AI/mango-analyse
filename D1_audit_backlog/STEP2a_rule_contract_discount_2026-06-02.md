# Контракт правила (Шаг 2a): «скидки». 2026-06-02.

Автор: Клод 1. По эталону `STEP2a_rule_contract_installment_ETALON`. Источник — `_discount_safe_template`
(subscription_llm 92 LOC) + 8 текст-констант (106-126). Прочитано по сырью. ОТВЕЧАЕТ на претензию аудитора
«стэкинг=argmax=мини-Prolog, многодетная=неотделимая логика»: показываю, что это ДАННЫЕ + одна функция `max`.

## Как сейчас (лазанья)
92 строки мешают: бренд + распознавание подвида (второй предмет / многодетная / стэкинг / МФТИ) по
ключевым словам + формат (онлайн/очно) + scope-различение (second_subject vs multichild) + правило
«не суммируются» + выбор из 8 захардкоженных (часто КОМБИНИРОВАННЫХ) текстов.

## Целевой контракт (слой 2 = ДАННЫЕ)

```yaml
rule_id: discount
title: Скидки (второй предмет, многодетная, МФТИ, стэкинг)
intent: discount_inquiry              # planner отдаёт intent + subvariant + uncertainty
intent_subvariants: [second_subject, multichild, mfti_employee, stacking, forced_competitor]
required_fact_keys:
  - discount.foton.second_subject     # онлайн 30 / очно 20
  - discount.unpk.second_subject      # 20 (второй и последующий, один ребёнок)
  - discount.multichild               # 10, по удостоверению
  - discount.unpk.mfti_employee       # 10, документ с работы
  - discount.stacking_rule            # не суммируются, наибольшая
# ДАННЫЕ ПРАВИЛА (структура, не предложения)
data:
  second_subject:
    foton: {online_pct: 30, offline_pct: 20, per_child: true}
    unpk:  {pct: 20, per_child: true, applies: "второй и последующий предмет"}
  multichild: {pct: 10, by_family_status: true, requires: "удостоверение"}   # по СТАТУСУ, не числу детей
  mfti_employee: {pct: 10, requires: "документ с места работы"}
  stacking: {summable: false, rule: take_max}        # «не суммируются, наибольшая»
  promocodes: {in_client_layer: false}               # промокоды НЕ в клиентском слое
# ЕДИНСТВЕННАЯ ЧИСТАЯ ФУНКЦИЯ (ответ аудитору: это не Prolog)
pure_functions:
  best_discount: "max(applicable_pcts)"   # одна строка: наибольшая из применимых. Не движок правил.
brand_split: true                          # Фотон 30/20 vs УНПК 20 — разные данные
blocking_conditions:
  - forced_competitor        # «дайте скидку как у конкурента» → менеджер (FORCED_DISCOUNT)
  - promocode_request        # промокод → не выдавать в клиентском слое (гейт)
  - asks_exact_personal_calc # точный расчёт под ситуацию → менеджер проверит
route_effect: bot_answer_self            # если факт есть и не P0/forced; иначе manager
text_effect: composer_generates_from_data  # 8 текст-констант ИСЧЕЗАЮТ; composer пишет из data+subvariant.
#   Комбинированный случай (второй предмет + многодетная) = composer применяет best_discount и объясняет
#   «выгоднее наибольшая», из ДАННЫХ, а не из захардкоженного абзаца.
preserve_exceptions:
  - plan_scope discount_second_subject ⟂ discount_multichild — НЕ смешивать подвиды (planner-disambiguation)
  - «физика+математика» в одном вопросе → подвид second_subject (два предмета)
  - многодетная: данное by_family_status=true → «по статусу семьи», НИКОГДА не по числу детей в базе
  - стэкинг: всегда оговаривать take_max, если применимо несколько оснований
  - формат: для Фотон second_subject — онлайн 30 / очно 20 (не путать)
```

## Распределение по слоям
- **Planner:** `intent=discount_inquiry` + subvariant + (формат онлайн/очно) + uncertainty. Заменяет
  keyword-детекцию (второй предмет/многодет/суммир/сотрудник) и regex «физика+математика».
- **Слой 2:** `data` выше + `best_discount = max(...)`. Изменилась скидка (30→25%) — правится ДАННОЕ.
- **Composer:** пишет ответ из data под subvariant; комбинированный случай — применяет `best_discount` и
  объясняет «наибольшая». 8 текст-констант (включая комбинированные абзацы) УДАЛЯЮТСЯ.
- **Гейт:** forced_competitor → manager; промокод-утечка → block; бренд-утечка (скидка чужого бренда) →
  block; обещание точного расчёта без факта → downgrade.

## Ответ аудитору (претензия 3 спора)
«Скидки не суммируются → argmax → мини-Prolog» — НЕТ: это `data: {summable:false, rule:take_max}` +
`best_discount=max(applicable)`. Одна строка над данными, не интерпретатор правил. «Многодетная = логика
скрытых условий» — это данное `by_family_status:true`, не код. То есть скидки ОТДЕЛИМЫ в данные+1 функцию.

## Обязательные НЕГАТИВНЫЕ тесты
1. Стэкинг: «скидки сложатся?» → «не суммируются, наибольшая» (take_max, НЕ сумма).
2. Многодетная: «у меня трое детей» → 10% по СТАТУСУ/удостоверению, НЕ «×3» и НЕ по числу детей.
3. Бренд: Фотон второй онлайн → 30%; УНПК второй → 20% — НЕ перепутать (бренд-различие).
4. Формат Фотон: второй ОЧНЫЙ → 20%, не 30% (формат).
5. «скидку как у конкурента» → менеджер (forced), НЕ выдумывать скидку.
6. Промокод → НЕ выдавать в клиентском слое (гейт).
7. Комбо: второй предмет + многодетный → объяснить наибольшую (best_discount), не сложить.
8. ПОЗИТИВ: «есть скидка на второй предмет?» → бренд+формат верно, живо, route=answer_self.

## Вывод
Скидки мигрируют так же чисто, как рассрочка: данные + одна функция max + planner-намерение + composer.
92-строчная функция и 8 текст-констант удаляются. Следующее семейство по эталону: цены.
