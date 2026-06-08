# Phase 12 · TZ-04 — миграция `admission_guarantee` в v2

Снимок: HEAD `36e23cb8`. Приоритет диспетчера: 31. Зависит от: диспетчера.

## 1. Цель
Не обещать поступление в вуз («гарантируете поступление?», «точно поступит?»). Отвечать утверждённым отказом + статистикой поступлений. Юр.-риск (обещание результата по услуге).

## 2. Текущее состояние в legacy
- Текст: `ADMISSION_GUARANTEE_SAFE_TEXT:101` («Мы не даём и не гарантируем поступление… 97% наших учеников поступают…»).
- Вход-триггер: `ADMISSION_GUARANTEE_INPUT_RE:565-573` (`гарантир…поступ/пройд`, `точно…поступ/пройд`, `поступ…точно`).
- Применение: `subscription_llm.py:2597-2599` — flags `admission_guarantee_safe_template_applied` + `placeholder_in_draft`.
- **В v2 НЕ вызывается.**

## 3. Точка вставки в v2
Диспетчер, priority 31 (сразу за result_guarantee). Шаблон-отказ, исключение из price-cluster-skip (применяется поверх unsupported_promise).

## 4. Зависимости (KB)
Текст — константа. Статистика «97% поступают» — внутри текста (не отдельный факт). **Новых KB-фактов не нужно.** Блок Г: OK. (Проверить: «97%» в тексте не должно триггерить unsupported_promise — это часть утверждённого safe-текста; добавить в `_is_verified_safe_numeric_template` whitelist, см. открытый вопрос.)

## 5. Точная правка
```python
_ADMISSION_GUARANTEE_SPEC = TemplateSpec(
    name="admission_guarantee", priority=31,
    produce=lambda r, cm, ctx: ADMISSION_GUARANTEE_SAFE_TEXT if ADMISSION_GUARANTEE_INPUT_RE.search(cm or "") else "",
    route_on_apply="draft_for_manager",
    flag="admission_guarantee_safe_template_applied",
    checklist="Не гарантировать поступление: только программа и статистика.",
)
# placeholder_in_draft флаг как legacy :2597. Исключение из PRICE_CLUSTER-skip.
# ADMISSION_GUARANTEE_SAFE_TEXT добавить в _is_verified_safe_numeric_template (:1631), чтобы «97%» не ловился unsupported_promise.
```

## 6. Регрессы
Positive:
1. `«гарантируете поступление в МФТИ?»` → ADMISSION_GUARANTEE_SAFE_TEXT, draft_for_manager.
2. `«точно поступит после ваших курсов?»` → отказной шаблон.
3. `«вы гарантируете, что пройдём?»` → отказной шаблон.
Контрольные negative:
4. `«какой процент поступает?»` (статистика) → можно привести «97%», не блокировать как обещание.
5. `«как поступить на ваш курс?»` (запись, не вуз) → НЕ admission_guarantee, обычный путь.
6. Сам ADMISSION_GUARANTEE_SAFE_TEXT (содержит «97%») → НЕ режется unsupported_promise (whitelist).
7. `«поможете подготовиться к поступлению?»` без «гарантир/точно» → мягкий ответ, не обязательно шаблон-отказ.

## 7. Backward compatibility
- Legacy admission_guarantee тесты зелёные.
- «97%» в safe-тексте проходит (whitelist), не ловится как unsupported.
- unsupported_promise на реальные выдумки — работает.

## 8. Открытые вопросы Кодексу
1. Добавить `ADMISSION_GUARANTEE_SAFE_TEXT` и `RESULT_GUARANTEE_SAFE_TEXT` в `_is_verified_safe_numeric_template:1631` (иначе «97%»/«25 баллов» в safe-тексте поймает unsupported_promise). Подтвердить.
2. Граница «гарантия поступления» vs «помощь в подготовке» — триггер только на `ADMISSION_GUARANTEE_INPUT_RE`, не на любое «поступление».
