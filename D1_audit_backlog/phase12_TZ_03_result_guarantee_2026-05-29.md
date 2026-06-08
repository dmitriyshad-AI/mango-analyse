# Phase 12 · TZ-03 — миграция `result_guarantee` в v2

Снимок: HEAD `36e23cb8`. Приоритет диспетчера: 30. Зависит от: диспетчера + развязки с `unsupported_promise`.

## 1. Цель
Не давать обещаний результата/балла («наберёте 100 баллов», «гарантируем результат ЕГЭ»). Отвечать утверждённым отказом + статистикой. Закрывает юр.-риск и класс обещания «N баллов» (пересекается с Волной 3a pattern `\b\d{1,3}\s*балл\w*`).

## 2. Текущее состояние в legacy
- Текст: `RESULT_GUARANTEE_SAFE_TEXT:97` («Мы не даём и не гарантируем конкретный балл… может показать статистику результатов»).
- Вход-триггер: `RESULT_GUARANTEE_INPUT_RE:558-560` (`гарантир…балл`, `(сдаст|балл|результат)…гарантир`, `(90|100)…балл`).
- Применение: `subscription_llm.py:2603-2605` — flags `result_guarantee_safe_template_applied` + `placeholder_in_draft`, метаданные.
- **В v2 НЕ вызывается.** v2 имеет `unsupported_promise` (`:1019`), который ловит число «N баллов», но НЕ даёт отказной шаблон по обещанию.

## 3. Точка вставки в v2
Диспетчер, priority 30 (выше ценовых). result_guarantee — шаблон-ОТКАЗ, должен срабатывать ДАЖЕ поверх `unsupported_promise` (усиливает безопасность). В правиле непересечения price-cluster (Блок Б) result/admission_guarantee — исключение (не блокируются срезанным числом).

## 4. Зависимости (KB)
Текст — константа. Статистика результатов — факт `results_social_proof.*` уже в v6.3 (но осторожно: «N баллов» в нём триггерит pattern Волны 3a — см. связку в Блоке Г и `score_pattern_false_positive_analysis`). **Новых фактов не нужно**, но pattern «N балл*» надо сузить до promise-контекста (отдельная правка Волны 3a, не блокер TZ-03).

## 5. Точная правка
```python
_RESULT_GUARANTEE_SPEC = TemplateSpec(
    name="result_guarantee", priority=30,
    produce=_result_guarantee_produce,  # триггер RESULT_GUARANTEE_INPUT_RE по client_message
    route_on_apply="draft_for_manager",
    flag="result_guarantee_safe_template_applied",
    checklist="Не гарантировать балл/результат: только программа и статистика.",
)
def _result_guarantee_produce(result, client_message, context):
    if RESULT_GUARANTEE_INPUT_RE.search(client_message or ""):
        return RESULT_GUARANTEE_SAFE_TEXT
    return ""
# В _apply: для result_guarantee добавить flag "placeholder_in_draft" (как legacy :2603).
# Развязка: result_guarantee НЕ входит в PRICE_CLUSTER-skip (см. Блок Б) — применяется поверх unsupported_promise.
```

## 6. Регрессы
Positive:
1. `«гарантируете 100 баллов на ЕГЭ?»` → RESULT_GUARANTEE_SAFE_TEXT, route draft_for_manager.
2. `«точно сдаст на 90+?»` → отказной шаблон.
3. `«вы гарантируете результат?»` → отказной шаблон.
Контрольные negative:
4. `«какой у вас средний результат?»` (запрос статистики, НЕ обещания) → бот может привести `results_social_proof` («+25 баллов»), НЕ блокировать как обещание (см. `score_pattern_false_positive_analysis`).
5. `«сколько стоит подготовка к ЕГЭ?»` → НЕ result_guarantee (это цена), идёт обычный путь.
6. Драфт «средний результат выше на 25 баллов» (статистика в rfk) → НЕ unsupported_promise и НЕ перекрыт result_guarantee (контроль политики «статистику можно»).
7. `«сдаст ли мой ребёнок?»` без «гарантир/балл» → мягкий ответ, не обязательно шаблон.

## 7. Backward compatibility
- Legacy result_guarantee тесты зелёные.
- `unsupported_promise` на реальные выдумки чисел — продолжает срабатывать.
- Статистика результатов остаётся разрешённой (не зарезать вместе с обещаниями).

## 8. Открытые вопросы Кодексу
1. Pattern Волны 3a `\b\d{1,3}\s*балл\w*` слишком широк (блокирует статистику «+25 баллов») — сузить до promise-контекста ДО/вместе с TZ-03 (иначе конфликт: result_guarantee разрешает статистику, а unsupported_promise её режет). См. `score_pattern_false_positive_analysis_2026-05-29.md`.
2. Нужен ли `placeholder_in_draft` флаг в v2 (legacy ставит его) — влияет ли на downstream.
