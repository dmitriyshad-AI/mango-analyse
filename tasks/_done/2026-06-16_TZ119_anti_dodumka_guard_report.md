# TZ-119 anti-dodumka guard report

## Итог

ТЗ-119 реализовано на ветке `codex/tz119-assumed-scope-guard`.

## Сделано

- Добавлен флаг `TELEGRAM_ASSUMED_SCOPE_GUARD`, выключен по умолчанию.
- Вернул провенанс слотов в direct path:
  - `confirmed_by_client` только при клиентской цитате;
  - `assumed_from_context` для CRM/контекста/бот-инференса без цитаты.
- Hard scope фактов теперь использует только подтверждённые клиентом параметры при включённом флаге.
- Неподтверждённые параметры остаются как мягкое ранжирование и как явный статус в prompt.
- Draft prompt и retriever prompt получают статус слота при включённом флаге.
- Финальный страж не повышает маршрут, а задаёт уточнение, если ответ утверждает неподтверждённый параметр.
- P0/рисковые ответы страж не трогает.
- `TELEGRAM_RETRIEVER_MODEL_DRIVEN` теперь работает только в связке с `TELEGRAM_ASSUMED_SCOPE_GUARD`.
- Добавлен trace в `dynamic_summary.json`.

## Проверки

- `py_compile`: passed.
- `tests/test_subscription_llm_draft_provider.py -k 'tz110 or tz119 or pilot_gold_v1'`: 16 passed.
- `tests/test_subscription_llm_draft_provider.py`: 471 passed.
- `tests/test_telegram_dynamic_client_sim.py`: 100 passed.
- Full pytest: 3277 passed, 2 skipped, 1 warning.

## Что не запускалось

Ночной динамический замер OFF->ON не запускался. Его нужно сделать отдельно после регрейда Claude #1.

## Audit pack

`audits/_inbox/tz119_assumed_scope_guard_20260616_032725/`
