# D1 intent_model_led clean main branch

Дата: 2026-06-26

## Задача

Прямой merge `codex/intent-model-led` был небезопасен: ветка стояла поверх
`4caa5eb` и тянула venue/autonomy слой, `fact_venue_scope.py` и большую KB-дельту.
Нужно было выделить чистый intent-only перенос поверх `main@a9f80ba`.

## Что сделано

- Создан clean worktree:
  `/Users/dmitrijfabarisov/Projects/Mango_intent_model_led_main_clean`
- Ветка:
  `codex/intent-model-led-main-clean`
- База:
  `main@a9f80ba`
- Перенесены только два intent-коммита:
  - `d35f662 Add model-led intent guard for direct path` -> `8e32575`
  - `5d20bb1 Enable intent model-led via pilot profile` -> `80b74e7`
- Конфликты resolved вручную:
  - `pilot_profile_runtime.py`: оставлен только `intent_model_led`, без
    `FACT_VENUE_SCOPE` / `AUTONOMY_SCOPE_PRECISION`.
  - `subscription_llm_parts/__init__.py`: экспортируются только
    `INTENT_MODEL_LED_ENV` и `_intent_model_led_enabled`, без venue.
  - `tests/test_telegram_public_pilot_bots.py`: оставлен selfcheck-тест
    intent-флага из `pilot_gold_v1`, release-flag тесты venue/autonomy не
    переносились.

## Diff-scope

`main..HEAD`:

- 13 файлов
- нет `src/mango_mvp/channels/fact_venue_scope.py`
- нет KB-файлов в diff
- `FACT_VENUE_SCOPE` / `AUTONOMY_SCOPE_PRECISION` встречаются только в старых
  report-файлах как описание исходного замера, не в коде.

## Проверки

Целевые:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py \
  tests/test_telegram_dynamic_client_sim.py \
  tests/test_telegram_public_pilot_bots.py
# 689 passed in 54.74s
```

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
# 3612 passed, 5 skipped, 1 warning in 90.14s
```

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Чистая ветка больше не тянет venue/autonomy и KB.
- Intent-флаг закреплён в `pilot_gold_v1`, то есть не потеряется при чистом
  профильном запуске.
- Видимость `model_intent` сохранена в transcript/CSV-слое из исходной ветки.
- OFF-путь сохраняет keyword fallback.

Notes:

- Этот отчёт не является финальным поведенческим регрейдом Claude #1.
- `main` не двигался.
- Live/stable_runtime/AMO/Tallanto не трогались.
