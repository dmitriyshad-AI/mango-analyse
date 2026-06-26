# TZ137 direct slots/fallback manual port

Дата: 2026-06-27

Источник порта: `589009e5110fbd2fba22e962867cdafbd90f568b` (`codex/tz137-adr002-direct-slots-fallback`).

Целевая база: `main@e55262b57773`.

## Что сделано

- Сохранены вне-git артефакты `tz137` в `/Users/dmitrijfabarisov/Claude Projects/Foton/_archive_tz137/`:
  - `2026-06-17_TZ137_behavior_micro_measure_D3_report.md`
  - `runs/tz137_behavior_micro_20260617/`
- Создан и запушен тег `archive/tz137-adr002-direct-slots-fallback` на `589009e5110fbd2fba22e962867cdafbd90f568b`.
- Ручной порт A/B/C выполнен без merge/cherry-pick старой ветки.
- Все новые флаги default OFF и не добавлены в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.

## Перенесённые флаги

### A: `TELEGRAM_DIRECT_PLAN_KNOWN_SLOTS`

- `conversation_intent_plan` под флагом ограничивает `do_not_reask_slots` подтверждёнными slot provenance / `client_confirmed_slots`.
- direct path под флагом читает `conversation_intent_plan.known_slots`, fallback к legacy `slots` сохранён.

### B: `TELEGRAM_DIRECT_KEYWORD_FALLBACK_RELEVANCE`

- keyword fallback больше не берёт широкий top-N без позитивной связи с вопросом.
- route-rubric получает `fact_pack` и может регенерировать пустой handoff при `empty_selection` / `timeout` и открытом вопросе.
- добавлен `apply_direct_keyword_fallback_reask_layer`: переводит безопасный пустой handoff в короткий уточняющий вопрос, но не срабатывает на P0/high-risk flags.

### C: `TELEGRAM_DIRECT_SLOT_TOPIC_SHADOW`

- добавлен observe-only shadow extractor слотов/темы.
- результат пишется только в `direct_path.slot_topic_shadow`.
- при ошибке/таймауте фиксируется `fallback_reason`; текст и маршрут не меняются.

## Проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile \
  src/mango_mvp/channels/conversation_intent_plan.py \
  src/mango_mvp/channels/subscription_llm_parts/direct_path.py \
  src/mango_mvp/channels/subscription_llm_parts/provider.py \
  tests/test_conversation_intent_plan.py \
  tests/test_subscription_llm_draft_provider.py
```

Результат: passed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_conversation_intent_plan.py \
  tests/test_subscription_llm_draft_provider.py
```

Результат: `573 passed in 59.70s`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3669 passed, 5 skipped, 1 warning in 84.89s`.

```bash
git diff --check
```

Результат: clean.

## Остаточные риски

- Это `formal_pass`, не `semantic_pass`.
- Флаги A/B/C не включать без регрейда Claude #1 и отдельного решения Дмитрия.
- По микро-замеру 2026-06-17 флаг B выглядит самым перспективным; A требует ручного разбора P0-текста и `assumed_unstated_need`; C должен оставаться тенью до отдельного байтового/parity регрейда.

