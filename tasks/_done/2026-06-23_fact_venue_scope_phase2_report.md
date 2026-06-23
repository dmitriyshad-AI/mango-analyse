# Fact venue scope Phase 2 report

Дата: 2026-06-23

Ветка: `codex/fact-venue-scope-phase2`

Рабочее дерево: `/Users/dmitrijfabarisov/Projects/Mango_fact_venue_scope_phase2`

Audit pack для Claude #1: `audits/_inbox/fact_venue_scope_phase2_20260623/`

## Live-изоляция

- Live PID `41798` работает из `/Users/dmitrijfabarisov/Projects/Mango_live_a9f80ba_combo`.
- Live HEAD фактически: `914f9af`, ветка `codex/fact-venue-scope`.
- Ожидаемый Дмитрием `a9f80ba` не совпал с фактическим live HEAD.
- Live-дерево не переключалось и не изменялось.
- Код Фазы 2 велся в отдельном worktree.

## Реализация

- Флаг `TELEGRAM_FACT_VENUE_SCOPE`, default OFF, не в `pilot_gold_v1`.
- LLM-селектор direct path под флагом возвращает `requested_scope`.
- В fact pack добавлены структурные оси `venue` и `program_kind`.
- Чужая площадка:
  - удаляется, если в выбранном паке есть факт нужной площадки;
  - понижается в adjacent и помечается, если нужного факта нет;
  - не трогается при `requested_scope=unspecified`.
- Verifier `wrong_intent_fact` под флагом сравнивает структурные venue-метки.
- Camp/regular конфликт в direct path под флагом использует `program_kind`, а не поиск слов в тексте факта.

## Проверки

```text
tests/test_fact_venue_scope.py
7 passed
```

```text
tests/test_fact_venue_scope.py tests/test_subscription_llm_draft_provider.py -k "llm_retrieve or wide_pack or wrong_intent_fact or pilot_gold_profile"
19 passed, 494 deselected
```

```text
Full pytest
3606 passed, 5 skipped, 1 warning
```

`git diff --check`: passed.

## Ограничения

- Фаза 3 не запускалась.
- Флаг в пилотный профиль не добавлялся.
- Live bot, AMO, Tallanto, `stable_runtime` не трогались.
