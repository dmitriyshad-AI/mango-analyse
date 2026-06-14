# ТЗ-104 — сведение веток в канон, этап 1

Дата: 2026-06-14
Исполнитель: Codex

## Что влито

1. `codex/tz102-model-p0-direct` -> `main`
   - Способ: fast-forward.
   - Принесло ТЗ-102 и уже включённый в эту ветку ТЗ-26 (`bd2e49c`).
   - Конфликты: нет.

2. `codex/tz25-graphify-structural` -> `main`
   - Способ: обычный merge-коммит `63550948`.
   - Конфликты: нет.
   - Добавлены структурные Graphify-файлы, тесты и отчёт Graphify.

Не вливали: `codex/d6-autonomy-sim` (этап 2, отдельная работа Кодекса-2).

## Проверки

После влива ТЗ-102/ТЗ-26:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3209 passed, 2 skipped, 1 warning in 46.62s
```

После влива Graphify:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3216 passed, 2 skipped, 1 warning in 48.76s
```

Рост с 3209 до 3216 связан с добавлением `tests/test_graphify_structural.py`.

## Graphify-карта

Карта пересобрана на новой вершине `main`.

```text
Command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/graphify_structural_build.py
Revision: 63550948b963a9b79454481fe38ada35144a4c91
Output: /Users/dmitrijfabarisov/Projects/Mango analyse_graphify_structural/output/graphify-out
Summary: /Users/dmitrijfabarisov/Projects/Mango analyse_graphify_structural/structural_build_summary.json
Reproducible: true
Graph: 8871 nodes, 34561 edges
Graphify pin: graphify 0.8.39, commit fd470faeee16e9f42e3f47204824a2002a1f899c
```

## Флаги

Проверено по `src/mango_mvp/channels/subscription_llm_parts/support.py`:

- `TELEGRAM_DIRECT_PATH_MODEL_P0` объявлен, но не входит в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- `TELEGRAM_DEAL_ACTION_DECISION` объявлен, но не входит в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.

Значит оба флага остаются default OFF и не добавлены в `pilot_gold_v1`.

## Финальная вершина

```text
main = 63550948
```

Служебные untracked-файлы очереди и аудита (`tasks/_inbox_codex/*`, новые `D1_audit_backlog/*`) не трогались и не коммитились.
