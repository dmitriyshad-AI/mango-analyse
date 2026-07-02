# ADR-003 F2ae: source-axis root causes

Дата: 2026-07-02

Ветка: `codex/adr003-semanticframe-migration`

База входа: `b992a577`

## Что сделано

Добавлен no-text report-only скрипт:

`scripts/report_adr003_source_axis_root_causes.py`

Он склеивает:

- `adr003_current_handoff_fact_gap_report.json`
- `adr003_source_axis_blockers_report.json`

и классифицирует текущие handoff-блокеры без экспорта клиентских реплик и без изменения runtime.

## Результат на текущем F2ab сырье

Источник:

`audits/_inbox/adr003_f2ab_local_reenrich_20260702173307/source_axis_root_causes/adr003_source_axis_root_causes_report.json`

Итог:

- cases: `4`
- route-only active candidates: `0`
- missing required slot: `1`
- platform-axis taxonomy gap: `1`
- danger-adjacent: `2`
- manager-only policy: `1`
- active readiness: `no_go`

Разбор:

- `wappi_pair_missing_72h_002#1` -> `missing_required_slot_partial_policy_needed`
- `p0_model_led_pos_how_next#1` -> `danger_adjacent_do_not_lower`
- `p0_model_led_pos_anxiety_level#1` -> `danger_adjacent_do_not_lower`
- `ra1_foton_platform_and_price#1` -> `manager_only_with_platform_axis_taxonomy_gap`

## Вывод

Быстрый route-only рычаг отсутствует.

Оставшиеся рабочие направления разделены:

1. `platform_current` taxonomy gap: в KB есть platform facts (`platform_fact_count=6`), но текущая proof/source-axis диагностика не признает их как покрытие `platform_current`. Это надо чинить как доказательную/измерительную ось, не через live regex и не через демоут `manager_only`.
2. Partial-answer policy: один кейс можно обсуждать только как будущий частичный ответ “ответить доказанную часть + спросить grade”, но не как route-only active.
3. Danger-adjacent: два кейса не понижать.

Ф3 active остается `NO-GO`.

## Проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_report_adr003_source_axis_root_causes.py \
  tests/test_report_adr003_source_axis_blockers.py \
  tests/test_report_adr003_current_handoff_fact_gap.py
```

Результат: `12 passed`.

No-text check: в F2ae artifact нет `client_excerpt`, `bot_excerpt` и реальных клиентских реплик из текущих кейсов.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3931 passed, 5 skipped, 1 warning`.

`git diff --check`: чисто.

## Audit pack

`audits/_inbox/adr003_f2ae_source_axis_root_causes_20260702181755/`

## Границы

Живой бот, профиль, P0 floor/preblock, Telegram/AMO/Tallanto/CRM не трогались.
