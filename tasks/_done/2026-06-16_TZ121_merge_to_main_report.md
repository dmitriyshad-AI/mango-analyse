# TZ-121 Merge To Main Report

Дата: 2026-06-16  
Ветка назначения: `main`  
Источник: `codex/tz121-group4-remaining`  
Merge commit: `96b6c5202e02ffb58d6af73f47d600ba9715b7eb`

## Что влито

В `main` влит финальный регрейд `PASS` для TZ-121 после rebase.

Фактический diff merge:

- `src/mango_mvp/insights/outcome_linker.py`;
- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`;
- `src/mango_mvp/customer_timeline/canonical_readonly_triage.py`;
- offline runner scripts `scripts/run_tz121_*`;
- fixtures/tests `tests/test_tz121_*`, `tests/fixtures/tz121_*`;
- отчеты TZ-121 в `tasks/_done/`.

Живой путь Telegram-бота не менялся.

## Итог Группы 4

- B — `primary` в офлайн-режиме: только allowlist flip `won_paid_or_active -> known_student_or_lead`; `payment_pending` не применять.
- E — `primary` в офлайн-режиме: `cyrillic_v2`, Foton по корню, cross-brand/unknown fail-closed.
- C — `hybrid primary` для офлайн-классификации; основной каталог не пересобирался.
- A — оставлен `shadow`; primary не включён, потому что `23/24` у модели против `22/24` у правила = шум + `1` уверенная ошибка модели.
- D — статус primary зафиксирован в отдельном TZ-118 отчёте/ветке (`ce836b0`). Этот merge содержит только подтверждённый TZ-121 diff.

## Безопасность

- AMO/Tallanto/CRM write: `0`.
- ASR: не запускался.
- OpenAI API key: не используется.
- `stable_runtime`: не менялся.
- Raw PII/data artifacts: не добавлялись в git.

## Проверки

Перед merge:

- `origin/main` = `dd00d65efe3c0d465539e31496a7875cb8b92063`;
- `codex/tz121-group4-remaining` = `6610c2c8e1298c83eb5688f376edf571bd0babbf`;
- conflict: none.

После merge:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3304 passed, 5 skipped, 1 warning
```
