# TZ-121 Block C Primary Enable

Дата: 2026-06-16  
Ветка: `codex/tz121-group4-remaining`

## Что включено

- После регрейда `PASS` включен `primary`-режим для офлайн-классификации вопросов.
- Основной каталог не пересобирался.
- Live-путь бота не менялся.
- Записей в DB/CRM/AMO/Tallanto нет.

## Важное уточнение по гибриду

Включен именно гибрид, прошедший shadow-регрейд `80/100, +8/-0`:

- по умолчанию берется сохраненный Codex-класс;
- для 8 разобранных регрессий ТЗ-116 остается старое правило;
- служебный guard сохранен только для реально уверенных служебных срабатываний.

Буквальная стратегия «оставлять все служебные правила» ранее давала `65/100`, поэтому не включалась как primary, чтобы не внести подтвержденную регрессию.

## Артефакт primary

`audits/_inbox/tz121_c_question_catalog_hybrid_primary_20260616/`

Счетчики:

- строк: `100`;
- rule vs gold: `37/100`;
- model vs gold: `72/100`;
- hybrid primary vs gold: `80/100`;
- guard `followup_regression`: `8`;
- `llm_calls_total`: `0`;
- `primary_run`: `true`;
- `rebuilds_main_catalog`: `false`.

## Измененные файлы

- `scripts/run_tz121_question_catalog_c_hybrid_shadow.py`
- `tests/test_tz121_question_catalog_c_hybrid_shadow.py`
- `tasks/_done/2026-06-16_TZ121_block_C_primary_enable_report.md`

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_tz121_question_catalog_c_hybrid_shadow.py

2 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_question_catalog_c_hybrid_shadow.py \
  --mode primary \
  --out-dir audits/_inbox/tz121_c_question_catalog_hybrid_primary_20260616

hybrid_vs_gold.correct=80
primary_run=true
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3295 passed, 5 skipped, 1 warning
```

## Следующий шаг

Блок A: gold-разметка 24 сделок и измерение модель vs эвристика. A primary не включать до регрейда.
