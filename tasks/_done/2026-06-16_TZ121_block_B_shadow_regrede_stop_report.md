# TZ-121 Блок B: исход сделки, shadow на микро-наборе

Дата: 2026-06-16

## Статус

Выполнен только блок B до точки остановки на регрейд.

Блоки E/C/A не начинались, потому что ТЗ требует порядок: реализация -> shadow на микро-наборе -> стоп на регрейд -> primary поблочно.

## Что реализовано

- В `outcome_linker` добавлен режим `outcome_model_mode=off|shadow|primary`.
- `off` сохраняет старое legacy-поведение.
- `shadow` считает negation-aware кандидата и пишет сравнение, но финальный исход не меняет.
- `primary` технически ограничен allowlist-правилом: применяет только flip `won_paid_or_active -> known_student_or_lead`.
- Flip `won_paid_or_active -> payment_pending` измеряется, но не применяется в primary.
- Добавлена защита от протекания отрицания через тире: `Не отказались - оплатили...` теперь сохраняет оплату.
- Добавлен shadow-only runner `scripts/run_tz121_outcome_b_micro_shadow.py`.
- Добавлен микро gold-набор `tests/fixtures/tz121_outcome_b_micro_gold.csv` без ПДн.

## Shadow-замер

Артефакты для регрейда:

- `audits/_inbox/tz121_b_outcome_micro_shadow_20260616/summary.json`
- `audits/_inbox/tz121_b_outcome_micro_shadow_20260616/tz121_b_outcome_trace.csv`
- `audits/_inbox/tz121_b_outcome_micro_shadow_20260616/tz121_b_outcome_trace.jsonl`
- `audits/_inbox/tz121_b_outcome_micro_shadow_20260616/REPORT.md`

Счётчики:

- строк: `10`
- allowed flip `won_paid_or_active -> known_student_or_lead`: `2`
- allowed flip correct: `2`
- allowed flip wrong: `0`
- `won_paid_or_active -> payment_pending`: `1`, primary заблокирован
- model_break: `0`
- error types: `{"model_fix": 5, "both_correct": 5}`
- `llm_calls_total=0`

## Безопасность

- Только синтетический микро-набор.
- Полные наборы не запускались.
- Модель не вызывалась.
- AMO/Tallanto/CRM не трогались.
- БД и `stable_runtime` не читались и не писались.
- ASR не запускался.
- Primary не запускался.

## Проверки

Точечно:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_outcome_linker.py tests/test_tz121_outcome_b.py
```

Результат: `15 passed`.

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3288 passed, 5 skipped, 1 warning`.

Shadow:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_outcome_b_micro_shadow.py --out-dir audits/_inbox/tz121_b_outcome_micro_shadow_20260616
```

Результат: см. счётчики выше.

## Стоп

Остановиться на регрейд Claude/Dmitry. До регрейда:

- не включать B primary в рабочих прогонах;
- не переходить к E;
- не запускать полные сеты.
