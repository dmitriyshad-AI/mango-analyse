# TZ-121 Block A Deal Gold Shadow

Дата: 2026-06-16  
Ветка: `codex/tz121-group4-remaining`

## Что сделано

- Собран offline gold-замер по 24 закрытым сделкам из сохраненного TZ116 snapshot.
- Новых обращений к CRM/AMO/Tallanto нет.
- Новых вызовов Codex/модели нет: использованы ранее сохраненные TZ116 shadow-результаты.
- A primary не включался.
- Writeback не включался.

## Артефакт

`audits/_inbox/tz121_a_deal_gold_shadow_20260616/`

Внутри:

- `deal_a_gold_labels.csv` — малый conservative-gold для регрейда;
- `tz121_a_deal_gold_trace.csv`;
- `tz121_a_deal_gold_trace.jsonl`;
- `summary.json`;
- `REPORT.md`;
- `semantic_review.md`.

## Счетчики

- строк: `24`;
- бренды: `foton=12`, `unpk=12`;
- rule exact vs gold: `22/24` = `0.9167`;
- model exact vs gold: `23/24` = `0.9583`;
- delta model-rule: `+1`;
- error types: `both_correct=22`, `model_fix=1`, `both_wrong=1`;
- high-confidence wrong model rows: `1`;
- `llm_calls_total`: `0`.

## Смысловой вывод

Gold-разметка сделана консервативно: если сделка закрыта как архив/нет связи, а в досье нет истории касаний, автоматический вывод «нужно follow-up» или «закрыто рано» считается недостаточно надежным; безопасный эталон — `manual_review`.

На таком эталоне модель улучшает один случай, но один уверенный модельный вывод остается неверным. Поэтому A primary не включен и должен ждать регрейда Claude/Dmitry по raw trace.

Semantic status: `PASS_WITH_NOTES`, потому что замер полезен и безопасен, но набор маленький и не является разрешением на primary.

## NEG / границы

- primary режим запрещен тестом;
- live CRM read не выполнялся;
- AMO/Tallanto/CRM/DB write не выполнялся;
- OpenAI API key не используется;
- raw PII в git не добавлялась;
- tracked-отчет содержит только агрегаты и обезличенные классы ошибок.

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_tz121_deal_a_gold_measure.py

2 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_deal_a_gold_measure.py \
  --out-dir audits/_inbox/tz121_a_deal_gold_shadow_20260616 \
  --write-gold

model_exact_vs_gold=23/24
rule_exact_vs_gold=22/24
llm_calls_total=0
primary_run=false
stop_for_regrede=true
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3297 passed, 5 skipped, 1 warning
```
