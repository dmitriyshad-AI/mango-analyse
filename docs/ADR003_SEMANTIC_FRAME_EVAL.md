# ADR-003 SemanticFrame Shadow Eval

Дата: 2026-06-30.

Этот документ фиксирует стартовый eval для ADR-003, этап 0/1. Цель этапа: измерять единый `SemanticFrame` в shadow-режиме без изменения маршрута и текста ответа.

## Набор

- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_wappi_latest25_20260701.jsonl` — 25 маскированных Wappi what-if кейсов `pair_missing` за последние 72 часа.
- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_m1_scenarios_20260701.jsonl` — runnable M1-сценарий: 25 Wappi + existing P0/brand/product/forward-payment/reliable/closing/tz147 регрессии.
- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_eval_manifest_20260701.json` — manifest с sha256 набора и существующих регрессий.
- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_baseline_20260701.json` — route/text baseline из уже снятого what-if артефакта.

Builder:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_adr003_semantic_frame_eval.py
```

## Мораторий

Новый провал понимания клиента сначала добавляется как eval-case в этот или следующий версионированный набор. Нельзя добавлять новый regex-детектор, SAFE_TEXT, флаг понимания или отдельную ветку маршрутизации без ADR/review и регрессионного кейса.

## Shadow-Gates

- `TELEGRAM_SEMANTIC_FRAME_SHADOW` default OFF.
- ON-режим не имеет права менять `route`, `draft_text`, `safety_flags`, `manager_checklist` и P0-пол.
- SemanticFrame должен приходить в существующем direct-path payload, без отдельного модельного вызова.
- M1-регрейд сравнивает OFF/ON на этом наборе: diff финальных черновиков = 0, extra model calls = 0, `frame.must_handoff` vs фактический P0 >= 95%.

## Baseline

Для 25 Wappi what-if кейсов на 2026-07-01:

- `bot_answer_self_for_pilot`: 12
- `draft_for_manager`: 10
- `manager_only`: 3

Это не semantic pass. Это зафиксированная точка сравнения для M1-регрейда и Claude #1 по сырью.

Важно: Wappi latest25 пока не имеет ручного `expected_frame` gold. Набор годится для route/text no-op, model-call count и coverage телеметрии. Точность SemanticFrame по полям можно заявлять только после отдельной gold-разметки и регрейда по сырью.
