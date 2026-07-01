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
- `TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW` default OFF. Это предпочтительный Stage 1 режим после smoke 2026-07-01: frame считается отдельным post-hoc shadow-вызовом после финального черновика и пишет только metadata.
- ON-режим не имеет права менять `route`, `draft_text`, `safety_flags`, `manager_checklist` и P0-пол.
- Same-payload `TELEGRAM_SEMANTIC_FRAME_SHADOW` оставлен как legacy/investigation path: малый smoke показал, что добавление frame-поля в основной draft prompt может менять текст.
- M1-регрейд сравнивает OFF/paired-enriched ON на этом наборе: diff финальных черновиков = 0, все добавочные model calls = только `bot_semantic_frame_shadow`, `frame.must_handoff` vs фактический P0 >= 95%.

## OFF/ON Report

После M1-прогона отчёт собирается из сохранённых `dynamic_dialog_transcripts.jsonl` и `dynamic_summary.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/report_adr003_semantic_frame_eval.py \
  --off-transcripts <OFF_RUN>/dynamic_dialog_transcripts.jsonl \
  --off-summary <OFF_RUN>/dynamic_summary.json \
  --on-transcripts <ON_RUN>/dynamic_dialog_transcripts.jsonl \
  --on-summary <ON_RUN>/dynamic_summary.json \
  --out-dir <REPORT_DIR>
```

Отчёт проверяет route/text no-op, delta модельных вызовов, coverage/schema `SemanticFrame`, `frame_decision_shadow` и расхождения `must_handoff` с фактическим handoff/P0-сигналом. Для post-hoc shadow допустим только delta, полностью объяснённый ролью `bot_semantic_frame_shadow`. Если OFF-ветка не передана, отчёт не заявляет no-op, а явно пишет `needs_review`.

Для строгого no-op используем paired enrichment, а не два независимых draft-прогона. Сначала снимаем обычный OFF-прогон, затем добавляем frame к его замороженным транскриптам:

```bash
TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW=1 TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/adr003_semantic_frame_m1_scenarios_20260701.jsonl \
  --semantic-frame-enrich-from <OFF_RUN>/dynamic_dialog_transcripts.jsonl \
  --client-mode scripted \
  --judge-mode fake \
  --memory-mode fake \
  --out-dir <ON_ENRICHED_RUN>
```

Этот режим не вызывает draft-модель повторно: он берёт route/text/safety/checklist из OFF-транскрипта и добавляет только post-hoc frame metadata. Поэтому route/text diff в отчёте проверяет именно нейтральность frame-слоя, а не случайный шум повторного LLM-draft.

Граница доказательства: paired enrichment доказывает только то, что post-hoc frame metadata можно добавить к уже готовому черновику без изменения сохранённых `route/text/safety_flags/manager_checklist`. Это не доказывает качество `SemanticFrame` и не разрешает использовать frame для изменения маршрута. Перед любым active decision path нужен отдельный gold-регрейд по frame-полям и fail-closed этап, где frame может только усиливать ручную проверку, но не понижать существующий `manager_only`/P0/brand/fact guard.

Если нужен независимый ON-прогон с повторной генерацией draft, его результат нельзя использовать как строгий no-op: повторный LLM-draft сам по себе шумит. Такой прогон годится только как дополнительный investigation artifact.

## Baseline

Для 25 Wappi what-if кейсов на 2026-07-01:

- `bot_answer_self_for_pilot`: 12
- `draft_for_manager`: 10
- `manager_only`: 3

Это не semantic pass. Это зафиксированная точка сравнения для M1-регрейда и Claude #1 по сырью.

Важно: Wappi latest25 пока не имеет ручного `expected_frame` gold. Набор годится для route/text no-op, model-call count и coverage телеметрии. Точность SemanticFrame по полям можно заявлять только после отдельной gold-разметки и регрейда по сырью.

## Local Wappi25 Paired Enrichment Smoke

Локальный замер 2026-07-01:

- Артефакты: `audits/_inbox/adr003_semantic_frame_enrich_wappi25_20260701/`
- Scope: первые 25 Wappi latest25 кейсов из M1-набора, OFF с реальным `bot-mode codex`, fake judge/memory/semantic auxiliary; ON = paired enrichment от OFF-транскрипта.
- Acceptance: `pass`
- `route/text/safety_flags/manager_checklist` diff: 0
- Frame coverage: 25 / 25, required fields complete 25 / 25
- ON model calls: 25 total, 25 `bot_semantic_frame_shadow`, 0 non-frame calls
- SHA256 scenario: `df2726ddd67aac15e8c5ededd38c849a4e06a59f176109fb6ecd9da5f69842ca`
- SHA256 OFF transcripts: `cb7d62d42326c584e38681527348d5c6b0d311984a81bef6542ada6a2d212204`
- SHA256 ON enriched transcripts: `327fdfdc4e08ddf9bc3088aaf54b87f7a94198861b5c97e06cfb40bbd7c30889`
- SHA256 report JSON: `53f572da5316040daf39f197ec7594acbae411d6b67b8befdeb91b36cb1b40c0`

Ограничение: это доказывает только paired metadata no-op на Wappi25. В отчёте есть frame-vs-current-route mismatches (`must_handoff_vs_route`: 17 match / 8 mismatch; `must_handoff_vs_p0_signal`: 13 match / 12 mismatch), поэтому использовать frame для изменения маршрута пока нельзя. Следующий шаг перед active-фазой — ручной `expected_frame` gold и разбор mismatch-классов.
