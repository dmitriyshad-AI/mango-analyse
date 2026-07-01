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
- SHA256 report JSON: `b3673d75c69f8b85ea7199400c9899e9718fef1ca205be62166c6aa441097769`

Ограничение: это доказывает только paired metadata no-op на Wappi25. В отчёте есть frame-vs-current-route mismatches (`must_handoff_vs_route`: 17 match / 8 mismatch; `must_handoff_vs_p0_signal`: 13 match / 12 mismatch), поэтому использовать frame для изменения маршрута пока нельзя. Следующий шаг перед active-фазой — ручной `expected_frame` gold и разбор mismatch-классов.

## Local Full131 Paired Enrichment

Локальный замер 2026-07-01:

- Артефакты: `audits/_inbox/adr003_semantic_frame_enrich_full131_20260701/`
- Scope: полный ADR003 M1-набор, 131 диалог / 241 ход.
- OFF: реальный `bot-mode codex`, fake judge/memory/semantic auxiliary; `pilot_gold_v1`, `P0_MODEL_LED=1`, `PROSE_MODEL_LED=1`.
- ON: paired enrichment от OFF-транскрипта, `parallel=4`.
- Время: OFF 1465 sec, ON enrichment 738 sec, total 2203 sec.
- Acceptance: `pass`
- `route/text/safety_flags/manager_checklist` diff: 0
- Input diff: 0
- Frame coverage: 241 / 241, required fields complete 241 / 241
- ON model calls: 241 total, 241 `bot_semantic_frame_shadow`, 0 non-frame calls
- SHA256 OFF transcripts: `ea2bfde7f34bdf11535a6b579980e9e081f977930a9bb4fd921b8359b53d8bb3`
- SHA256 ON enriched transcripts: `c43f67e48ba8c58108f1a424df01259cddc8af51805b0911d5da6bf14c19b15c`
- SHA256 report JSON: `e3a58dc9069d3b75c65bae1e6060a51fe0580c1947b006390c2845c4e966116f`

Ограничение: full131 доказывает технический shadow/no-op, но не semantic-pass для руления. `frame.must_handoff` всё ещё существенно расходится с текущими детекторами (`must_handoff_vs_route`: 203 match / 38 mismatch; `must_handoff_vs_p0_signal`: 172 match / 69 mismatch). Эти расхождения нужно разметить по gold: часть может быть желаемым снижением over-handoff, часть — ложный handoff или риск пропуска.

## Gold Queue For Frame Mismatches

Для ручного регрейда SemanticFrame добавлен локальный builder:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_adr003_frame_gold_queue.py \
  --transcripts <ON_ENRICHED_RUN>/dynamic_dialog_transcripts.jsonl \
  --out-dir <GOLD_QUEUE_DIR>
```

Он не вызывает модель и не меняет поведение: читает enriched transcript, сравнивает `frame.must_handoff` с текущим route-handoff и P0-сигналом, затем пишет JSONL/CSV-очередь для ручного заполнения `expected_*` полей. По умолчанию в очередь попадают только расхождения; `--include-matches` нужен только для выборочного контроля совпадений.

Локальная очередь по full131 от 2026-07-01:

- Артефакты: `audits/_inbox/adr003_semantic_frame_gold_queue_full131_20260701/`
- Dialogs/turns: 131 / 241
- Framed turns: 241 / 241
- Queue rows: 75
- `frame_handoff_no_p0_signal`: 37
- `frame_handoff_current_self+frame_handoff_no_p0_signal`: 32
- `frame_handoff_current_self`: 3
- `frame_self_current_handoff`: 3
- Brands in queue: Foton 35, UNPK 40
- PII risk rows: 0

Эти 75 строк могут содержать клиентские тексты из transcript, поэтому очередь остаётся локальным audit artifact и не добавляется в git. Коммитится только сам builder, тесты и агрегированная сводка в документах.

## Manual Gold Review Summary

Первый смысловой разбор очереди зафиксирован в `audits/_inbox/adr003_semantic_frame_gold_review_20260701/semantic_gold_summary.md`.

Итог по 75 строкам:

- `frame_correct`: 29
- `frame_too_cautious`: 43
- `unclear`: 3

Вывод: `SemanticFrame` уже полезен как сигнал `manager_action` для оплаты/чека/брони/живого наличия/запроса администратора, но пока слишком часто поднимает `must_handoff` на справочных вопросах, обычных next-step ходах и safe deferral. Поэтому active behavior остаётся заблокированным. Следующий этап должен быть не "frame рулит всем", а узкий `manager_action_gate` за отдельным default-OFF флагом и с gold-регрессиями из этой очереди.

## Manager-Action Gate Prototype

Коммит `3aedeea` добавляет узкий default-OFF флаг `TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE`.

Гейт работает только после post-hoc frame со статусом `ok`; inline legacy `semantic_frame_shadow` не считается активным источником. Он может только повысить автономный `bot_answer_self*` до `draft_for_manager`, не меняет текст, не поднимает в `manager_only`, не понижает существующие handoff/P0 routes и не включён в профиль.

Локальный replay поверх сохранённого full131 paired-enrichment артефакта:

- source: `audits/_inbox/adr003_semantic_frame_enrich_full131_20260701/on/dynamic_dialog_transcripts.jsonl`;
- первый вариант дал 17 promoted / 241, 0 text changes;
- после сужения `check_availability` до поздних/операционных стадий: 12 promoted / 241, 0 text changes;
- убраны ранние справочные `check_availability` на `interest/qualification`, где SemanticFrame был too-cautious.

Ограничение: это `formal_pass` + локальный semantic smoke, но не разрешение на включение флага. Перед включением нужен отдельный semantic gold-регрейд promoted rows и negative rows: цены, адрес, платформа, формат, порядок записи, рассрочка, pre-sale refund, "подумаем/оплачу позже", safe deferral.

## Phase 1 Frame Gold Calibration

Локальная Ф1-калибровка 2026-07-01:

- Gold: `product_data/telegram_dynamic_test_sets/adr003_frame_gold_labels_20260701.jsonl`.
- Gold SHA256: `4c505bb23cddbdfb8a0d4324bd2822b253001dc46fc0e75b90ab5ba2d6fb5be3`.
- Gold scope: 75 строк из full131 mismatch/manager-action очереди, без клиентских текстов; 73 сравнимые, 2 `unclear`.
- Scorer: `scripts/report_adr003_frame_gold_calibration.py`.
- Report: `audits/_inbox/adr003_frame_gold_calibration_20260701/`.
- `must_handoff` accuracy: `0.6027` (44/73).
- `too_cautious`: `29`.
- `too_confident`: `0`.
- `current_over_handoff_candidates`: `21`.
- `safe_self_candidates`: `32`.
- Confidence buckets: `0.80-0.89` = 40 rows / 47.5% must_handoff accuracy / 21 too_cautious; `0.90-1.00` = 33 rows / 75.76% accuracy / 8 too_cautious.
- Field accuracy: `risk_class` 37/75, `requested_action` 61/75, `answerability` 1/75.

Вывод Ф1: `SemanticFrame` пока не готов рулить автономией. `too_confident=0` на этой очереди подтверждает безопасную сторону ошибки, но `must_handoff` accuracy низкая, а `answerability` не соблюдает заявленную схему (`yes/no` вместо `answer_self/manager_only` в сохранённых frame). Следующий шаг — улучшать инструкцию/схему и повторять Ф1-калибровку, а не включать Ф2/Ф3 active self-answer gate.

## Phase 1 Prompt Calibration v4

Локальная калибровка 2026-07-01 после исправления enum `answerability` и уточнения границы "safe справка" vs "manager action":

- Code/tests: `src/mango_mvp/channels/subscription_llm_parts/provider.py`, `tests/test_subscription_llm_draft_provider.py`.
- OFF source: `audits/_inbox/adr003_semantic_frame_enrich_full131_20260701/off/dynamic_dialog_transcripts.jsonl`.
- ON paired enrichment: `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/on/`.
- Gold report: `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_frame_gold_calibration_report.json`.
- Paired report: `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_semantic_frame_eval_report.json`.
- Dialogs/turns: 131 / 241.
- Route/text diff: 0.
- ON model calls: 241 total, 241 `bot_semantic_frame_shadow`, 0 non-frame calls.
- Frame required fields complete: 241 / 241.
- Gold compared rows: 73 comparable + 2 unclear.
- `must_handoff` accuracy: `0.9315` (68/73).
- `too_confident`: `0`.
- `too_cautious`: `5`.
- Field accuracy: `answerability` 68/75, `risk_class` 67/75, `requested_action` 62/75.
- Confidence calibration: bucket `0.90-1.00` = 62 rows / 98.39% `must_handoff` accuracy / 0 `too_confident`; bucket `0.80-0.89` = 11 rows / 63.64% `must_handoff` accuracy / 0 `too_confident` / 4 `too_cautious`.

Что изменилось относительно первой Ф1-калибровки:

| Run | must_handoff accuracy | too_cautious | too_confident | answerability accuracy | risk_class accuracy | requested_action accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 0.6027 | 29 | 0 | 0.0133 | 0.4933 | 0.8133 |
| prompt v2 | 0.8219 | 1 | 12 | 0.8000 | 0.8133 | 0.8267 |
| prompt v3 | 0.8904 | 5 | 2 | 0.8667 | 0.8533 | 0.8000 |
| prompt v4 | 0.9315 | 5 | 0 | 0.9067 | 0.8933 | 0.8267 |

Остаточные 5 `too_cautious` строк: safe age/format/address/schedule reference, compare formats/prices without live availability, age suitability without free seats, general discount/benefit explanation, general pedagogical guidance. Это не опасные пропуски, но это всё ещё lost-autonomy и требует регрейда Claude #1 перед Ф2/F3.

Вывод: Ф1 дала сильный безопасный прирост и убрала опасные `too_confident` на gold-75, но это не разрешение включать активный self-answer gate. Следующий допустимый шаг — Ф2 shadow lowering с порогом не ниже confidence `0.90`, class-specific статистикой, freshness checks и отдельным регрейдом. Ф3 active остаётся заблокированной до регрейда Ф2 и решения Дмитрия.
