# ADR003 SemanticFrame Enrichment Measurement

Дата: 2026-07-01.

## Что сделано

- Добавлен `--semantic-frame-enrich-from` в `scripts/run_telegram_dynamic_client_sim.py`.
- Режим читает готовый `dynamic_dialog_transcripts.jsonl`, строит замороженный `SubscriptionDraftResult` из сохранённых `bot_route`/`bot_text`/`bot_safety_flags`/`bot_manager_checklist` и добавляет только post-hoc `SemanticFrame` metadata.
- Draft-модель в enrichment-режиме не вызывается повторно.
- Enrichment теперь уважает `--parallel`: диалоги обогащаются параллельно, а порядок выходного JSONL сохраняется по порядку входного transcript.
- `dynamic_summary.json` теперь пишет `semantic_frame_enrichment.status`: `none` / `partial` / `all`.
- `scripts/report_adr003_semantic_frame_eval.py` различает обычный full-run pair и paired enrichment. Для enrichment PASS возможен только когда все ON-вызовы являются `bot_semantic_frame_shadow`, входные реплики совпадают, route/text/safety/checklist не изменились.
- Manifest ADR003 обновлён: каноничный Stage 1 замер = OFF run + paired enrichment + report. Старый same-payload command оставлен только как legacy investigation path.

## Измерение

Локальный Wappi25 paired enrichment:

- Артефакты: `audits/_inbox/adr003_semantic_frame_enrich_wappi25_20260701/`
- Scope: первые 25 Wappi latest25 кейсов из ADR003 M1-набора.
- OFF: реальный `bot-mode codex`, `judge/memory/semantic/semantic-verifier` fake, `selling` det.
- ON: paired enrichment от OFF-транскрипта.

Результат:

- Acceptance: `pass`
- Frame present: 25 / 25
- Required fields complete: 25 / 25
- Route/text/safety/checklist diff: 0
- Input diff: 0
- ON LLM calls: 25 total, 25 `bot_semantic_frame_shadow`, 0 non-frame
- `must_handoff_vs_route`: 17 match / 8 mismatch
- `must_handoff_vs_p0_signal`: 13 match / 12 mismatch

Full131 paired enrichment:

- Артефакты: `audits/_inbox/adr003_semantic_frame_enrich_full131_20260701/`
- Scope: 131 диалог / 241 ход.
- OFF: реальный `bot-mode codex`, fake judge/memory/semantic auxiliary.
- ON: paired enrichment от OFF-транскрипта, `parallel=4`.
- Acceptance: `pass`
- Frame present: 241 / 241
- Required fields complete: 241 / 241
- Route/text/safety/checklist diff: 0
- Input diff: 0
- ON LLM calls: 241 total, 241 `bot_semantic_frame_shadow`, 0 non-frame
- `must_handoff_vs_route`: 203 match / 38 mismatch
- `must_handoff_vs_p0_signal`: 172 match / 69 mismatch
- Time: OFF 1465 sec, ON 738 sec, total 2203 sec

Хэши:

- Scenario: `df2726ddd67aac15e8c5ededd38c849a4e06a59f176109fb6ecd9da5f69842ca`
- OFF transcripts: `cb7d62d42326c584e38681527348d5c6b0d311984a81bef6542ada6a2d212204`
- ON enriched transcripts: `327fdfdc4e08ddf9bc3088aaf54b87f7a94198861b5c97e06cfb40bbd7c30889`
- Report JSON: `b3673d75c69f8b85ea7199400c9899e9718fef1ca205be62166c6aa441097769`

Full131 hashes:

- OFF transcripts: `ea2bfde7f34bdf11535a6b579980e9e081f977930a9bb4fd921b8359b53d8bb3`
- ON enriched transcripts: `c43f67e48ba8c58108f1a424df01259cddc8af51805b0911d5da6bf14c19b15c`
- Report JSON: `e3a58dc9069d3b75c65bae1e6060a51fe0580c1947b006390c2845c4e966116f`

## Вывод

Paired enrichment доказал только узкий no-op: frame metadata можно добавить к уже готовому черновику без изменения сохранённых route/text/safety/checklist.

Это не semantic-pass для использования frame как decision policy. Frame-vs-current-route mismatches остаются и требуют gold-разметки и регрейда до любого active behavior.

## Gold Queue Builder

Добавлен `scripts/build_adr003_frame_gold_queue.py`.

Назначение:

- читает enriched `dynamic_dialog_transcripts.jsonl`;
- сравнивает `frame.must_handoff` с текущим route-handoff и P0-сигналом;
- считает route-handoff только по `bot_route in {manager_only, draft_for_manager}`; общий пилотный флаг `manager_approval_required` не считается handoff, потому что он стоит и на self-answer черновиках;
- требует строгий bool в `frame.must_handoff`; строки `"false"`/`"true"` помечаются как invalid frame, а не приводятся через `bool(...)`;
- явно пишет `input_status`, чтобы OFF/no-frame input не выглядел как "нет расхождений";
- помечает PII-риск в строках очереди;
- пишет JSONL/CSV очередь ручной разметки `expected_*` полей;
- по умолчанию включает только mismatch-строки, `--include-matches` доступен для выборочного контроля совпадений.

Локальный прогон по full131:

- Артефакты: `audits/_inbox/adr003_semantic_frame_gold_queue_full131_20260701/`
- Dialogs/turns: 131 / 241
- Queue rows: 75
- `frame_handoff_no_p0_signal`: 37
- `frame_handoff_current_self+frame_handoff_no_p0_signal`: 32
- `frame_handoff_current_self`: 3
- `frame_self_current_handoff`: 3
- Brands in queue: Foton 35, UNPK 40
- PII risk rows: 0

Очередь может содержать клиентские тексты из transcript, поэтому она оставлена как локальный ignored audit artifact. В git добавлены только builder, тесты и агрегированная сводка.
