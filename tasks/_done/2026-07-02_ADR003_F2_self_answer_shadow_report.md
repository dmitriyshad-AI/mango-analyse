# ADR-003 F2 self-answer shadow report

Дата: 2026-07-02
Ветка: `codex/adr003-semanticframe-migration`
База измерения: `9f8bac6` + локальное усиление блоков sanitizer/prose-placeholder перед коммитом.

## Что сделано

- Добавлен и усилен теневой self-answer gate `TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW`.
- Флаг default OFF, не в профиле, live не тронут.
- Слой пишет только `semantic_frame_self_answer_shadow`; route/text/safety/checklist не меняет.
- После ручного просмотра локального Ф2-замера добавлен блок на `output_sanitizer:*`, `client_name_echo`, `internal_client_placeholder` и `deferral_text_in_self`, чтобы не пропускать кривой/санитайзерный текст в будущий self-route.

## Локальный Ф2-замер

Тип: lightweight local enrichment поверх v4 transcript set, без live, без CRM/AMO/Tallanto, без изменения route/text.

Входы:
- `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/on/dynamic_dialog_transcripts.jsonl`
- `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Выходы:
- `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/on/dynamic_dialog_transcripts.jsonl`
- `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/adr003_semantic_frame_eval_report.json`
- `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/self_answer_shadow_candidates.csv`
- `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/gold_calibration/adr003_frame_gold_calibration_report.json`

Примечание: для локального enrichment восстановлены `valid_until`/`client_safe` из snapshot по exact fact keys, потому что старый v4 transcript был снят до добавления этих полей в `wide_fact_metadata`.

## Числа

- Ходов: `241`
- Route/text diff: `0`
- Input diff: `0`
- `would_demote_to_self`: `6`
- Классы кандидатов: `price=5`, `format=1`
- `p0_lowered_count`: `0`
- `manager_only_lowered_count`: `0`
- `freshness_unknown_self_candidates`: `0`
- Unsafe candidate examples: `0`

Gold-калибровка на Ф2-транскрипте:
- Gold rows: `79`
- Compared `must_handoff` rows: `77`
- `must_handoff_accuracy`: `0.9351`
- `too_confident`: `0`
- `too_cautious`: `5`
- `p0_misses`: `0`
- confidence bucket `0.90-1.00`: `65/66`, `too_confident=0`

Кандидаты:
- `rz_foton_offline_price_06` turn 1, `price`
- `rz_foton_offline_price_06` turn 2, `price`
- `rz_unpk_offline_price_07` turn 1, `price`
- `ra1_foton_platform_and_price` turn 1, `price`
- `ra1_unpk_unknown_slot_price` turn 1, `price`
- `cf142_over_handoff_unpk_clean_ready` turn 1, `format`

Блокировки:
- `protected_p0`: `80`
- `route_not_draft_for_manager`: `114`
- `risk_class_not_safe`: `25`
- `low_confidence`: `9`
- `blocking_safety_flags`: `4`
- `deal_stage_blocked`: `2`
- `missing_facts`: `1`

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_report_adr003_semantic_frame_eval.py`: `36 passed`
- `git diff --check`: pass
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`: `3816 passed, 5 skipped, 1 warning`

## Статус

Это `formal_pass` Ф2-shadow и полезное сырьё для регрейда, но не `semantic_pass` и не разрешение на Ф3-active.

Нужно дальше:
- Claude #1 регрейдит 6 candidates по raw transcript.
- Для более сильного доказательства нужен fresh M1/enrichment run на текущем коде, потому что локальный замер переиспользует v4 frame-транскрипты.
- Ф3 active можно обсуждать только после регрейда, class-by-class precision и отдельного `да` Дмитрия.

Сырьё и SHA-указатели зафиксированы в `tasks/_done/2026-07-02_ADR003_F2_self_answer_shadow_trace_manifest.md`. Большие raw transcripts остаются локально в `audits/_inbox/`: каталог gitignored, файлы 9-10 МБ и содержат публичные телефоны/почту из ответов бота; без отдельного решения не force-add'ятся в git.
