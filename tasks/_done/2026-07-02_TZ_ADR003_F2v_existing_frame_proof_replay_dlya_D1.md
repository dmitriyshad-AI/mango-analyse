# ADR-003 F2v: existing-frame proof shadow replay

Дата: 2026-07-02

## Задача

После регрейда 36ea110 нужно было проверить, можно ли получить честный proof/reconciliation-сигнал по свежим M1-транскриптам без повторного LLM-вызова. Предыдущий M1-прогон уже содержит `bot_semantic_frame`, но не содержит `semantic_frame_existence_proof_shadow` / `semantic_frame_proof_reconciliation_shadow`.

## Сделано

- Добавлен `scripts/enrich_adr003_existing_frame_proof_shadow.py`.
- Скрипт читает уже сохранённый `bot_semantic_frame` из JSONL и локально применяет существующие shadow-слои:
  - `apply_semantic_frame_existence_proof_shadow`;
  - `apply_semantic_frame_proof_reconciliation_shadow`;
  - `apply_semantic_frame_self_answer_shadow`.
- Скрипт не вызывает Codex/LLM, не трогает live, не меняет `bot_route` и `bot_text`.
- Добавлены тесты `tests/test_enrich_adr003_existing_frame_proof_shadow.py`.
- Replay прогнан на свежем M1-наборе 36ea110:
  - input: `.../adr003_f2_self_answer_shadow_36ea110/ON/dynamic_dialog_transcripts.jsonl`;
  - KB: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`;
  - output: `audits/_inbox/adr003_f2v_existing_frame_proof_replay_20260702155709/reports_36ea110/`.

## Результат по сырью

- turns_total: `241`
- turns_with_frame: `241`
- model_calls_added: `0`
- input_source_rev: `36ea110`
- route_text_diff_count: `0`
- protected_turn_field_diff_count: `0`
- existence proof `exists`: `44`
- proof reconciliation `would_reconcile_to_safe_reference`: `9`
- current over-handoff rows with proof reconciliation in real-lever report: `1`
- proof reconciliation send-as-is candidates: `0`
- proof reconciliation text blocked: `2`
- fact-gated strict F3 candidates: `0`

## Вывод

Комментарий Claude #1 подтверждён: быстрый route-only active-step не готов. Даже когда точный факт найден, текущий текст обычно manager/deferral или требует отдельной текстовой политики/шаблона. Следующий продуктивный шаг - не включать понижение маршрута, а проектировать безопасный renderer/шаблон для доказанных стабильных фактов существования/формата.

Live pid, профиль, P0-floor/preblock и runtime-путь бота не трогались.
