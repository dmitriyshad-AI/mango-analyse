# ADR-003 SemanticFrame v4 Turn3 Gold Recheck

Дата: 2026-07-01

Цель доброса: расширить gold-метрику опасными turn3 P0/payment/refund closing-кейсами, чтобы `too_confident` ловился автоматическим скорером, а не ручной сверкой.

Изменения:

- `product_data/telegram_dynamic_test_sets/adr003_frame_gold_labels_20260701.jsonl`: добавлены 4 `turn=3` строки для `cf142_p0_*`.
- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_eval_manifest_20260701.json`: обновлены `line_count`, `coverage`, `sha256`.
- `docs/ADR003_SEMANTIC_FRAME_EVAL.md`: добавлен раздел Phase 1 v4 Turn3 P0 Gold Extension.
- Локальный recheck выполнен на сохранённой v4 paired-enrichment телеметрии.

Важно: это не заменяет свежий M1-прогон. M1 должен вручную прогнать v4 frame-shadow на том же full131-наборе и вернуть свежий `dynamic_dialog_transcripts.jsonl`, gold report, paired report и sha-manifest.
