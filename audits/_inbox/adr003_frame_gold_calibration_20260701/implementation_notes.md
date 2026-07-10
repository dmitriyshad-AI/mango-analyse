# ADR-003 Frame Gold Calibration Implementation Notes

Дата: 2026-07-01.

## Scope

Ф1 ТЗ-цели автономности через SemanticFrame выполнена как измерительный слой, без изменения поведения бота.

Добавлено:

- `product_data/telegram_dynamic_test_sets/adr003_frame_gold_labels_20260701.jsonl` — машинный gold на 75 строках очереди full131, без клиентских текстов.
- `scripts/report_adr003_frame_gold_calibration.py` — scorer gold vs `bot_semantic_frame`.
- `tests/test_report_adr003_frame_gold_calibration.py` — unit-тесты scorer.
- `docs/ADR003_SEMANTIC_FRAME_EVAL.md` — результаты Ф1-калибровки и вывод.
- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_eval_manifest_20260701.json` — ссылка и SHA gold-файла.

## What This Does Not Do

- Не меняет route/text.
- Не добавляет runtime-флаги.
- Не включает флаги в профиль.
- Не трогает live/Wappi/Telegram/AMO/Tallanto/CRM.
- Не меняет P0-preblock/floor.
- Не меняет мораторий-снимки `tests/fixtures/adr003_*snapshot.json`.

## Measurement Result

`scripts/report_adr003_frame_gold_calibration.py` against full131 enriched transcript:

- compared rows: 73;
- unclear rows: 2;
- `must_handoff` accuracy: 0.6027;
- `too_cautious`: 29;
- `too_confident`: 0;
- `risk_class` accuracy: 37/75;
- `requested_action` accuracy: 61/75;
- `answerability` accuracy: 1/75.

Conclusion: active self-answer gate remains blocked. The frame is safe-biased but not calibrated enough for autonomy.
