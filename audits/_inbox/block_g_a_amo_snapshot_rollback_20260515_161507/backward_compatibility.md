# Backward Compatibility

## Что сохраняется

- Обычный Stage6 dry-run остается без live AMO-доступа.
- Старые выходные файлы live-write сохраняются:
  - `deal_stage6_writeback_report.csv`;
  - `deal_stage6_dry_run_report.csv`;
  - `deal_stage6_writeback_report.json`;
  - `summary.json`.
- `scripts/readback_deal_aware_amo_fields.py` не менялся и продолжает читать старый writeback report.
- `build_custom_fields_values()` не менялся, поэтому старые AMO update-вызовы не начали внезапно писать пустые значения.

## Что добавлено

- Дополнительные live-write outputs:
  - `pre_write_snapshot.jsonl`;
  - `pre_write_snapshot.csv`;
  - `rollback_manifest.json`;
  - `live_write_report.csv`;
  - `live_write_report.json`.
- Новые CLI-аргументы live-write:
  - `--batch-size`;
  - `--delay-ms`;
  - `--max-retries`;
  - `--resume-from-report`.

## Что намеренно не совместимо с опасным поведением

Если snapshot не удалось сохранить, строка больше не пишется в AMO. Это намеренное fail-closed поведение.
