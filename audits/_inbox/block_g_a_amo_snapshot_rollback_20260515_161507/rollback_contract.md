# Rollback Contract

Rollback-скрипт:

`scripts/rollback_deal_aware_amo_fields.py`

## Режимы

- Без `--apply`: только dry-run отчет.
- С `--apply`: требуется отдельный token `ROLLBACK_DEAL_AWARE_AMO_FIELDS`.

Token отката намеренно отличается от live-write token.

## Правило отката

Для каждого поля из snapshot:

1. Прочитать текущий AMO lead.
2. Сравнить текущее значение с `new_value`.
3. Если текущее значение равно `new_value`, можно вернуть `old_value`.
4. Если текущее значение отличается, пропустить поле с причиной `current_value_changed_after_write`.
5. Если `old_value` пустой и текущий helper не умеет безопасно очистить поле, поставить `manual_restore_required`.
6. Поля вне snapshot не трогать.

## Outputs

- `rollback_dry_run_report.csv`
- `rollback_dry_run_report.json`
- `rollback_apply_report.csv`
- `rollback_apply_report.json`
- `rollback_resume_state.json`
- `rollback_summary.json`

## Rate limit / retry

- default `--batch-size`: 10;
- default `--delay-ms`: 750;
- default `--max-retries`: 3;
- 429 и 5xx повторяются с растущей паузой;
- постоянные 4xx не повторяются бесконечно.
