# Snapshot Contract

Перед каждым AMO PATCH live-write обязан:

1. Прочитать текущий lead из AMO.
2. Извлечь текущие значения всех полей, которые планируется писать.
3. Сохранить snapshot-строки.
4. Только после успешного сохранения snapshot вызвать PATCH.

## Snapshot fields

- `schema_version`
- `batch_id`
- `input_csv`
- `input_sha256`
- `row_index`
- `review_id`
- `lead_id`
- `field_name`
- `field_id`
- `field_type`
- `old_value`
- `new_value`
- `old_value_sha256`
- `new_value_sha256`
- `snapshot_taken_at`
- `writer_version`
- `operator_approval_path`

## Outputs

- `pre_write_snapshot.jsonl`
- `pre_write_snapshot.csv`
- `rollback_manifest.json`

## Fail-closed rule

Если snapshot не записан, AMO PATCH не вызывается.
