# Transcript Quality Phase 7 Backup / Rollback Report

Дата: 2026-05-09.

## Что сделано

Построен read-only backup/rollback manifest для phase 6 apply-plan.

Вход:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_phase6/full_apply_plan.csv`

Выход:

`stable_runtime/non_conversation_hard_gate_backup_manifest_20260509_phase7/`

SQLite не изменялись. Реальные копии БД не создавались: пока нет финального `auto_apply_ready` subset после GPT-review, копировать все БД преждевременно. Вместо этого создан точный план копирования и rollback snapshot.

## Итог

| Метрика | Значение |
|---|---:|
| Apply-plan rows | 5 761 |
| Affected SQLite DBs | 25 |
| Rollback rows captured | 5 761 |
| Missing rows | 0 |
| Schema warnings | 0 |
| DB SHA256 calculated | да |

## Риск-приоритеты

| Risk | Count |
|---|---:|
| critical | 448 |
| high | 3 693 |
| medium | 869 |
| low | 751 |

## Основные артефакты

- `db_manifest.csv` — список 25 затрагиваемых БД, размер, mtime, SHA256, количество строк.
- `rollback_snapshot.csv` — табличный snapshot старых значений по 5 761 строке.
- `rollback_snapshot.jsonl` — JSONL snapshot старых значений по 5 761 строке.
- `missing_rows.csv` — пустой, все строки найдены.
- `backup_copy_plan.sh` — план копирования БД перед apply.
- `ROLLBACK_RESTORE_NOTES.md` — заметки по восстановлению.

## Важное ограничение

Этот manifest построен по всем `5 761` строкам phase 6, которые пока находятся в `gpt_review_required`. После GPT-review нужно пересобрать phase 6 plan с decisions и получить `auto_apply_ready.csv`. Перед реальным staged apply нужно снова построить phase 7 manifest уже по фактическому `auto_apply_ready` subset или применить queue-filter.

## Следующий шаг

Запустить GPT-review по priority chunks:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_phase6/gpt_review_shards/priority_chunks_500/`

После этого:

1. объединить GPT decisions;
2. пересобрать apply-plan с `--gpt-decisions-jsonl`;
3. пересобрать backup/rollback manifest по `auto_apply_ready.csv`;
4. перейти к staged apply 50-100 строк.
