# TZ-21 Block A: tail 3,439 import report

Дата: 2026-06-13

## Scope

Выполнено read-only/SQLite-вливание результатов хвоста `analyze_tail_20260612` в каноническую базу звонков.

Записи во внешние системы не выполнялись:

- AMO/CRM: нет
- Tallanto: нет
- ASR: не запускался
- Resolve+Analyze: не запускался
- LLM-вызовы: 0

## Inputs

- Results: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_tail_20260612/results_part1..4.jsonl.gz`
- Manifest: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_tail_20260612/data/manifest.json`
- Canonical DB: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`
- Import script: `scripts/import_tz19_analyze_tail_results.py`

Input checks:

- Manifest rows: 3,439
- Unique result ids: 3,439
- Result parts: 860 + 860 + 860 + 859 = 3,439
- ids sha256: `8680b5456824ac7159cc1ec5993399aa8ae57712602aa4d4c2d582b65041ad5e`
- prompt sha256: `12718ea6b8a5ee500910300c4c2de7c3695f78217c3b63a62d572de612b5eacf`
- blacklist overlap: 0
- result statuses: `done=3439`
- prompt versions: `v7=3439`
- models: `gpt-5.4-mini=3439`

## Backup

Primary backup before write:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_20260613`

Idempotence-run backup:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_idempotence_20260613`

Both backup files exist locally and are ignored by git.

## Counters

Dry-run:

- read: 3,439
- updated: 3,439
- skipped_same: 0
- rejected: 0 across all rejection classes
- prompt_version_rows before/after in manifest set: 0 / 0

Apply:

- read: 3,439
- updated: 3,439
- skipped_same: 0
- rejected: 0 across all rejection classes
- prompt_version_rows before/after in manifest set: 0 / 3,439

Idempotence apply:

- read: 3,439
- updated: 0
- skipped_same: 3,439
- rejected: 0 across all rejection classes
- prompt_version_rows before/after in manifest set: 3,439 / 3,439

## Acceptance checks

- `PRAGMA quick_check`: `ok`
- Total v7 rows after import: 26,118
- Manifest rows with v7 after import: 3,439
- Allowed update columns only: `analysis_json`, `analysis_status`, `analysis_json_chars`, `has_analysis_json`, `last_error`
- `needs_review` preserved inside `analysis_json` payloads as received.
- Blacklist rows were not included in this import.

## Local artifacts

Ignored local reports:

- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_tail_import_20260613/dry_run_report.json`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_tail_import_20260613/apply_report.json`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_tail_import_20260613/idempotence_apply_report.json`
