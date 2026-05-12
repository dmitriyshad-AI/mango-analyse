# Canonical Master Build Report

Дата: 2026-05-09

## Короткий вывод

Создана реальная canonical master SQLite DB по всему корпусу звонков за период январь 2025 - май 2026 включительно.

Старые рабочие БД, batch-папки и аудио не изменялись и не удалялись. Запись выполнена только в новую папку `stable_runtime/canonical_master_20260509_v1/`.

## Новый главный артефакт

- `stable_runtime/canonical_master_20260509_v1/canonical_calls_master.db`
- Размер: примерно `1.4 GiB`
- Папка целиком: примерно `1.5 GiB`

Сопутствующие файлы:

- `stable_runtime/canonical_master_20260509_v1/summary.json`
- `stable_runtime/canonical_master_20260509_v1/canonical_preview.csv`
- `stable_runtime/canonical_master_20260509_v1/coverage_by_month.tsv`
- `stable_runtime/canonical_master_20260509_v1/db_scan_summary.tsv`
- `stable_runtime/canonical_master_20260509_v1/selected_by_db.tsv`
- `stable_runtime/canonical_master_20260509_v1/duplicate_conflicts.csv`
- `stable_runtime/canonical_master_20260509_v1/README.md`

## Что внутри canonical DB

Таблицы:

- `canonical_builds` - метаданные сборки.
- `source_artifacts` - входные DB, coverage report, exclusions file, source audio dir.
- `canonical_calls` - одна строка на каждый source audio.
- `call_record_provenance` - выбранный source DB/id и проигравшие кандидаты для дублей.
- `call_exclusions` - исключенные звонки manager-manager/no-ASR.
- `call_quality_current` - только текущие факты качества из выбранного `analysis_json`, без смешивания с hard-gate dry-run/backfill.
- `validation_results` - результаты проверок сборки.

Важно: hard-gate/backfill quality layer намеренно не записан как текущая истина. Его нужно вести отдельным shadow/backfill слоем после завершения Claude/GPT-аудита.

## Проверенные счетчики

По `summary.json`:

- Source audio: `64 867`
- Excluded manager-manager/no-ASR: `35`
- Actionable source audio: `64 832`
- ASR done actionable: `64 832`
- Full R+A actionable: `64 832`
- Missing ASR actionable: `0`
- Missing full R+A actionable: `0`
- Included DB count: `36`
- DBs with selected records: `25`
- Source filenames with multiple DB candidates: `35 604`

Независимая SQL-сверка DB:

- `canonical_calls`: `64 867`
- `distinct source_filename`: `64 867`
- `full_ra`: `64 832`
- `excluded_manager_manager_no_asr`: `35`
- `call_record_provenance`: `107 794`
- `selected_primary`: `64 832`
- `candidate_lost`: `42 962`
- `call_exclusions`: `35`
- `call_quality_current`: `64 832`
- failed validation checks: `0`

## Archive dry-run v4

После создания canonical master пересобрана read-only инвентаризация проекта:

- `stable_runtime/project_inventory_20260509_v4_after_canonical_master/`

Ключевые файлы:

- `summary.json`
- `top_level_sizes.tsv`
- `stable_runtime_sizes.tsv`
- `db_inventory.tsv`
- `archive_candidates_dry_run.tsv`
- `README.md`

Результат:

- Project size by `du`: `55 806 365 696` bytes, примерно `52.0 GiB`
- DB files: `127`
- DB total logical size: `10 766 073 856` bytes, примерно `10.03 GiB`
- Archive candidate rows: `90`
- Potential archive candidate size after master: `13 786 566 656` bytes, примерно `12.84 GiB`
- Replacement artifact для строк archive plan: `stable_runtime/canonical_master_20260509_v1/canonical_calls_master.db`

Крупнейшие кандидаты:

- `stable_runtime/external_m1_jan_mar_2025_asr_only_20260504`
- `stable_runtime/jun_jul_aug_2025_asr_only_20260503`
- `telegram_exports (2)` - только review, не auto-archive.
- `_local_archive_20260424` - только review.
- `stable_runtime/venv_stable.broken_20260407`
- `2026-03-05-21-06-49-ч1` - только review.
- `2026-03-05-21-06-49-ч2` - только review.
- `.venv-asrbench` - только review.
- `stable_runtime/apr_may_2025_asr_only_20260502`
- `stable_runtime/ab_tests`

Физическое архивирование пока не выполнялось.

## Реализованные изменения в коде

Добавлен write-mode для canonical master:

- `src/mango_mvp/maintenance/canonical_master.py`
- `scripts/build_canonical_calls_master.py`
- `tests/test_canonical_master.py`

Усилен inventory/archive dry-run:

- `src/mango_mvp/maintenance/project_inventory.py`
- `scripts/build_project_inventory.py`
- `tests/test_project_inventory.py`

## Проверка

Пройдены таргетные тесты:

```text
PYTHONPATH=src python3 -m pytest tests/test_project_inventory.py tests/test_canonical_master.py
4 passed
```

Пройден полный suite:

```text
PYTHONPATH=src python3 -m pytest
611 passed, 1 warning
```

Единственный warning внешний: `urllib3` сообщает, что системный Python собран с `LibreSSL 2.8.3`, а не с OpenSSL 1.1.1+. На результаты тестов canonical/inventory это не влияет.

## Следующий безопасный шаг

1. Прогнать полный `pytest`.
2. После результата Claude/GPT-аудита по hard-gate пакету решить, применять ли quality backfill.
3. До физического удаления выполнить ручное review `stable_runtime/project_inventory_20260509_v4_after_canonical_master/archive_candidates_dry_run.tsv`.
4. Если review подтвержден, сделать не удаление, а перенос кандидатов в отдельную локальную архивную папку, например `_local_archive_after_canonical_master_20260509/`, с manifest-файлом.
5. После стабилизации canonical master перевести downstream сборки ROP/contact/KB на чтение из `canonical_calls_master.db`, а не из набора исторических batch DB.
