# Canonical Master Dry-Run Report

Дата: 2026-05-09

## Что было сделано

Выполнен безопасный этап наведения порядка без удаления и без изменения существующих рабочих БД:

1. Проверен текущий план cleanup/canonical master.
2. Собран read-only `canonical master preview` по всему корпусу звонков.
3. Собрана read-only инвентаризация проекта и кандидатов на архивирование.
4. Добавлены воспроизводимые скрипты и тесты, чтобы повторять эту проверку.

## Важные ограничения

Ничего не удалялось.

Существующие SQLite DB не изменялись.

Фактическая canonical master DB пока не записывалась. Создан только preview/manifest-layer для проверки, что единая база истины собирается без потерь.

## Canonical Master Preview

Актуальная папка:

`stable_runtime/canonical_master_20260509_dry_run_v2/`

Ключевые файлы:

- `summary.json`
- `canonical_preview.csv`
- `coverage_by_month.tsv`
- `db_scan_summary.tsv`
- `selected_by_db.tsv`
- `duplicate_conflicts.csv`
- `README.md`

Результат:

- Source audio: `64 867`
- Excluded manager-manager/no-ASR: `35`
- Actionable source audio: `64 832`
- Selected canonical records: `64 867`
- ASR done actionable: `64 832`
- Full R+A actionable: `64 832`
- Missing ASR actionable: `0`
- Missing full R+A actionable: `0`
- Included DB count: `36`
- DBs with selected records: `25`
- Scan errors: `0`
- Validation passed: `true`

Статусы в `canonical_preview.csv`:

- `full_ra`: `64 832`
- `excluded_manager_manager_no_asr`: `35`

Важная находка:

- `35 604` `source_filename` имеют несколько DB-кандидатов.

Вывод: dedupe/provenance слой обязателен. Нельзя безопасно удалять старые batch-папки, пока не будет записана canonical master DB и сохранен provenance по проигравшим дублям.

## Project Inventory

Актуальная папка:

`stable_runtime/project_inventory_20260509_v3_du/`

Ключевые файлы:

- `summary.json`
- `top_level_sizes.tsv`
- `stable_runtime_sizes.tsv`
- `db_inventory.tsv`
- `archive_candidates_dry_run.tsv`
- `README.md`

Результат:

- Project size by `du`: `53 888 458 752` bytes, примерно `50.2 GiB`
- Top-level entries: `61`
- `stable_runtime` entries: `137`
- DB files: `126`
- DB total logical size: `9 255 518 208` bytes, примерно `8.62 GiB`
- Archive candidate rows: `90`
- Potential archive candidate size after master: `13 786 566 656` bytes, примерно `12.84 GiB`

Крупнейшие `do_not_touch_now`:

- `2026-03-09--26` — основная аудио-папка, около `24.5 GiB`
- `stable_runtime/ra_missing_all_20260506` — текущий важный рабочий слой/backup-зона
- актуальные `non_conversation_hard_gate*` и `transcript_quality*` папки

Крупные кандидаты только после canonical master:

- `stable_runtime/external_m1_jan_mar_2025_asr_only_20260504`
- `stable_runtime/jun_jul_aug_2025_asr_only_20260503`
- `stable_runtime/apr_may_2025_asr_only_20260502`
- `stable_runtime/ab_tests`
- `stable_runtime/benchmarks`
- старые `overnight_*`
- backup DB вида `.before_*`

Все это только dry-run список. Физическое архивирование/удаление требует отдельного подтверждения.

## Добавленные инструменты

Скрипты:

- `scripts/build_canonical_calls_master.py`
- `scripts/build_project_inventory.py`

Модули:

- `src/mango_mvp/maintenance/canonical_master.py`
- `src/mango_mvp/maintenance/project_inventory.py`
- `src/mango_mvp/maintenance/__init__.py`

Тесты:

- `tests/test_canonical_master.py`
- `tests/test_project_inventory.py`

## Проверка

Локально пройдены таргетные тесты:

- `tests/test_canonical_master.py`
- `tests/test_project_inventory.py`

Проверенные свойства:

- duplicate resolution выбирает полный R+A поверх более свежей partial-записи;
- manager-manager no-ASR не считается missing gap;
- dry-run не создает canonical DB и не меняет исходные SQLite;
- inventory находит DB и archive candidates без удаления.

## Следующий безопасный шаг

Следующий этап — не удаление, а запись actual canonical master DB в новую папку, например:

`stable_runtime/canonical_master_20260509/`

Минимальный безопасный вариант:

1. Создать новую SQLite DB, не трогая старые DB.
2. Таблица `canonical_calls`: одна строка на каждый из `64 867` source audio.
3. Таблица `call_record_provenance`: выбранный DB/id и проигравшие DB/id для дублей.
4. Таблица `source_artifacts`: входные DB, coverage report, exclusions.
5. Таблица `call_exclusions`: `35` manager-manager no-ASR.
6. Валидация против coverage v5.
7. После этого подготовить archive plan v2 уже с привязкой каждого кандидата к replacement artifact.

Только после successful actual master DB можно обсуждать перенос/архивирование старых batch-папок.
