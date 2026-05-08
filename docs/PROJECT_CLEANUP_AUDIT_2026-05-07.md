# Аудит проекта и план наведения порядка

Дата: 2026-05-07

Цель: подготовить Mango Analyse к следующему этапу - от локального набора скриптов и рабочих папок к стабильному продукту, который можно автономно запускать для своей компании, затем демонстрировать и затем выборочно продавать как сервис.

## Короткий вывод

Проект рабочий и тесты проходят, но папка стала смесью production-кода, runtime-артефактов, сырого аудио, экспортов, backup-БД и исследовательских файлов. Для локальной разработки это терпимо, для SaaS/product-stage - уже нет.

Порядок нужно наводить не удалением "на глаз", а через правила хранения:

1. Код, тесты, документация и deploy остаются в репозитории.
2. Сырой звук, runtime-БД, Excel-выгрузки, Telegram exports и external-worker результаты остаются локальными артефактами вне git.
3. Тяжелые исторические артефакты переводятся в архив с manifest, а не лежат вперемешку в корне.
4. Любое удаление рабочих БД только после проверки, что они уже включены в coverage/index и не являются единственным источником результата.

## Проверка стабильности

Команда:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Результат на 2026-05-07:

```text
152 passed, 1 warning in 11.68s
```

Warning: `urllib3` сообщает, что локальный Python собран с LibreSSL. Это не блокирует тесты, но для SaaS/deploy нужно уйти на управляемый Python 3.12 runtime с OpenSSL.

## Инвентаризация кода

| Область | Количество |
|---|---:|
| Python-модулей в `src/mango_mvp` без `__pycache__` | 51 |
| Python-скриптов в `scripts` | 44 |
| Файлов в `scripts` всего | 48 |
| Тестовых файлов `test_*.py` | 19 |
| Пройденных тестов | 152 |

Основные слои кода:

- `src/mango_mvp/cli.py` - batch CLI.
- `src/mango_mvp/gui.py` - локальный UI для запуска pipeline.
- `src/mango_mvp/services/` - ingest, transcribe, resolve, analyze, export, sync.
- `src/mango_mvp/amocrm_runtime/` - AMO/Tallanto runtime, writeback, deals, dossier, LLM, agent runtime задел.
- `src/mango_mvp/clients/` - внешние API-клиенты.
- `src/mango_mvp/utils/` - аудио, телефоны, кодировки, retry, JSON.
- `scripts/` - рабочие batch/экспорт/исследовательские утилиты.

## Документированность функций

AST-аудит docstring-покрытия:

| Область | Файлов | Модулей с docstring | Public functions | Functions with docstring | Classes | Classes with docstring |
|---|---:|---:|---:|---:|---:|---:|
| `src/mango_mvp` | 51 | 1 | 235 | 0 | 41 | 0 |
| `scripts` | 44 | 1 | 182 | 0 | 5 | 0 |
| `tests` | 20 | 0 | 189 | 0 | 37 | 0 |

Вывод: проект функционально покрыт тестами лучше, чем документацией. Для внутреннего использования это еще можно терпеть, для SaaS и передачи другому разработчику/клиенту - нет.

Минимальный documentation gate перед SaaS:

1. README переписать из MVP-формата в product/runtime guide.
2. Добавить `docs/ARCHITECTURE_CURRENT.md`: какие слои есть сейчас и как они связаны.
3. Добавить `docs/CLI_AND_SCRIPTS_CATALOG.md`: назначение каждой команды и скрипта, какие входы/выходы, какие опасны.
4. Добавить `docs/DATA_MODEL.md`: основные таблицы, статусы pipeline, что значит `done/manual/skipped`.
5. Добавить `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`: что пишем, что не трогаем, какие поля textarea.
6. Добавить `docs/OPERATIONS_RUNBOOK.md`: как запускать UI, batch, R+A, ASR-only, как проверять coverage, как восстанавливаться после падения.
7. Добавить docstring хотя бы к public API сервисов и опасным batch-скриптам.

## Размеры и тяжелые артефакты

Крупные top-level директории/файлы:

| Путь | Размер | Комментарий |
|---|---:|---|
| `2026-03-09--26` | 24G | Основная папка аудио, пока хранить |
| `stable_runtime` | 14G | Главный runtime/history слой, требует индекса и архивации старых прогонов |
| `.git` | 2.4G | Репозиторий раздут локальной историей/объектами, нужен git cleanup отдельно |
| `telegram_exports (2)` | 1.2G | Сырой экспорт Telegram, после обработки лучше вынести из корня |
| `.venv-asrbench` | 1.1G | Локальное окружение benchmark, не часть продукта |
| `_local_archive_20260424` | 1.1G | Архив уже лежит внутри проекта, лучше вынести во внешний архив |
| `2026-03-05-21-06-49-ч1` | 984M | Старый сырой экспорт, кандидат на архив |
| `2026-03-05-21-06-49-ч2` | 985M | Старый сырой экспорт, кандидат на архив |
| `.cache` | 276M | Локальный кэш |
| `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` | 269M | Результат M1, после подтвержденного импорта можно архивировать |
| `mango_mvp.db` | 89M | Основная локальная DB, хранить/бэкапить осознанно |
| `.codex_workers` | 85M | Локальный служебный артефакт Codex |

Тяжелые DB/backups в `stable_runtime`:

- `stable_runtime/ra_missing_all_20260506/ra_missing_all_20260506.db` - 452M, важный итоговый R+A прогон.
- Много `.before_*` DB по 50-193M каждая - это аварийные backup-снимки перед requeue/repair. Их нельзя удалять до фиксации финального покрытия, но после этого нужно архивировать по правилу хранения.
- `stable_runtime/ab_tests/.../test.db` и `stable_runtime/benchmarks/...` - исследовательские артефакты, кандидаты на сжатие/внешний архив.

## Безопасные кандидаты на удаление после подтверждения

Эти артефакты не являются источником бизнес-данных и могут удаляться первыми:

1. Все `__pycache__` - найдено 2358 директорий.
2. Все `.DS_Store` - найдено 7 файлов.
3. `.pytest_cache`.
4. `src/mango_call_mvp.egg-info`.
5. Временные локальные кэши, если не нужен warm cache: `.cache`, `.codex_workers`, `.codex_local`.

Важно: `mango_mvp.db-wal` и `mango_mvp.db-shm` не удалять вручную, пока есть шанс, что DB открыта. Для SQLite WAL нужен корректный checkpoint/остановка процессов.

## Кандидаты на архив, а не прямое удаление

1. `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` и zip рядом - только после подтвержденного import report.
2. `external_m1_jan2025_test300_20260503` в корне и копия в `stable_runtime` - оставить одну canonical-копию или archive manifest.
3. `_local_archive_20260424` - вынести за пределы проекта или на внешний диск.
4. `telegram_exports (2)` - после финального outreach/CRM merge report вынести из корня.
5. Старые source export папки `2026-03-05-21-06-49-ч1/ч2` - если все записи уже перенесены в основную папку и coverage, архивировать.
6. Старые Excel `АКТУАЛЬНО_*.xlsx` - оставить последние production-версии, остальные перенести в `exports/archive/YYYY-MM-DD/` или внешний архив.
7. `stable_runtime/*/*.before_*.db` - после финального coverage v4 оставить не более 1-2 контрольных backup на batch или перенести во внешний архив.

## Что нельзя удалять без отдельной проверки

1. `2026-03-09--26` - основная папка аудио.
2. Любые DB, включенные в `stable_runtime/final_processing_coverage_report_20260507_v3/included_dbs.tsv`.
3. Последние batch DB: `messages35`, `final_asr_tail_1526`, `ra_missing_all_20260506`.
4. `transcripts/`, если какие-то DB ссылаются на файловые расшифровки только там.
5. `.env`, AMO/Tallanto токены, runtime private env - не коммитить и не переносить в клиентский пакет.

## Предлагаемая структура перед SaaS-этапом

```text
Mango analyse/
  src/                      # только код продукта
  tests/                    # тесты
  scripts/                  # поддерживаемые утилиты, с каталогом и статусом
  docs/                     # архитектура, runbook, field mapping, roadmap
  deploy/                   # server/docker/systemd/launchd
  examples/                 # маленькие синтетические примеры
  stable_runtime/           # локальный runtime, игнорируется git, индексируется manifest
  local_data/               # опционально: сырой звук/экспорты, игнорируется git
  exports/                  # актуальные XLSX/CSV отчеты, игнорируется git
  archive/                  # только manifest-ссылки; тяжелый архив лучше вне проекта
```

Для product/SaaS лучше не хранить 24G аудио и 14G runtime внутри рабочей папки репозитория. Практичнее: кодовый репозиторий отдельно, data/runtime volume отдельно.

## План наведения порядка

1. Создать `docs/CLI_AND_SCRIPTS_CATALOG.md` и классифицировать каждый скрипт: production, maintenance, one-off, deprecated.
2. Создать `stable_runtime/MANIFEST.md` или `.json`: какие batch DB финальные, какие backup, какие можно архивировать.
3. Удалить безопасные кэши: `__pycache__`, `.DS_Store`, `.pytest_cache`, egg-info.
4. После финального coverage v4 архивировать `.before_*` DB по retention policy.
5. Вынести Telegram/export/external-worker крупные артефакты из корня в `local_data/` или внешний архив.
6. Обновить `.gitignore`, чтобы новые batch-папки не всплывали как untracked в `git status`.
7. Переписать README: quickstart, runtime profiles, operational commands, safety rules.
8. Добавить docstring/модульные комментарии в критичные сервисы и опасные скрипты.
9. Добавить smoke-команду `make audit` или `scripts/project_audit.sh`, которая запускает tests + coverage DB sanity + size/untracked report.
10. После cleanup сделать git branch/commit как baseline stable product state.

## Git-заметка

Сейчас `git status` показывает много измененных и untracked файлов. Перед большой чисткой нужен отдельный baseline:

1. Не смешивать cleanup, runtime-архивацию и новую разработку в одном коммите.
2. Сначала сохранить текущий кодовый baseline.
3. Потом отдельным коммитом удалить/перенести кэши и generated artifacts.
4. Runtime/data лучше не добавлять в git, а индексировать manifest-файлом.
