# Operations Runbook

Дата: 2026-05-07

Цель: зафиксировать короткий рабочий порядок эксплуатации Mango Analyse до перехода в SaaS/server-архитектуру.

## Базовые проверки

```zsh
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
make test
make audit-fast
```

Полный аудит с тестами:

```zsh
make audit
```

Ожидаемый test gate на 2026-05-07:

```text
152 passed, 1 warning
```

Warning по LibreSSL/urllib3 не блокирует локальную работу, но для server/SaaS runtime нужен управляемый Python с OpenSSL.

## Safety matrix for scripts

Перед запуском незнакомого скрипта проверять:

```text
docs/SCRIPT_SAFETY_MATRIX.md
```

Короткое правило:

- `SAFE_READ_ONLY`, `SAFE_REPORT_WRITES`, `NETWORK_READ_ONLY` можно запускать для диагностики, если понятен output path.
- `CONTROLLED_DOWNLOAD` запускать только в отдельную staging/inbox/quarantine папку.
- `PROCESSING_MUTATES_DB` принадлежит processing-диалогу.
- `CRM_LIVE_GUARDED` по умолчанию должен работать как dry-run/preview.
- `DANGEROUS_LEGACY` не запускать без отдельного чтения кода и подтверждения.

Для live-записи в amoCRM использовать только guarded команды с явным подтверждением:

```zsh
--execute-live-write --live-confirmation WRITE_AMO_LIVE
```

## Проверка покрытия обработки

Текущий финальный отчет:

```text
stable_runtime/final_processing_coverage_report_20260507_v4/
```

Ключевые файлы:

- `summary.json` - агрегированные счетчики.
- `coverage_by_month.tsv` - разрез по месяцам.
- `missing_asr.txt` - actionable ASR gap.
- `missing_full_ra.txt` - actionable R+A gap.
- `manual_not_full_ra.txt` - manual-хвосты Resolve/R+A.
- `excluded_no_asr.txt` - явно исключенные no-ASR звонки.

Текущий status gate:

```text
source_audio = 64867
excluded_no_asr = 35
actionable_source_audio = 64832
asr_done = 64832
missing_asr_actionable = 0
full_ra = 64364
missing_full_ra_actionable = 468
manual_not_full_ra = 468
asr_no_full_ra_non_manual = 0
```

Вывод: actionable ASR закрыт полностью. Остался один крупный технический долг: `468` manual R+A хвостов.

## Пересборка coverage report

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_final_processing_coverage_report.py \
  --project-root . \
  --source-dir 2026-03-09--26 \
  --start-date 2025-01-01 \
  --end-date 2026-05-31 \
  --baseline-included-dbs stable_runtime/final_processing_coverage_report_20260507_v3/included_dbs.tsv \
  --extra-db stable_runtime/mar2026_client_asr_tail_129_20260507/mar2026_client_asr_tail_129_20260507.db \
  --exclude-csv stable_runtime/mar2026_client_asr_tail_129_20260507/exclusions/manager_manager_excluded.csv \
  --out-root stable_runtime/final_processing_coverage_report_YYYYMMDD_vN
```

Правило: новый coverage report считается валидным только если `errors=[]` в `summary.json`.

## ASR-only batch запуск

Для UI batch использовать launcher вида:

```zsh
./stable_runtime/run-ui-<batch-name>.sh
```

В UI нажимать:

```text
Параллельный pipeline старт
```

Для ASR-only launchers должны быть отключены:

```text
Resolve = 0
Analyze = 0
Sync = 0
```

Headless MLX ASR из Codex sandbox не считать надежным: 2026-05-07 был crash Metal/MLX. Для ASR использовать обычный UI/Terminal macOS session.

## Resolve+Analyze batch запуск

Для R+A использовать supervisor scripts вида:

```zsh
./stable_runtime/start-<batch>-resolve-analyze-YYYYMMDD.sh
```

Стандартная конфигурация:

```text
RESOLVE_WORKERS=2
ANALYZE_WORKERS=6
STAGE_LIMIT=20
ANALYZE_PROVIDER=codex_cli
CODEX_ANALYZE_MODEL=gpt-5.4-mini
RESOLVE_LLM_PROVIDER=codex_cli
CODEX_RESOLVE_MODEL=gpt-5.4-mini
```

После достижения `actionable=0` workers можно остановить вручную, чтобы не ждать долгий `max_idle_cycles`; финальный статус обязательно записать в `final_status.json`.

## Что нельзя делать без отдельного подтверждения

- Не удалять `2026-03-09--26`.
- Не удалять DB из актуального `included_dbs.tsv`.
- Не удалять `mango_mvp.db-wal`/`*.db-wal` во время активной работы; сначала checkpoint/остановка процессов.
- Не запускать AMO writeback без dry-run и quality gate.
- Не распознавать manager-manager звонки, если нет отдельного бизнес-запроса.

## Следующий operational gate

Перед пересборкой contact-layer и AMO/ROP пакетов закрыть или явно финализировать `468` manual R+A хвостов.
