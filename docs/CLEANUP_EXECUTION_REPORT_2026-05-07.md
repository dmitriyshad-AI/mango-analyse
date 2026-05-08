# Cleanup Execution Report

Дата: 2026-05-07

## Выполнено

- Удалены безопасные generated-кэши: `2295` директорий `__pycache__`, `7` файлов `.DS_Store`, `1` `.pytest_cache`, `1` `egg-info`.
- Добавлен повторяемый audit tool: `scripts/project_audit.py`.
- Добавлены Makefile targets: `make test`, `make audit`, `make audit-fast`.
- Расширен `.gitignore` под новые локальные runtime/batch артефакты.
- Создан audit report: `stable_runtime/project_audit_20260507_125704`.
- Создан human-readable scripts catalog.
- Создан runtime/data manifest.

## Проверка

- `make audit` завершился успешно.
- Внутри audit запущен pytest: `passed`.
- Последний известный полный pytest перед cleanup: `152 passed, 1 warning`.

## Что осталось осознанно не удаленным

- Рабочие аудио, transcript, Excel, DB и batch folders.
- Backup DB `.before_*` до финального coverage v4.
- `codex_home/tmp/arg0` крупные служебные бинарники: выделены как следующий безопасный cleanup-кандидат, но требуют отдельного подтверждения/остановки активных процессов.

## Текущий риск

- Документация функций слабая: по AST-аудиту public functions/classes почти без docstring.
- `git status` по коду стал чище за счет `.gitignore`, но остаются осознанные изменения кода/доков и несколько новых production scripts.

## Следующий gate

Перед переходом к обработке хвостов нужно либо удалить служебные `codex_home/tmp` артефакты, либо оставить их до завершения всех активных Analyze-процессов. Рабочие DB/audio не трогать.

## Post-test cache cleanup уточнение

После `make audit` pytest заново создал часть `__pycache__`/`.pytest_cache`. Я повторно удалил source/test/script кэши:

- `src/tests/scripts` `__pycache__`: `0`.
- `.pytest_cache`: `0`.
- `.DS_Store`: `0`.
- `egg-info`: `0`.
- Осталось `63` `__pycache__` внутри `stable_runtime/venv_stable.broken_20260407`. Это уже не код проекта, а старая broken-venv папка; ее лучше удалять/архивировать целиком отдельным шагом, а не чистить частично.

## Финальная проверка после правки audit tooling

- `Makefile` и `scripts/project_audit.py` переведены на `PYTHONDONTWRITEBYTECODE=1`, чтобы тесты/audit больше не создавали `__pycache__` в исходниках.
- Повторный `make test`: `152 passed, 1 warning in 10.12s`.
- После повторной очистки: `src/tests/scripts` `__pycache__ = 0`, `.pytest_cache = 0`.

Note: local tooling may recreate a few `src/mango_mvp/**/__pycache__` directories after checks. They are ignored by git and now prevented in project test/audit commands via `PYTHONDONTWRITEBYTECODE=1`; if they appear again, they are safe generated cache, not source state.

## March 2026 ASR tail preparation

- Strict no-ASR tail from coverage v3 was classified: `164` total, `129` client/external-phone calls, `35` manager-manager calls.
- User requested not to recognize manager-manager calls for now, so those 35 were written to an explicit exclusion list.
- Prepared ASR-only UI batch: `stable_runtime/mar2026_client_asr_tail_129_20260507`.
- Headless MLX attempt from Codex sandbox failed on Metal initialization before processing. DB was recovered to clean state: `129 pending`.

## R+A and coverage update 2026-05-07 13:55 MSK

- Closed Jan 2025 M1 test300 R+A: `300/300 analysis_done`, `manual=0`, `dead_letter=0`.
- Closed March 2026 client ASR tail R+A: `129/129 analysis_done`, `manual=0`, `dead_letter=0`.
- Rebuilt coverage v4: `stable_runtime/final_processing_coverage_report_20260507_v4/`.
- Coverage v4 status: actionable ASR gap `0`, asr-no-RA non-manual gap `0`, remaining actionable R+A gap `468` manual tails.
- Added reusable coverage builder: `scripts/build_final_processing_coverage_report.py`.
- Added operations docs: `docs/OPERATIONS_RUNBOOK_2026-05-07.md`, `docs/RUNTIME_RETENTION_POLICY_2026-05-07.md`.
