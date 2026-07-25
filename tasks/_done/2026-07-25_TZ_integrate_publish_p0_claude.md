> DONE 2026-07-25 14:36 | ветка codex/ai-employee-final | codex

> TAKE 2026-07-25 14:22 | ветка codex/ai-employee-final | codex

Ветка: codex/ai-employee-final
Зоны: scripts/publish_snapshot/, src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/canonical_readonly_import.py, src/mango_mvp/customer_timeline/ids.py, tests/test_publish_snapshot_tooling.py, tests/test_bot_safe_runtime_context.py, tests/test_customer_timeline_bot_safe_summary.py, tests/test_customer_timeline_canonical_readonly_import.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_publish_snapshot_tooling.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_canonical_readonly_import.py
Семантический-аудит: да

# Интеграция publish safety и P0 identity-ref из блока Claude

## Цель

Перенести в единственную интеграционную ветку стабильный и проверенный блок из грязного
`Mango analyse`, не меняя ту папку: явное управление рестартом readers при flip/rollback,
симметричный lsof-гейт rollback и нормализацию `customer:customer:*` конфликтных ссылок.

## Приёмка

- исходный dirty diff не меняется во время переноса;
- восемь чистых файлов применяются механически, один тестовый конфликт разрешается вручную;
- P0/brand/ПДн полы не ослаблены;
- точечный и полный pytest зелёные;
- один коммит в `codex/ai-employee-final`, без live/data write и без удаления веток/worktree.

## СТОП

- исходный dirty diff изменился во время переноса;
- появился конфликт в продуктовом коде;
- любой P0/brand/ПДн тест стал красным;
- `main` или live требуют записи для завершения блока.
