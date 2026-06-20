# Post-rebase verification

Дата: 2026-06-20

## Git

```text
branch: codex/tz-uskoreniya-3-punkta
origin/main: 43134ae1db2c33bdf059e93fb2a85d5fcb32440a
HEAD after rebase: edf3d266a44b74b9c05ac045ae61768f5e21dcf5
merge-base HEAD origin/main: 43134ae1db2c33bdf059e93fb2a85d5fcb32440a
```

## Full pytest

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3365 passed, 5 skipped, 1 warning in 55.03s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -n auto
3365 passed, 5 skipped, 16 warnings in 21.00s
```

## bulk_write determinism

Сравнение: `origin/main` через `git archive` против текущего HEAD.

Стенд: 300 синтетических клиентов, общий подготовленный вход, один `out_root.name`, фиксированный `CustomerTimelineSQLiteStore._now` в тестовом процессе.

```text
before origin/main: 2.7820s
after HEAD: 1.3749s
logical_hash_equal: true
logical_hash: ef2ec0dc32ff8444c7a2a1c9c9cf56212bf42e6a6a2c0d2f61e0eb3911e50d75
logical_row_count: 6000
```

Примечание: фиксация `_now` нужна из-за текущей схемы `customer_id_mappings`, где `created_at`, `updated_at` и `ingestion_run_id` входят в `record_hash`. Это только стабилизация измерителя.

## Parallel regrade determinism

Сравнение: `origin/main` sequential против текущего HEAD `--workers 4`.

Стенд: 2500 синтетических `analysis_json`; запись только во временную sqlite-БД и временный каталог transcript export.

```text
before origin/main sequential: 4.1315s
after HEAD --workers 4: 2.1332s
old_report: code=0 row_count=2500 workers=1
new_report: code=0 row_count=2500 workers=4
rows_equal: true
```

## Read-only

- AMO/CRM/Tallanto write не запускались.
- ASR, Resolve+Analyze по реальным данным, LLM/API не запускались.
- `stable_runtime` не менялся.
- Все benchmark-записи шли во временные каталоги `/var/folders/.../T/tz_uskor_*`.

## Rollback semantics

`bulk_write` меняет семантику неуспешного canonical import: вместо частичных доменных коммитов batch откатывает доменные записи целиком. `start_ingestion_run` оставлен вне batch, а failed `finish_ingestion_run` остаётся в `except`, поэтому failure-run сохраняется отдельно.
