# Отчёт: три ускорения

Дата: 2026-06-20
Ветка: `codex/tz-uskoreniya-3-punkta`
База: `0e0c7b7 customer-profile: record combined cache rerun`

## Что сделано

1. `pytest-xdist` добавлен как dev-зависимость:
   - `pyproject.toml`: `[project.optional-dependencies].dev`
   - `uv.lock`: синхронизирован через `uv lock`
   - добавлен pytest-маркер `serial`, но тесты им не помечались: реальных xdist-падений не найдено.

2. `bulk_write` подключён в каноническом импортёре customer timeline:
   - `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
   - `start_ingestion_run` оставлен вне batch, чтобы failed-run можно было сохранить в `except`
   - последовательный порядок телефонов и single-writer не менялись
   - основной write-loop и successful `finish_ingestion_run` выполняются внутри `with store.bulk_write():`

3. Регрейд `migrate-analysis-schema` распараллелен по вычислению:
   - `src/mango_mvp/services/analyze.py`: top-level `migrate_analysis_payload(...)` + сериализуемый snapshot звонка
   - `src/mango_mvp/cli.py`: `--workers N`, default `1`
   - в `ProcessPoolExecutor` уходит только нормализация payload
   - запись `analysis_json`, экспорт файлов и `session.commit()` остались однопоточными в родительском процессе

## Коммиты

- `c5f4160 Add pytest-xdist dev dependency`
- `73f9b24 Batch canonical timeline import writes`
- `df43ad0 Parallelize analysis schema migration compute`

## Замеры и детерминизм

### 1. pytest-xdist

База до xdist:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3365 passed, 5 skipped, 1 warning in 51.78s
```

Проверка гонок:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -n 2
3365 passed, 5 skipped, 2 warnings in 29.66s
```

Итоговый параллельный режим:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -n auto
3365 passed, 5 skipped, 16 warnings in 21.01s
```

Реальных serial-тестов не выявлено.

### 2. bulk_write canonical timeline

Focused:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_customer_timeline_canonical_readonly_import.py \
  tests/test_customer_timeline_store.py \
  tests/test_customer_timeline_read_api.py
28 passed in 0.83s
```

Полный pytest после пункта:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -n auto
3366 passed, 5 skipped, 16 warnings in 20.58s
```

Замер на 300 синтетических клиентов:

```text
before c5f4160: 2.7756s
after worktree: 1.3913s
logical_hash_equal: true
logical_hash: 9277523d1f5c774344a517d590a1f1b2464618f80ccf934b513ca39434ae1ffa
```

Хэшировались стабильные доменные таблицы по ключам и `record_hash`; служебные `audit_log`, `schema_migrations`, `ingestion_runs` не использовались как критерий содержимого из-за runtime timestamps/metadata.

### 3. Parallel regrade

Focused:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_analysis_schema.py tests/test_cli.py
27 passed, 1 warning in 1.61s
```

Полный pytest после пункта:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -n auto
3368 passed, 5 skipped, 16 warnings in 20.44s
```

Замер на 2500 синтетических `analysis_json`:

```text
before 73f9b24 sequential: real 4.62s
after df43ad0 --workers 4: real 2.47s
old_report: scanned=2500 updated=2500 errors=0
new_report: scanned=2500 updated=2500 errors=0 workers=4
rows_equal: true
```

## Read-only и запретные зоны

- AMO/CRM/Tallanto write не запускались.
- `stable_runtime` в рабочем репозитории не менялся.
- ASR, Resolve+Analyze по реальным данным, LLM/API не запускались.
- Замеры писали только во временные каталоги `/tmp/...`.
- Для проверки xdist пакет `pytest-xdist` был установлен в пользовательскую Python-среду; в репозитории он зафиксирован только как dev-зависимость.

## Остаточные риски

- `pytest -n auto` теперь явно быстрее, но не включён в `addopts`: команда остаётся явной, чтобы CI/локальные сценарии не поменялись молча.
- `bulk_write` меняет failure-семантику canonical import: при исключении доменные записи внутри batch откатываются, а failed-run сохраняется отдельно. Это ожидаемое поведение для batch-write, но отличается от прежних частичных коммитов.
- `migrate-analysis-schema --workers N` копирует payload/transcript в процессы; на очень больших batch стоит подбирать `--workers` и `--limit`, а не ставить максимум CPU.
