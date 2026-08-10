# Test evidence

Все команды выполнены из корня worktree без live credentials и внешних
записей.

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_mango_calls_m1_bootstrap.py \
  tests/test_relocate_mango_calls_pipeline.py
72 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_mango_calls_m1_bootstrap.py \
  tests/test_mango_calls_schedule.py \
  tests/test_mango_calls_remote_handoff.py \
  tests/test_mango_calls_two_processes.py \
  tests/test_export_daily_mango_calls_resolve.py \
  tests/test_publish_daily_mango_calls_google.py \
  tests/test_productization_call_processing_readiness.py \
  tests/test_productization_capture_staging.py \
  tests/test_relocate_mango_calls_pipeline.py
358 passed

python3 scripts/skills/tz_lint.py \
  tasks/_running/2026-08-07_TZ_m1_calls_runtime_readiness.md
PASS

python3 scripts/preflight.py --root . \
  --tz tasks/_running/2026-08-07_TZ_m1_calls_runtime_readiness.md
PREFLIGHT: OK

zsh -n scripts/bootstrap_m1_mango_calls.sh
PASS

git diff --check
clean
```

Синтетические отрицательные контроли покрывают dry-run, повтор без изменений,
crash-resume на всех durable checkpoints, оборванную JSONL-запись, полную
проверку SQLite business fields, потерю/подмену файлов, symlink, special file,
внешний hardlink, смену inode, небезопасные права, missing DB, active WAL,
rollback journal policy, unsafe WAL sidecar, URI со спецсимволами и запрет
смешивания CLI-режимов.

Тест с committed записью только в WAL подтверждает STOP до immutable-чтения и
побайтовую неизменность DB/WAL/SHM. Тесты symlink и hardlink sidecar также
подтверждают STOP до открытия SQLite и неизменность внешнего файла.
Отдельный тест rollback journal требует тот же STOP и exact-неизменность
полного pipeline snapshot.
Параметризованный контроль меняет mode основной DB и создаёт внешнюю
hardlink как во время `sqlite_checks`, так и после повторного sidecar-snapshot;
все четыре случая обязаны завершиться STOP.
Ещё один timing-контроль создаёт непустой WAL сразу после финального
чтения DB; последующий sidecar-snapshot обязан вернуть STOP.
