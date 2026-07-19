> DONE 2026-07-20 02:13 | ветка main | codex

Ветка: main
Зоны: scripts/preflight.py, scripts/task_move.py, scripts/make_audit_pack.py, scripts/project_now.py, src/mango_mvp/replay_exam/pseudonymizer.py, tests/test_preflight.py, tests/test_task_move.py, tests/test_audit_pack_pii.py, tests/test_project_now.py, docs/RUNBOOK.md, docs/ADR003_E3_ENV_MATRIX.md, pyproject.toml, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_preflight.py tests/test_task_move.py tests/test_audit_pack_pii.py tests/test_project_now.py
Семантический-аудит: нет

# Репозиторный wrapper финального ТЗ 2026-07-20

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-20_TZ_FINAL_chto_vnedryaem.md`.

Владелец явно разрешил выполнить подтвержденную часть ТЗ. После read-only аудита и
независимой архитектурной проверки в этот блок входят только пункты 1, 2, 3, 6,
7, 10 и документальная часть 9. Пункты, затрагивающие live-путь, редеплой,
несуществующую staging-БД или требующие результата M1, не входят.

## Приемка

- preflight не исполняет произвольную команду из ТЗ;
- внешний ТЗ при `--take` не удаляется;
- audit pack маскирует российские и международные телефоны и email;
- PROJECT_NOW fail-soft показывает фактический live snapshot без ложного PASS;
- RUNBOOK начинает чтение решений с актуального хвоста;
- устаревшая env-матрица явно помечена;
- optional dependencies объявлены;
- целевые тесты и полный pytest зеленые.
