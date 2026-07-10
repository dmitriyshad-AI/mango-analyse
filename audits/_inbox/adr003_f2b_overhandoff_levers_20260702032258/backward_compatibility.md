# Backward Compatibility

## Runtime

Совместимость: `PASS`.

Ф2b не меняет runtime-код прямого пути. Добавлен только read-only скрипт отчёта и тесты.

## Флаги

Новые флаги не добавлялись и не включались.

## Route/Text

Route/text бота не менялись. Отчёт строится по уже готовым M1-транскриптам.

## Тесты

Целевая команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_overhandoff_levers.py tests/test_report_adr003_semantic_frame_eval.py tests/test_direct_path_semantic_frame_shadow.py
```

Результат: `52 passed`.

## Документы

`docs/worktrees_registry.md` обновлён до фактических 6 worktree, чтобы preflight отражал текущую реальность.
