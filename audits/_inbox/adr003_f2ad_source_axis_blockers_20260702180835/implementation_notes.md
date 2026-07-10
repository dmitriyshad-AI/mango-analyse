# Что сделано

Добавлен no-text report-only скрипт `scripts/report_adr003_source_axis_blockers.py`.

Он суммирует current handoff queue из `adr003_frame_calibration_queue_report.json`:
route-only candidates, source-axis blockers, danger-adjacent rows, manager-only
policy blockers и renderer readiness.

# Как проверялось

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_source_axis_blockers.py tests/test_report_adr003_partial_answer_policy_shadow.py tests/test_report_adr003_frame_calibration_queue.py
```

Результат: `22 passed`.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3928 passed, 5 skipped, 1 warning`.

`git diff --check`: clean.

No-text check: F2ad output JSON не содержит `client_excerpt`/`bot_excerpt` и
реальные клиентские реплики из current handoff cases.

На текущем F2ab сырье:

- current handoff rows: 4
- route-only review candidates: 0
- source-axis blocked rows: 2
- alignment review unclear rows: 1
- danger-adjacent rows: 2
- shadow renderer candidates: 0

# Что осталось

Active Ф3 остается `NO-GO`. Следующий реальный workstream - proof/source-axis
alignment и отдельная manager-only policy review, а не route-only включение.
