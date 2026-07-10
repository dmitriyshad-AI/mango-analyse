# Что сделано

Добавлен no-text report-only скрипт `scripts/report_adr003_source_axis_root_causes.py`.

Он классифицирует current handoff blockers поверх F2y/F2ad:

- danger-adjacent;
- manager_only policy;
- missing required slot / partial-answer policy gap;
- platform-axis taxonomy gap.

# Как проверялось

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_source_axis_root_causes.py tests/test_report_adr003_source_axis_blockers.py tests/test_report_adr003_current_handoff_fact_gap.py
```

Результат: `12 passed`.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3931 passed, 5 skipped, 1 warning`.

`git diff --check`: clean.

No-text check: F2ae output JSON не содержит `client_excerpt`/`bot_excerpt` и
реальные клиентские реплики из current handoff cases.

На текущем F2ab сырье:

- route-only active candidates: 0
- missing required slot: 1
- platform-axis taxonomy gap: 1
- danger-adjacent: 2
- manager-only policy: 1

# Что осталось

Active Ф3 остается `NO-GO`. Следующий шаг - чинить platform_current/fact-axis
taxonomy как доказательную ось, а partial-answer policy рассматривать отдельно
после semantic review.
