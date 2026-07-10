# Что сделано

Добавлен `scripts/report_adr003_partial_answer_policy_shadow.py` и тест
`tests/test_report_adr003_partial_answer_policy_shadow.py`.

Скрипт строит report-only диагностику для ADR-003 F2ac: берет частичные кандидаты
из `adr003_partial_answer_opportunities_report.json` и обязательно склеивает их
с blockers из `adr003_frame_calibration_queue_report.json`.

Ключевой смысл: F2z не может сам по себе считаться сигналом готовности к
активному частичному ответу. Нужно увидеть, нет ли рядом danger-adjacent,
source-axis mismatch, blocked renderer/text readiness.

# Как проверялось

Тесты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_partial_answer_policy_shadow.py tests/test_report_adr003_partial_answer_opportunities.py tests/test_report_adr003_frame_calibration_queue.py tests/test_adr003_regex_understanding_moratorium.py
```

Результат: `27 passed`.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3925 passed, 5 skipped, 1 warning`.

`git diff --check`: clean.

Leak check: F2ac output JSON intentionally does not include `client_excerpt`; real
client excerpts from candidate cases are absent from the audit pack report.

На свежем F2ab local reenrich:

- partial candidates input: 2
- joined with queue: 2
- policy candidates: 0
- blocked danger-adjacent: 1
- blocked source-axis mismatch: 1

# Что осталось

Active/text слой не готов. Следующий осмысленный шаг - исправлять proof-axis /
source-axis alignment и отдельно проектировать политику частичных справочных
ответов, но только после semantic review и решения владельца.
