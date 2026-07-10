# ADR-003 F2ac: partial-answer policy shadow

Дата: 2026-07-02

Ветка: `codex/adr003-semanticframe-migration`

Ревизия входа: `8cd2d966`

## Что сделано

Добавлен report-only слой `scripts/report_adr003_partial_answer_policy_shadow.py`.

Он склеивает два уже существующих отчета:

- `adr003_partial_answer_opportunities_report.json` - где F2z нашел частичные кандидаты;
- `adr003_frame_calibration_queue_report.json` - где видны blockers, source-axis и danger-adjacent статусы.

Слой не меняет route/text/runtime, не генерирует клиентский текст и всегда пишет `active_behavior_allowed=false`.

## Зачем

F2z сам по себе был слишком оптимистичным: он нашел 2 `draft_partial_shadow_candidate`, но без склейки с calibration queue было не видно, что эти кандидаты нельзя обсуждать как route-only/partial-answer рычаг.

Новый F2ac отчет делает этот стоп явным.

## Результат на текущем сырье

Источник:

`audits/_inbox/adr003_f2ab_local_reenrich_20260702173307/partial_policy_shadow/adr003_partial_answer_policy_shadow_report.json`

Итог:

- partial draft candidates input: `2`
- joined with queue: `2`
- policy shadow candidates: `0`
- blocked by danger adjacency: `1`
- blocked by source-axis mismatch: `1`
- active readiness: `no_go`

Кейсы:

- `wappi_pair_missing_72h_002#1` -> `blocked_source_axis_mismatch`
- `p0_model_led_pos_how_next#1` -> `blocked_danger_adjacent`

## Проверки

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_report_adr003_partial_answer_policy_shadow.py \
  tests/test_report_adr003_partial_answer_opportunities.py \
  tests/test_report_adr003_frame_calibration_queue.py \
  tests/test_adr003_regex_understanding_moratorium.py
```

Результат: `27 passed`.

Дополнительно: `python3 -m py_compile scripts/report_adr003_partial_answer_policy_shadow.py`.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3925 passed, 5 skipped, 1 warning`.

Проверка на утечку клиентского текста в F2ac artifact: `client_excerpt` удален из выходного JSON; в audit pack нет реплик клиентов из двух кейсов.

`git diff --check`: чисто.

## Семантический вывод

Комментарий Claude про реальный рычаг подтвержден частично, но важнее уточнен:

- price route-only рычаг остается мертвым;
- частичный ответ пока тоже не дает безопасного активного кандидата;
- оба найденных кандидата блокируются соседними отчетами;
- следующий полезный шаг - не включение текста, а исправление proof-axis/source-axis и отдельная политика частичных справочных ответов после semantic review.

## Audit pack

Создан audit pack:

`audits/_inbox/adr003_f2ac_partial_answer_policy_shadow_20260702175642/`

В него добавлены JSON/MD отчета, тестовый вывод, risk/semantic/backward notes.

## Границы

Живой бот, профиль, P0 floor/preblock, Telegram/AMO/Tallanto/CRM не трогались.

Вердикт на active: `NO-GO`.
