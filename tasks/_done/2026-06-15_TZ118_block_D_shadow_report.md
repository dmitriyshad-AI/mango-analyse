# TZ-118 Block D: роли, shadow перед primary

Дата: 2026-06-15

## Статус

Выполнен только блок D и остановка на регрейд. Файл `2026-06-15_TZ118_gruppa4_do_primary.md` в рабочих деревьях не найден, реализация сделана по тексту ТЗ из чата Дмитрия.

Primary не включался. AMO/Tallanto/CRM не трогались.

## Ветка

Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tz118_primary`

Ветка: `codex/tz118-group4-primary-d`

База: текущий `main` + перенесённые коммиты TZ-116/TZ-117, потому что в `main` ещё не было `codex_selective` и TZ-117 trace.

## Что изменено

- Добавлен флаг `MONO_ROLE_LOW_INFO_FILTER_MODE=off|mark|filter`, default `off`.
- `off` сохраняет старое поведение без low-info meta.
- `mark` помечает короткие служебные реплики `low_info`, но не меняет роли.
- `filter` заменяет роли коротких служебных реплик на rule-role.
- Runner D пишет `low_info_*` поля в CSV/summary.
- TZ-117 trace умеет `--blocks d` и помечает low-info строки в rationale.
- Добавлены NEG-тесты на default off, mark/filter, короткие содержательные вопросы и D-only trace.

## Замеры

Вход gold23:

`/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_mono_role_gold23_manual_20260615/mono_role_gold23_input.csv`

Baseline TZ-117 D:

- mean per-turn: `0.944275`
- model_fix: `387`
- model_break: `28`
- both_wrong: `31`
- high_conf_wrong: `52`

### Проверенный вариант filter

Runner:

`audits/_inbox/tz118_d_gold23_shadow_filter_20260616_000944/`

Trace:

`audits/_inbox/tz118_d_trace_filter_20260616_002335/`

Итог:

- llm_calls_total: `23`
- codex_cli: `23/23`
- low_info calls: `20`
- low_info turns: `60`
- low_info changed: `18`
- mean per-turn: `0.940051`
- model_fix: `378`
- model_break: `19`
- both_wrong: `40`
- high_conf_wrong: `55`

Решение: `filter` не проходит стоп-условия. Он снижает `model_break`, но ухудшает `both_wrong` и `high_conf_wrong`.

### Кандидат на регрейд: mark

Runner:

`audits/_inbox/tz118_d_gold23_shadow_mark_20260616_002357/`

Trace:

`audits/_inbox/tz118_d_trace_mark_20260616_004151/`

Итог:

- llm_calls_total: `23`
- codex_cli: `23/23`
- пустых `codex_rationale`: `0`
- low_info calls: `20`
- low_info turns: `60`
- low_info changed: `0`
- mean per-turn: `0.9450099565`
- model_fix: `394`
- model_break: `22`
- both_wrong: `24`
- high_conf_wrong: `46`
- avg confidence correct: `0.929681`
- avg confidence errors: `0.912174`

Решение: `mark` лучше baseline по ключевым trace-метрикам и не меняет роли коротких реплик до регрейда.

Осторожность: exact-call accuracy ниже baseline (`3/23` против `6/23`), поэтому primary не включать без регрейда Claude #1.

## Safety

- mode: `shadow`
- primary: не включался
- writeback: не запускался
- AMO/Tallanto/CRM: записей нет
- audio/ASR: не читались и не запускались
- stable_runtime: не изменялся
- модель: только Codex CLI, `OPENAI_API_KEY/OPENAI_ORG_ID/OPENAI_PROJECT` удалены из env
- trace TZ-117 включён: `d_trace.csv`, `d_trace.jsonl`, `d_trace_REPORT.md`, `d_trace_summary.json`

## Проверки

Точечно:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_format.py tests/test_tz116_offline_modes.py tests/test_smoke.py::SmokePipelineTest::test_get_settings_parses_float_env_values`

Результат:

`37 passed, 1 warning`

Полный pytest:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`

Результат:

`3293 passed, 5 skipped, 1 warning`

## llm_calls_total

- `filter`: `23`
- `mark`: `23`
- trace builder: `0`

## Стоп

Останавливаюсь после блока D. B/C/E/A не начинались. Следующий шаг только после регрейда Claude #1 по сырью.
