# ADR-003 F2ad: source-axis blockers

Дата: 2026-07-02

Ветка: `codex/adr003-semanticframe-migration`

База входа: `d0946d65`

## Что сделано

Добавлен no-text report-only скрипт:

`scripts/report_adr003_source_axis_blockers.py`

Он берет `adr003_frame_calibration_queue_report.json`, смотрит только текущую handoff-очередь и отвечает на один вопрос: есть ли сейчас route-only кандидат для автономности, или всё заблокировано proof/source-axis, danger-adjacent, manager_only-policy и text-readiness.

Скрипт не читает live, не меняет route/text/runtime и не экспортирует клиентские реплики или тексты ответов.

## Результат на текущем F2ab сырье

Источник:

`audits/_inbox/adr003_f2ab_local_reenrich_20260702173307/source_axis_blockers/adr003_source_axis_blockers_report.json`

Итог:

- current handoff rows: `4`
- route-only review candidates: `0`
- source-axis blocked rows: `2`
- alignment review unclear rows: `1`
- danger-adjacent rows: `2`
- manager_only route rows: `1`
- shadow renderer candidates: `0`
- active readiness: `no_go`

Кейсы:

- `wappi_pair_missing_72h_002#1` -> `blocked_source_axis_mismatch`
- `p0_model_led_pos_how_next#1` -> `blocked_danger_adjacent`
- `p0_model_led_pos_anxiety_level#1` -> `blocked_danger_adjacent`
- `ra1_foton_platform_and_price#1` -> `blocked_manager_only_route`

## Вывод

Комментарий Claude подтвержден по сути, но с уточнением по свежему сырью:

- быстрый `route-only` рычаг сейчас отсутствует;
- оставшийся рычаг автономности не в маршруте, а в доказательном слое;
- прежде чем обсуждать частичный/самостоятельный текст, надо чинить proof/source-axis alignment и отдельно решать manager_only-policy;
- active Ф3 остается `NO-GO`.

## Проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_report_adr003_source_axis_blockers.py \
  tests/test_report_adr003_partial_answer_policy_shadow.py \
  tests/test_report_adr003_frame_calibration_queue.py
```

Результат: `22 passed`.

No-text check: в F2ad artifact нет `client_excerpt`, `bot_excerpt` и реальных клиентских реплик из текущих кейсов.

Полный безопасный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3928 passed, 5 skipped, 1 warning`.

`git diff --check`: чисто.

## Audit pack

`audits/_inbox/adr003_f2ad_source_axis_blockers_20260702180835/`

## Границы

Живой бот, профиль, P0 floor/preblock, Telegram/AMO/Tallanto/CRM не трогались.
