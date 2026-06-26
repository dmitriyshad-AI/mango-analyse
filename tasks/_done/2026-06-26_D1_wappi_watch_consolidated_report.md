# D1 Wappi watch consolidated branch

Дата: 2026-06-26

## Задача

Свести Wappi-контур в один кандидат под регрейд: контекст 50 сообщений,
AMO-events resolver, формат заметки, наблюдаемость/stabilization pack и flex
service tier.

## Ветка

- Worktree:
  `/Users/dmitrijfabarisov/Projects/Mango_wappi_watch_consolidated`
- Branch:
  `codex/wappi-watch-consolidated`
- Base:
  `main@a9f80ba`
- HEAD:
  `bc647a1 Default Codex exec service tier to flex`

## Что вошло

- `0ea6af4 Improve Wappi draft context window`
  - Wappi запрашивает 50 последних текстовых сообщений.
  - В prompt идут summary старой части + последние 15 сырых сообщений.
- `abb6799 Fix Wappi note format and place intent`
  - Содержательный черновик выше технической информации.
  - Узкий фикс «место» как территория vs «места» в группе.
- `40318d5 Resolve Wappi chats via AMO events`
  - Resolver через AMO `/events`, а не пустой raw Wappi `crm_entities`.
  - Старый fallback сохраняется только когда AMO-event связи нет.
- `cc0d9e6 Add Wappi draft-loop stabilization ops`
  - `scripts/wappi_draft_loop_ops.py`: passport, daily-report, quality-table.
  - Endpoint-only / smoke / kill-switch тесты.
- `bc647a1 Default Codex exec service tier to flex`
  - Codex exec по умолчанию `flex`, чтобы не жечь fast-tier лимиты.

## Safety boundaries

- Клиенту ничего не отправляется.
- Live-write заметок не включён.
- Единственный write-путь остаётся CLI flag `--live-write`.
- Поиск по Wappi/ops/integration показал только явный `--live-write`; новых
  customer-send путей не добавлено.
- AMO/Tallanto/CRM live-write не выполнялся.

## Проверки

Целевые Wappi tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_amo_wappi_phase1.py \
  tests/test_conversation_intent_plan.py \
  tests/test_codex_exec_service_tier.py \
  tests/test_wappi_draft_loop_ops.py \
  tests/test_wappi_stabilization_smoke.py
# 119 passed in 1.29s
```

Full pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
# 3619 passed, 5 skipped, 1 warning in 82.18s
```

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Ветка стала единым Wappi-кандидатом, а не набором расходящихся веток.
- Контекст, resolver, формат заметки, наблюдаемость и flex теперь в одном HEAD.
- Live-write по-прежнему требует отдельного явного запуска.

Notes:

- Это не доказывает, что постоянный watch сейчас работает в live.
- Перед включением controlled watch нужен отдельный dry-run/passport на живом
  journal/env и явное «да» на live-write заметок.
- Вердикт «в прод» не выносился.
