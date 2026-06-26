# D1 Wappi Watch On Main

Дата: 2026-06-26

Ветка: `codex/wappi-watch-on-main`
Base: `main@b543991`

## Что сделано

Собрана чистая Wappi-ветка поверх актуального локального `main`, где уже включен `TELEGRAM_INTENT_MODEL_LED` в `pilot_gold_v1`.

Перенесены функциональные коммиты:

- `184ea0f` (`0ea6af4`) — расширенное окно Wappi-контекста: старая часть как краткая выжимка + последние 15 сырых сообщений.
- `4a3866a` (`abb6799`) — формат AMO-заметки: сначала текст черновика, затем технический блок; исправление смысла "место" как территория.
- `1024b05` (`40318d5`) — auto-resolver через AMO events: Wappi-chat/message связывается с AMO lead/contact по `incoming_chat_message`, origin и узкому временному окну; fallback остается только когда AMO-событий нет.
- `6711e64` (`cc0d9e6`) — Wappi stabilization pack: passport, daily-report, quality table, endpoint-only guard, smoke tests, kill-switch/quarantine.
- `f0a70e3` (`a436875`) — Codex exec service tier по умолчанию `flex`, чтобы не тратить fast-tier лимиты без явного `MANGO_CODEX_SERVICE_TIER=fast`.

Старый отчётный коммит `6919ac5` не переносился; этот файл фиксирует состояние ветки после пересборки поверх `main`.

## Safety

- Live-write не запускался.
- AMO/Tallanto/CRM write: 0.
- Клиентам ничего не отправлялось.
- Основной live Telegram bot и `stable_runtime` не трогались.
- Единственный Wappi write-entry по-прежнему требует явный CLI-флаг `--live-write`.
- Клиентские send-path не добавлялись.

## Проверки

Целевые тесты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_wappi_draft_loop_ops.py \
  tests/test_wappi_stabilization_smoke.py \
  tests/test_amo_wappi_phase1.py \
  tests/test_conversation_intent_plan.py \
  tests/test_codex_exec_service_tier.py
```

Результат: `119 passed in 1.27s`

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3632 passed, 5 skipped, 1 warning in 78.84s`

`git diff --check`: clean.

## Остаточные риски

- Это `formal_pass`, не `semantic_pass`: качество реальных Wappi-черновиков должно оцениваться по live/dry-run таблице качества.
- Controlled watch с live-write заметок не включался в этом блоке. Для запуска нужен отдельный явный гейт: свежий passport, daily-report, auto-resolver dry-run, список разрешенных профилей/пар и подтверждение AMO note allowlist/allow-all.
- Auto-resolver через AMO events проверен тестами и прежними отчётами, но перед расширением write-охвата нужен наблюдаемый dry-run на живом журнале.

## Вывод

Ветка готова к регрейду как кандидат на перенос Wappi-watch пакета поверх актуального `main`. Прод-вердикт и live-write не заявлялись.
