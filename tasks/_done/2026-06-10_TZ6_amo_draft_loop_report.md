# TZ6 AMO + Wappi draft loop report

Дата: 2026-06-10
Ветка: main

## Статус

Реализован offline/dry-run каркас цикла:

`Wappi polling -> bot draft -> AMO draft note`

Код не отправляет сообщения клиентам. AMO write ограничен единственным note endpoint и тестовыми allowlist/pair checks.

## Шаг 0 Wappi

Документ: `D1_audit_backlog/WAPPI_DRAFT_LOOP_STEP0_2026-06-10.md`.

Подтверждено living API:

- `GET /tapi/sync/chats/get`
- `GET /tapi/sync/messages/get`
- стабильный `id`
- `fromMe`
- `chatId`
- `type`
- `time`
- `senderName`/`contact_name`

Параметр `date` на живой выборке дал нестабильный результат с дублями, поэтому v1 использует fallback из ТЗ: последние K сообщений + diff по `(profile_id, chat_id, message_id)`.

Не закрыто без Дмитрия: ручная проверка, что сообщение, отправленное именно из интерфейса AMO по тестовой сделке 49832125, видно в Wappi history как исходящее. В истории видны `fromMe=true` с `from_where=api/phone`, но это не доказывает AMO UI.

## Коммит 1

Вынесена чистая сборка контекста:

- `src/mango_mvp/pilot_context_assembly.py`
- `scripts/run_telegram_public_pilot_bots.py`
- `tests/test_telegram_public_pilot_bots.py`

Что проверено:

- публичный Telegram runtime даёт тот же context, что вынесенный сборщик;
- draft-loop mode выставляет `sends_client_replies=False`;
- новый модуль не импортирует transport отправки Telegram.

## Коммит 2

Добавлено ядро draft loop:

- `src/mango_mvp/integrations/amo_wappi_transport.py`
- `src/mango_mvp/integrations/draft_loop.py`
- расширен `src/mango_mvp/integrations/amo_wappi_phase1.py`
- тесты: `tests/test_amo_wappi_transport.py`, `tests/test_draft_loop.py`, `tests/test_amo_wappi_phase1.py`

NEG:

- default-deny блокирует неизвестный GET и side-effect Wappi `mark_all=true`;
- голый `chat_id` в pairs запрещён;
- один `chat_id` в разных `profile_id` не склеивается;
- auto-candidate не пишет note без явной пары;
- STOP не вызывает бота и не пишет AMO;
- crash `note_pending` -> одна retry-попытка тем же `bot_draft_text`;
- state loss после `note_written` в journal не даёт второго note;
- не-текст и debounce не вызывают бота;
- manager edit window: superseded draft может стать `unedited`, одно исходящее засчитывается одному draft.

## Коммит 3

Добавлен CLI и документация:

- `scripts/run_amo_wappi_draft_loop.py`
- `docs/AMO_WAPPI_DRAFT_LOOP_README_2026-06-10.md`
- `tests/test_run_amo_wappi_draft_loop.py`

CLI:

- `--once`
- `--loop`
- `--dry-run` по умолчанию
- `--live-write` только после явного запуска

State/journal:

- `~/.mango_local/draft_loop/state.json`
- `~/.mango_local/draft_loop/journal.jsonl`
- `~/.mango_local/draft_loop/manager_edits.jsonl`

Секреты и пары:

- `~/.mango_secrets/amo_wappi.env`
- `~/.mango_secrets/amo_wappi_profiles.json`
- `~/.mango_secrets/draft_loop_pairs.json`
- `~/.mango_secrets/amo_wappi_phase1.json`

## Проверки

Точечные:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_amo_wappi_phase1.py \
  tests/test_amo_wappi_transport.py \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_telegram_public_pilot_bots.py
```

Результат: `64 passed`.

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `2884 passed, 2 skipped, 1 warning`.

## Что не делалось

- AMO note live не писался.
- Сообщения клиентам не отправлялись.
- Wappi send API не использовался.
- `stable_runtime` не менялся.
- `draft_loop_pairs.json` и `amo_wappi_phase1.json` не создавались, потому что содержат ПДн/allowlist и должны быть подтверждены Дмитрием.

## Нужно от Дмитрия для live-пробы

1. Создать/подтвердить `~/.mango_secrets/draft_loop_pairs.json` для тестового чата и сделки 49832125.
2. Подтвердить allowlist для AMO note в `~/.mango_secrets/amo_wappi_phase1.json` или принять fallback из пары.
3. Отправить одно тестовое сообщение из AMO UI, чтобы проверить видимость исходящего в Wappi history.
4. После этого запустить live-пробу:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_amo_wappi_draft_loop.py --once --live-write
```

