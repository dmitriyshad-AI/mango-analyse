# D7 Wappi Context Window Report

Дата: 2026-06-25
Ветка: `codex/wappi-context-window`
Режим: Wappi/AMO dry-run only, live-write не запускался.

## Что изменено

- Wappi-loop теперь запрашивает 50 последних сообщений и берёт в работу только текстовые.
- В prompt-контекст для Wappi кладётся:
  - одна extractive-строка `Ранее в диалоге:` по старой части;
  - последние 15 сырых строк в порядке хронологии с префиксами `Клиент:` / `Ответ:`.
- Служебные маркеры (`message_id`, `chat_id`, `profile_id`, `lead_id`, `source_system`, JSON-скобки) отбрасываются из prompt history.
- Для активного бренда простая keyword-защита не переносит строку явно другого бренда в старую выжимку.
- Для не-Wappi каналов старый лимит recent history сохранён.

## Prompt Context До/После

Синтетический пример, без клиентских ПДн.

До:

```text
Клиент: последний сырой текст 40
...
Клиент: последний сырой текст 49
```

Проблема: первое сообщение `Сын в 7 классе, интересует физика онлайн` отрезалось и не попадало в prompt.

После:

```text
Ранее в диалоге: Клиент: Сын в 7 классе, интересует физика онлайн; Клиент: старый уточняющий текст 3; Клиент: старый уточняющий текст 4
Клиент: последний сырой текст 35
...
Клиент: последний сырой текст 49
```

Служебная строка вида `message_id=... chat_id=... profile_id=... lead_id=... source_system={...}` и строка явно чужого бренда в этот контекст не попали.

## Тесты

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py
```

Результат:

```text
42 passed in 0.98s
```

Покрыто:

- 50 Wappi-сообщений запрашиваются с `mark_all=False`.
- Старый первый клиентский контекст попадает в extractive summary.
- Последние 15 сообщений идут сырыми и в порядке.
- Служебные id/JSON/source markers не попадают в prompt history.
- Wappi direct-path видит summary + 15 сырых строк; обычный Telegram остаётся на старом лимите.
- Контур Wappi draft-loop по-прежнему помечает `sends_client_replies=False`.

## Dry-run

Первая попытка read-only dry-run зависла на SSL-handshake при чтении Wappi messages; процесс остановлен вручную до LLM/AMO-note ветки.

Успешная повторная команда:

```bash
AMO_WAPPI_HTTP_TIMEOUT_SEC=8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_amo_wappi_draft_loop.py --once --dry-run --chat-limit 5 --local-dir .codex_local/wappi_context_window_dryrun_retry --stop-file .codex_local/wappi_context_window_dryrun_retry/STOP --timeout-sec 60
```

Итог:

```json
{
  "dry_run": true,
  "bot_calls": 1,
  "processed": 0,
  "skipped": 133,
  "auto_resolver_counts": {"not_enabled": 13},
  "auth_error": false
}
```

Проверка журнала `.codex_local/wappi_context_window_dryrun_retry/journal.jsonl` агрегатами:

```text
journal_rows=14
events={'pair_missing': 13, 'draft_created': 1}
statuses={'skipped': 13, 'dry_run': 1}
note_written_events=0
```

AMO note=0. Client sends=0: этот скрипт не имеет клиентского send-пути, а `public_pilot_mode.sends_client_replies=False` покрыт тестом.

## Semantic Review

Статус: `PASS_WITH_NOTES`.

Что прошло:

- Старая часть не передаётся сырой простынёй, а остаётся короткой extractive-выжимкой.
- Summary не придумывает факты: берутся только очищенные строки из истории.
- Prompt не получает служебные id, JSON и source markers.
- Явный чужой бренд в старой части не переносится в выжимку.
- Live-write не запускался.

Остаточный риск:

- Brand-filter простой keyword-based; он защищает от явных строк чужого бренда, но не является полноценным семантическим бренд-классификатором.
- Если полезное старое сообщение содержит служебный маркер внутри текста, строка отбрасывается целиком, чтобы не рисковать утечкой id.

Вердикт "в прод" не выносился.
