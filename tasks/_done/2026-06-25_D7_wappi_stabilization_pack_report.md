# D7 Wappi stabilization pack v2 — отчёт

Дата: 2026-06-25  
Ветка: `codex/wappi-stabilization-pack`  
База: `codex/wappi-context-window`, HEAD `40318d5`  
Режим: read-only по продукту, AMO/Tallanto/CRM write = 0, отправка клиентам = 0, live-write заметок не включался.

## 1. Сверка базы 40318d5

Подтверждено по коду:

- Контекст Wappi до 50 сообщений: `scripts/run_amo_wappi_draft_loop.py` задаёт `--chat-limit` по умолчанию 50; `src/mango_mvp/integrations/draft_loop.py` хранит `chat_limit=50`.
- Prompt-окно: сырые последние сообщения + `Ранее в диалоге:` для старой части уже есть в `src/mango_mvp/integrations/draft_loop.py`, покрыто `tests/test_draft_loop.py` и `tests/test_run_amo_wappi_draft_loop.py`.
- Заметка: текст черновика сверху, техинформация снизу через `build_draft_note_text()` в `src/mango_mvp/integrations/amo_wappi_phase1.py`.
- Фикс `место/места`: промпт содержит запрет обещать места без проверки, smoke проверяет `место занятий` != `места есть`.
- AMO-event resolver на месте: причины `amo_chat_event_sequence_unconfirmed`, `amo_chat_event_rate_limited`, `amo_chat_event_ambiguous`, `closed_lead`, `max_phone_missing` покрыты в `tests/test_run_amo_wappi_draft_loop.py`.

## 2. Что реализовано

### Read-only ops script

Добавлен `scripts/wappi_draft_loop_ops.py`:

- `passport` — JSON-паспорт рантайма: путь, ветка, commit, PID/команда/launch path/screen, реальный env текущего процесса с префиксами `TELEGRAM_*` и `DRAFT_LOOP_*`, секреты редактируются.
- `daily-report` — суточный отчёт по journal/heartbeat/state, включая resolver-причины:
  `amo_chat_event_sequence_unconfirmed`, `amo_chat_event_rate_limited`, `amo_chat_event_ambiguous`, `brand_mismatch`, `closed_lead`, `max_phone_missing`, `quarantined_pairs`, `pending_notes`, плюс короткие алиасы `quarantined`, `pending`.
- `quality-table` — CSV для ручной оценки заметок с колонками:
  `created_at, note_id, lead_id, contact_id, profile_id, chat_suffix, message_id, route, safety_flags, draft_text, manager_reply_if_seen, manual_label, comment`.

Артефакты локального read-only прогона записаны в `.codex_local/wappi_stabilization/` и не коммитятся, потому что могут содержать клиентские тексты.

### Endpoint-only contract

Расширен fake/contract тест `tests/test_amo_wappi_phase1.py::test_ai_office_note_client_posts_only_server_endpoint`:

- проверяется только endpoint заметки `/api/integrations/amocrm/leads/{lead_id}/notes`;
- запрещены соседние пути `/tasks`, `/custom_fields`, `/messages`;
- payload только `{"text": ...}`;
- транспорт fake, реального POST нет.

### Детерминированный smoke без LLM

Добавлен `tests/test_wappi_stabilization_smoke.py`:

- `место занятий` не превращается в обещание мест;
- `есть места/бронь/запись` уходит на менеджера без обещания;
- статус оплаты не подтверждается по одному источнику;
- возврат денег/P0 уходит в `manager_only`;
- prompt-контракт содержит бренд, документы, семейный телефон, запрет обещать места.

LLM-путь не запускался и не считается детерминированным тестом.

## 3. Паспорт рантайма на текущем окружении

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/wappi_draft_loop_ops.py passport --out .codex_local/wappi_stabilization/passport.json
```

Факт по текущему окружению:

- `process_found`: `false`
- `pid`: `null`
- `runtime_env.source`: `process_not_found`
- `runtime_env.values`: `{}`
- `profiles_count`: `4`
- `profile_channels`: `{"telegram": 2, "max": 2}`
- `pairs_count`: `3`
- `stop_active`: `false`

Отдельная проверка `ps` показала активный watchdog `mango_draft_loop_watchdog`, но активного процесса `run_amo_wappi_draft_loop.py` нет. Поэтому паспорт не подставляет ожидаемые флаги и честно показывает отсутствие PID/env текущего draft-loop процесса.

## 4. Ежедневный отчёт

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/wappi_draft_loop_ops.py daily-report --out .codex_local/wappi_stabilization/daily_report.json
```

Факт по текущему journal/heartbeat:

- heartbeat есть, `last_cycle_at=2026-06-24T09:03:54.668704+00:00`;
- heartbeat stale: `fresh=false`;
- за последние 24 часа: `rows_considered=0`, `draft_created=0`, `notes_written=0`, `errors=0`, `pair_missing=0`;
- resolver buckets присутствуют все, сейчас значения нулевые:
  `amo_chat_event_sequence_unconfirmed=0`, `amo_chat_event_rate_limited=0`, `amo_chat_event_ambiguous=0`, `brand_mismatch=0`, `closed_lead=0`, `max_phone_missing=0`, `quarantined_pairs=0`, `pending_notes=0`, `quarantined=0`, `pending=0`.

## 5. Таблица качества

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/wappi_draft_loop_ops.py quality-table --out .codex_local/wappi_stabilization/quality_table.csv
```

Факт:

- собрано строк: `5`;
- обязательные колонки на месте;
- `chat_id` не печатается целиком, только `chat_suffix`;
- `manual_label` и `comment` пустые для ручной разметки;
- `manager_reply_if_seen` подтягивается из manager-edit log при наличии совпадения по `profile_id/chat_id/message_id`.

## 6. Kill-switch

Локальный stop-file уже есть в основании:

- default: `~/.mango_secrets/STOP_DRAFT_LOOP`;
- подключается через `--stop-file`;
- при наличии файла цикл останавливается до обработки;
- покрыто существующими тестами `tests/test_draft_loop.py`.

Серверный kill-switch/health endpoint не реализован и не выдумывался.

GAP / мини-ТЗ для отдельного блока:

1. Добавить в AI Office read-only health/config endpoint с полем `draft_loop_stop=true/false`.
2. Wappi-loop перед Wappi/LLM/AMO-note стадиями читает этот флаг через safe transport.
3. При `true` или недоступности endpoint в режиме write-note — остановить создание/запись заметок и писать journal-событие `server_stop_active` или `server_stop_unavailable`.
4. Покрыть fake/contract тестом без реального POST.
5. Не менять текущий локальный stop-file, оставить его аварийным ручным рубильником.

## 7. Тесты

Целевой Wappi/D7 набор:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_amo_wappi_phase1.py \
  tests/test_amo_wappi_transport.py \
  tests/test_wappi_draft_loop_ops.py \
  tests/test_wappi_stabilization_smoke.py \
  tests/test_draft_prompt_builder.py \
  tests/test_bot_policy_v2.py \
  tests/test_rules_engine.py
```

Результат: `175 passed in 1.50s`.

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3619 passed, 5 skipped, 1 warning in 83.93s`.

Warning: `urllib3` сообщает, что системный Python собран с LibreSSL 2.8.3. На проверяемую логику Wappi это не влияет.

## 8. Семантический аудит

Вердикт: `PASS_WITH_NOTES`.

Что прошло:

- Клиенту ничего не отправляется.
- AMO/Tallanto/CRM write не выполнялся.
- Endpoint-only тест не делает реальный POST и проверяет только заметку.
- Детерминированный smoke закрывает основные бизнес-риски: места, документы, оплату, возврат/P0, бренд, семейный телефон.
- Паспорт не маскирует отсутствие процесса ожидаемыми флагами.
- Daily report теперь видит причины resolver, карантин и pending.

Неблокирующие риски:

- Активного draft-loop процесса сейчас нет, поэтому реальный env живого процесса не продемонстрирован на running-процессе; механизм покрыт unit-тестом и текущий паспорт честно показывает `process_not_found`.
- LLM dry-run не запускался; это должен быть отдельный отчёт, не часть deterministic smoke.
- Серверного kill-switch нет; зафиксирован GAP и мини-ТЗ.

Regression checks, добавленные в код:

- `tests/test_wappi_draft_loop_ops.py` — паспорт реального env, resolver reasons, quality CSV.
- `tests/test_wappi_stabilization_smoke.py` — детерминированный smoke без LLM.
- `tests/test_amo_wappi_phase1.py` — endpoint-only fake contract.

Вердикт «в прод» не выносится; требуется регрейд Claude #1 по сырью.
