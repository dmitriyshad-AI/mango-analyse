# TZ-6 live probe report — Wappi -> bot draft -> AMO note

Дата: 2026-06-10

## Scope

Тестовая сделка: `49832125`.

Клиентам ничего не отправлялось. Единственная разрешённая live-запись по ТЗ — AMO note в allowlist-сделку; она не была выполнена из-за отсутствия действующего AMO bearer.

## 1. Wappi chat discovery

Telegram-профиль Фотона подтверждён:

- `profile_id`: `ec2eed50-b55f`
- `brand`: `foton`

Найден тестовый чат Дмитрия:

- `chat_id`: `290027369`
- последний входящий текст: `Тест бота: интересует математика, 7 класс`
- Wappi `message_id`: `18217`
- `fromMe=false`, `type=text`, `from_where=phone`

Также в истории есть автоисходящее Wappi:

- `message_id`: `18218`
- `fromMe=true`, `from_where=api`

Это подтверждает чтение Wappi history, но НЕ закрывает гипотезу 0в: сообщение именно из интерфейса AMO ещё не проверено.

## 2. Config created outside repo

Созданы внешние файлы с правами `600`:

- `~/.mango_secrets/draft_loop_pairs.json`
- `~/.mango_secrets/draft_loop_profiles_foton_test.json`
- `~/.mango_secrets/amo_wappi_phase1.json`

Пара:

```json
{
  "profile_id": "ec2eed50-b55f",
  "chat_id": "290027369",
  "lead_id": "49832125",
  "expected_brand": "foton"
}
```

Allowlist phase1 содержит только `49832125`.

## 3. Dry-run

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_amo_wappi_draft_loop.py \
  --once --dry-run \
  --profiles-file "$HOME/.mango_secrets/draft_loop_profiles_foton_test.json" \
  --pairs-file "$HOME/.mango_secrets/draft_loop_pairs.json" \
  --phase1-config "$HOME/.mango_secrets/amo_wappi_phase1.json"
```

Итог:

```json
{
  "bot_calls": 1,
  "deferred": 0,
  "dry_run": true,
  "processed": 0,
  "retried_pending": 0,
  "skipped": 258,
  "stop_active": false
}
```

AMO POST не выполнялся.

Черновик из `~/.mango_local/draft_loop/journal.jsonl`:

```text
Здравствуйте! Поняла: интересует математика для 7 класса. По подтверждённым данным сейчас нет расписания, формата и стоимости именно по курсу Фотон для 7 класса, поэтому не буду придумывать. Чтобы подобрать вариант точнее, подскажите, пожалуйста, цель занятий: подтянуть школьную программу, подготовиться к контрольным или нужен более сильный уровень?
```

Смысловая заметка: черновик безопасен для draft, но слабый по полезности — бот не нашёл факт про 7 класс и ушёл в уточнение цели.

## 4. Live AMO note

Live-write НЕ запускался, потому что все найденные локальные direct AMO tokens вернули `401 Unauthorized` уже на read-only проверке `GET /api/v4/leads/49832125`.

Проверенные источники токена:

- `prod_runtime_transfer/.env.private`
- `AI Office/artifacts/amo_handoff/.env.private`
- `AI Office/artifacts/amo_handoff/prod_runtime_transfer/.env.private`
- `AI Office/artifacts/amo_handoff/revenue_leakage_os_handoff/.env.private`
- `Revenue leakage OS/amo_handoff/revenue_leakage_os_handoff/.env.private`

Результат у всех одинаковый: `HTTP 401 Unauthorized`.

Дополнительно:

- локальный `~/.mango_secrets/amo_wappi.env` содержит Wappi tokens, но не содержит AMO token;
- `stable_runtime/amocrm_runtime/amo_runtime.db` не содержит активных OAuth connections;
- удалённый AI Office status по локальному ключу вернул `403 Forbidden`.

Для продолжения live-note нужен один из вариантов:

1. свежий `AMOCRM_ACCESS_TOKEN` / `AMO_WAPPI_AMO_ACCESS_TOKEN` для `educent.amocrm.ru`;
2. отдельная доработка `run_amo_wappi_draft_loop.py` под AI Office OAuth/proxy note endpoint, если прямой bearer больше не должен использоваться.

## 5. STOP_DRAFT_LOOP

Проверка стоп-крана выполнена.

Команда с `~/.mango_secrets/STOP_DRAFT_LOOP` вернула:

```json
{
  "bot_calls": 0,
  "deferred": 0,
  "dry_run": true,
  "processed": 0,
  "retried_pending": 0,
  "skipped": 0,
  "stop_active": true
}
```

STOP-файл удалён после проверки.

Выдержки журнала:

```json
{"event":"draft_created","status":"dry_run","profile_id":"ec2eed50-b55f","chat_id":"290027369","message_id":"18217","lead_id":"49832125","brand":"foton","route":"bot_answer_self_for_pilot"}
{"event":"stop_raw_inbound","status":"stop_not_processed","profile_id":"ec2eed50-b55f","chat_id":"290027369","message_id":"18217"}
```

## 6. Hypothesis 0в: AMO outgoing visible in Wappi

Не проверено: Дмитрий ещё не отправлял ответ из интерфейса AMO после успешной AMO note, и live-note не был создан из-за блокера AMO token.

Текущий статус: `OPEN`.

Промежуточный факт: Wappi history показывает `fromMe=true` сообщения с `from_where=api`, но это не доказывает, что ответы из AMO UI видны в Wappi.

## 7. Repo cleanliness

До создания этого отчёта tracked diff был пустой. Runtime state/journal находятся вне репозитория: `~/.mango_local/draft_loop/`.

