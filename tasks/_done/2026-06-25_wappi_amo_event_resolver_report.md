# Wappi -> AMO authoritative resolver via AMO chat events

Дата: 2026-06-25
Ветка: `codex/wappi-amo-event-resolver`
Base: `abb6799` (`codex/wappi-context-window`)

## Задача

Найти и реализовать надёжный способ понять по входящему Wappi-сообщению, к какой AMO-сделке и контакту оно относится.

## Вывод

Прямой Wappi payload (`crm_entities`, `item_link`) в проверенных чатах пустой. Надёжная связь есть в AMO events:

- событие `incoming_chat_message`;
- `entity_type=lead`;
- `entity_id=<amo lead id>`;
- `_embedded.entity.linked_talk_contact_id=<amo contact id>`;
- `value_after[0].message.origin=<wappi origin>`;
- `value_after[0].message.talk_id=<amo talk id>`.

После независимого аудита матчинг усилен: одного `timestamp + origin` недостаточно. Resolver теперь принимает auto-pair только если AMO `talk_id` подтверждён последовательностью сообщений этого же Wappi-чата: минимум 2 совпадения `direction + timestamp` в окне ±15 секунд.

## Проверка на кейсе со скриншота

Wappi:

- profile: `18b255b8-7a67`
- chat suffix: `4977`
- message_id: `15623`
- timestamp: `1782404856`
- channel: Telegram

AMO event match:

- unique match count: `1`
- `lead_id=50101349`
- `contact_id=77345755`
- `talk_id=3040`
- `origin=pro.wappi.tg`
- `amo_event_dt_sec=4`
- `amo_sequence_match_count=2`

Новый resolver smoke:

```text
status=matched
lead_id=50101349
contact_id=77345755
match_key=amo_chat_event
match_value=talk:3040
amo_talk_id=3040
amo_event_dt_sec=4
amo_sequence_match_count=2
```

## Проверка на выборке из live watch journal

Источник: `pair_missing` из `live_watch_context_20260625_201021/journal.jsonl`.

Сэмпл: 32 свежих входящих из Wappi.

Результат:

- unique: `17`
- no_match: `12`
- ambiguous: `3`

Правило безопасности: при `ambiguous` resolver не делает fallback и не угадывает сделку.

После усиления sequence-confirmation:

Артефакт: `tasks/_done/2026-06-25_wappi_amo_event_resolver_sequence_check.json`

Сэмпл: 25 свежих входящих из Wappi, с подтягиванием Wappi-истории по каждому чату.

Результат:

- matched: `6`
- sequence_unconfirmed: `3`
- rate_limited: `5`
- brand_mismatch: `2`
- closed_lead: `3`
- max_phone_missing: `6`

AMO writes: `0`; client sends: `0`.

## Что изменено

- `scripts/run_amo_wappi_draft_loop.py`
  - `AmoAutoResolver` сначала пытается найти AMO `incoming_chat_message` событие по `timestamp + origin`.
  - Для принятия связи требует минимум 2 совпадения между Wappi history и AMO events по `talk_id`, направлению сообщения и времени.
  - Для Telegram origin: `pro.wappi.tg`.
  - Для Max origin: `pro.wappi.3`.
  - Проверяет открытую сделку, привязку contact->lead и brand mismatch.
  - Если AMO `lead?with=contacts` не вернул contacts, resolver отказывается.
  - Старый поиск по Telegram ID / Max phone оставлен как fallback только если AMO event вообще не найден.
  - Если AMO event неоднозначен или sequence не подтверждён, resolver возвращает `rejected`, fallback не используется.
  - Добавлен лимит AMO event lookup: `30` запросов на `60` секунд, чтобы не выбивать read-only rate limit.

- `src/mango_mvp/integrations/amo_wappi_transport.py`
  - Разрешён только read-only `GET /api/v4/events`.
  - AMO write через этот транспорт по-прежнему запрещён.

- `tests/test_run_amo_wappi_draft_loop.py`
  - Добавлены регрессии на unique event + sequence match, ambiguous refusal, single-event unconfirmed refusal, contact-not-linked refusal, missing-contact readback refusal, event-unavailable refusal, fallback when no event, и GET-only `/events`.

## Проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_run_amo_wappi_draft_loop.py tests/test_draft_loop.py
# 50 passed in 1.02s
```

Read-only live smoke по AMO events:

- AMO writes: `0`
- client sends: `0`
- Tallanto/CRM writes: `0`

## Остаточные ограничения

- Если по timestamp/origin найдено 0 событий, используется старый fallback.
- Если найдено 2+ подтверждённых событий, resolver отказывается и не пишет заметку.
- Если найден только одиночный event без подтверждения последовательности Wappi-чата, resolver отказывается.
- На первом массовом запуске resolver может упираться в AMO read-only rate limit; добавлен локальный лимит, но включать в watch лучше сначала в dry-run.
- Для ещё более сильного reverse lookup по `talk_id` нужен whitelist на AMO Talks API; текущий read-only connector его не пропускает.
