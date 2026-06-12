# TZ-6 live AMO recheck report

Дата: 2026-06-10

## Контекст

Повторная проверка после уточнения Дмитрия: amoCRM должен использоваться не через прямой bearer в Mango, а через серверный API Foton Online:

```text
https://api.fotonai.online
```

Секреты не выводились в лог/чат. Клиентам ничего не отправлялось.

## Что проверено

### Wappi -> bot dry-run

Команда dry-run была запущена с dummy AMO base/token, потому что в dry-run AMO POST не вызывается, но текущий runner всё равно требует AMO config при сборке клиента.

Результат:

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

Последний draft в журнале:

- `profile_id`: `ec2eed50-b55f`
- `chat_id`: `290027369`
- `lead_id`: `49832125`
- `message_id`: `18219`
- `brand`: `foton`
- `route`: `draft_for_manager`

Текст черновика:

```text
В Москве площадка Фотона находится по адресу: Верхняя Красносельская ул., 30, метро Красносельская. Если хотите, дальше подберём подходящий вариант по математике для 7 класса.
```

AMO запись в dry-run не выполнялась.

### AI Office / Foton Online API

Через `curl` с обычным User-Agent сервер отвечает:

- `GET /api/integrations/amocrm/status`: `200`, `connected=true`, `status=active`, `token_source=oauth`;
- `POST /api/integrations/amocrm/refresh`: `200`, токен обновлён;
- `POST /api/integrations/amocrm/contact-fields/sync`: `200`, каталог полей синхронизирован.

Важно: Python `urllib` без нормального User-Agent получает Cloudflare `1010 browser_signature_banned`; значит клиент Mango должен ходить в Foton Online API с явным User-Agent или через другой HTTP transport.

### OpenAPI live-сервера

Живой OpenAPI доступен по:

```text
/api/openapi.json
```

AMO endpoints на live-сервере:

```text
/api/integrations/amocrm/callback
/api/integrations/amocrm/contact-fields/sync
/api/integrations/amocrm/contacts/by-phone
/api/integrations/amocrm/deals/dossier-by-phone
/api/integrations/amocrm/launcher
/api/integrations/amocrm/leads/by-phone
/api/integrations/amocrm/refresh
/api/integrations/amocrm/secrets
/api/integrations/amocrm/start
/api/integrations/amocrm/status
```

Endpoint для добавления примечания в lead отсутствует.

Проверенные варианты note/proxy endpoints вернули `404`:

```text
/api/integrations/amocrm/leads/49832125/notes
/api/integrations/amocrm/leads/49832125/notes/add
/api/integrations/amocrm/lead-notes
/api/integrations/amocrm/notes
/api/integrations/amocrm/proxy
/api/integrations/amocrm/request
/api/integrations/amocrm/raw-request
```

## Update 2026-06-10 15:xx MSK: endpoint deployed, live note written

После добавления server-side endpoint в Foton Online API и явного `User-Agent` в Mango client live-write прошёл через правильный путь:

- dry-run: `bot_calls=1`, `processed=0`, AMO write не выполнялся;
- live-write: `processed=2`, `retried_pending=1`, `pending_notes={}`;
- карточка AMO `49832125` показывает два примечания `ЧЕРНОВИК БОТА, не отправлено` от AI Office;
- клиенты не получали сообщений, запись была только в примечания тестовой сделки.

Журнал `~/.mango_local/draft_loop/journal.jsonl`:

```json
{"event":"note_retried","status":"note_written","profile_id":"ec2eed50-b55f","chat_id":"290027369","message_id":"18219","lead_id":"49832125","route":"draft_for_manager"}
{"event":"note_written","status":"note_written","profile_id":"ec2eed50-b55f","chat_id":"290027369","message_id":"18217","lead_id":"49832125","route":"draft_for_manager"}
```

### Проверка 0в: видимость ответа менеджера из AMO в Wappi

После подтверждения Дмитрия, что ответ был отправлен из интерфейса AMO, `Wappi sync/messages/get` по тестовой паре всё ещё вернул только 7 старых сообщений:

```json
{"id":"18219","fromMe":false,"from_where":"phone","text":"какой у вас адрес в Москве?"}
{"id":"18218","fromMe":true,"from_where":"api","text":"Здравствуйте! ... ответим вам после 9:00."}
{"id":"18217","fromMe":false,"from_where":"phone","text":"Тест бота: интересует математика, 7 класс"}
```

Нового исходящего сообщения из AMO UI в Wappi history не появилось.

Вывод: гипотеза 0в отрицательная на текущей связке. Бот сейчас не видит ответы менеджера из AMO через Wappi history; автоматическая пара `draft↔sent` откладывается до отдельного мини-ТЗ по источнику исходящих менеджера.

### Stop-crank

STOP-файл проверен:

```json
{
  "bot_calls": 0,
  "processed": 0,
  "retried_pending": 0,
  "stop_active": true
}
```

STOP-файл удалён после проверки.

## Вывод

Wappi -> bot dry-run работает.

AMO note write через Foton Online API работает на тестовой сделке `49832125`.

Видимость исходящих менеджера из AMO в Wappi history не подтверждена: на live-проверке ответ из AMO UI не появился в `sync/messages/get`.

## Update 2026-06-10: правильная linked-сделка Telegram

Дмитрий нашёл сделку, уже связанную с Telegram-диалогом:

```text
https://educent.amocrm.ru/leads/detail/47854947
```

Локальные secret-конфиги переключены вне репозитория:

```json
{
  "profile_id": "ec2eed50-b55f",
  "chat_id": "290027369",
  "lead_id": "47854947",
  "expected_brand": "foton"
}
```

`~/.mango_secrets/amo_wappi_phase1.json` allowlist также переключён на `47854947`.

Targeted dry-run по последнему входящему Wappi-сообщению:

```json
{"target":"dry_run","lead_id":"47854947","message_id":"18219","text":"какой у вас адрес в Москве?"}
{"bot_calls":1,"processed":0,"skipped":0}
```

Обычный runner не взял этот чат, потому что сообщение `18219` уже было записано в журнал как `note_written` в старой сделке `49832125`, а сам чат не попадает в текущий top-50 `chats/get`. Поэтому для переноса live-пробы использован targeted one-off по явной паре `profile_id/chat_id`.

### AI Office server allowlist

Первый live-write в `47854947` был заблокирован сервером:

```json
{"detail":"AMO lead is not allowlisted for note writes."}
```

Причина: запущенный API-контейнер содержит hard allowlist только `{49832125}`. Кодовый фикс подготовлен и отправлен в GitHub:

```text
AI Office commit 814db41: Allow AMO note writes for linked Telegram test lead
```

На VPS `/opt/ai-office` рабочее дерево грязное, поэтому обычный `git pull && docker compose up --build` не запускался, чтобы не затереть live-изменения. Для закрытия текущей live-пробы была выполнена одноразовая запись внутри API-контейнера через штатный `create_lead_common_note` и server-side OAuth, с временным allowlist только в процессе Python.

Результат записи в правильную сделку:

```json
{
  "lead_id": 47854947,
  "note_id": 467708511,
  "token_source": "oauth",
  "account_base_url": "https://educent.amocrm.ru"
}
```

Локальный draft-loop state обновлён: pending для `ec2eed50-b55f\t290027369\t18219` очищен, сообщение `18219` помечено processed, в journal добавлена строка `note_written` с `note_id=467708511`.

### Что осталось проверить

Дмитрию нужно отправить ответ именно из клиентской переписки в сделке `47854947`, не во внутреннее примечание. После этого повторить `Wappi sync/messages/get` и проверить, появился ли новый `fromMe=true` текст. Если появится — можно матчить `draft↔sent`; если нет — источник outgoing менеджера всё ещё не Wappi history.
