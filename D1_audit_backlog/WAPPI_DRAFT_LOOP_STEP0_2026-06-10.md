# Wappi draft loop Step 0 discovery

Дата: 2026-06-10
Режим: read-only. Сообщения клиентам не отправлялись, AMO note не создавались.

## Вывод

Endpoint для чтения истории Telegram-чата подтверждён living API:

`GET https://wappi.pro/tapi/sync/messages/get`

Параметры, пригодные для пилота:

- `profile_id`
- `chat_id`
- `limit`
- `offset`
- `order=desc`
- `mark_all=false`

`mark_all=true` в публичной Postman-коллекции является побочным эффектом, поэтому в allowlist поллера должен проходить только безопасный вариант без отметки прочитанным.

Список активных чатов подтверждён living API:

`GET https://wappi.pro/tapi/sync/chats/get`

Параметры, использованные read-only: `profile_id`, `limit`, `offset`, `order=desc`, `show_all=false`.

## Реальный JSON shape

Значения ПДн отредактированы; структура и имена полей сохранены.

```json
{
  "status": "done",
  "timestamp": 1781044758,
  "time": "2026-06-10T01:39:18+03:00",
  "messages": [
    {
      "id": "18216",
      "type": "text",
      "from": "<chat_id>",
      "to": "<profile_account_id>",
      "fromMe": false,
      "senderName": "<contact_name>",
      "contact_name": "<contact_name>",
      "username": "<contact_username>",
      "time": 1781018800,
      "stanzaId": "",
      "chatId": "<chat_id>",
      "isForwarded": false,
      "isReply": false,
      "file_name": "",
      "isRead": false,
      "delivery_status": "delivered",
      "s3Info": {},
      "poll_votes": null,
      "mimetype": "",
      "from_where": "phone",
      "isEdited": false,
      "isGif": false,
      "isFromAPI": true,
      "isDeleted": false,
      "location": null,
      "isPinned": false,
      "reactions": null,
      "wappi_bot_id": "",
      "task_id": "",
      "template_id": "",
      "crm_entities": {
        "crm_type": "",
        "crm_id": "",
        "message_id": "",
        "chat_id": "",
        "manager_id": ""
      },
      "is_blacklist": false,
      "billable": false,
      "attaches": null
    }
  ],
  "task_id": "<request_task_id>",
  "uuid": "<profile_id>",
  "gifts": null
}
```

## Критерии ТЗ

| Критерий | Статус | Комментарий |
|---|---:|---|
| stable `message_id` | PASS | Поле `id` стабильно в истории и подходит для дедупа. |
| направление | PASS | Поле `fromMe` есть в polling history. |
| `chatId`, текст, тип, время, имя контакта | PASS | `chatId`, `body`, `type`, `time`, `senderName`/`contact_name` есть в history. |
| только приватные чаты | PASS для списка чатов | `/tapi/sync/chats/get` отдаёт `type=user`; в `messages/get` тип чата не приходит, фильтр должен опираться на список чатов. |
| "новые с момента X" | FALLBACK | Параметр `date` на живой выборке вернул нестабильный результат с дублями. Для v1 использовать последние K сообщений + diff по `(profile_id, chat_id, message_id)`. |
| исходящие менеджера из AMO видны в Wappi | НЕ ЗАКРЫТО | В истории найдены исходящие `fromMe=true` с `from_where=api` и `from_where=phone`, но ручная проверка именно отправки из интерфейса AMO по тестовой сделке 49832125 ещё не выполнена. До проверки unedited-rate считать экспериментальным. |

## Проверка исходящих

Read-only scan последних чатов Telegram-профилей показал:

- Фотон: есть исходящие `fromMe=true` с `from_where=api` и `from_where=phone`.
- УНПК: есть исходящие `fromMe=true` с `from_where=api` и `from_where=phone`.
- `crm_entities` в найденных исходящих пустой, поэтому по истории нельзя доказать, что эти сообщения отправлены именно из AMO-интерфейса.

Решение для реализации до ручной проверки:

- история для бота может использовать Wappi исходящие как `Ответ: ...`, но в отчёте и note нужно явно фиксировать, что видимость AMO-исходящих требует live-проверки;
- классификация `unedited_rate` должна быть включаемой только после подтверждения AMO visibility или должна оставаться ручной метрикой;
- если Дмитрий подтвердит, что AMO-исходящие не видны в Wappi, история в пилоте должна содержать только клиентские реплики, а каждое AMO note должно включать пометку "бот не видит ответы менеджера".

## Конфиги

Подтверждено без вывода секретов:

- `~/.mango_secrets/amo_wappi.env` есть, права `600`, Wappi token keys есть.
- `~/.mango_secrets/amo_wappi_profiles.json` есть, права `600`, содержит 4 профиля: Telegram/Max для Фотона и УНПК.
- `~/.mango_secrets/amo_wappi_phase1.json` отсутствует.
- `~/.mango_secrets/draft_loop_pairs.json` отсутствует.

Без `draft_loop_pairs.json` live-запись note невозможна по ТЗ: авто-резолв сделки имеет право писать только кандидат в журнал, но не note.

## Нужно от Дмитрия для live-проверки

1. Создать явную пару в `~/.mango_secrets/draft_loop_pairs.json`: `(profile_id, chat_id) -> {lead_id: 49832125, expected_brand}`.
2. Подтвердить AMO allowlist/config для тестовой сделки 49832125.
3. Отправить одно тестовое сообщение из интерфейса AMO в тестовый чат и дать команду на read-only сверку Wappi history.

