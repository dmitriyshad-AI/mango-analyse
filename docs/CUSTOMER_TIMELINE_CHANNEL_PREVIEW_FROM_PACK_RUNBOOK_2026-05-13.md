# Customer Timeline Channel Preview From Pack Runbook

Дата: 2026-05-13

## Зачем нужен слой

Этот слой берет входящее сообщение клиента и проверенный пакет контекста клиента.
На выходе он создает черновик ответа для менеджера.

Он не отправляет сообщение клиенту, не пишет в amoCRM/Tallanto, не меняет продуктовую базу, не трогает runtime-БД и `stable_runtime`.

## Что нужно на входе

- `approved_context_pack.json` - проверенный пакет контекста клиента из предыдущего этапа.
- `inbound_message.json` - нормализованное входящее сообщение клиента.

Минимальный пример входящего сообщения:

```json
{
  "channel": "site_chat",
  "channel_message_id": "msg-1",
  "channel_thread_id": "thread-1",
  "channel_user_id": "visitor-1",
  "direction": "inbound",
  "text": "Сколько стоит подготовка к ЕГЭ?",
  "received_at": "2026-05-13T09:10:00+03:00"
}
```

## Команда

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/customer_timeline_channel_preview_from_pack.py \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --context-pack-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approved_context_pack.json" \
  --inbound-message-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/inbound_message.json" \
  --out-preview-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/channel_preview_from_pack.json"
```

Код возврата:

- `0` - черновик создан.
- `1` - входы прочитаны, но черновик заблокирован: пакет контекста не approved, сообщение не входящее, найден safety-блокер.
- `2` - ошибка путей, схемы JSON или запуска.

## Что внутри результата

- `input_message` - безопасное описание входящего сообщения, без `raw_payload`.
- `draft_preview` - черновик ответа и рекомендуемые действия менеджеру.
- `context_policy` - подтверждение, что это только черновик.
- `source_refs` - хэши входных файлов и идентификаторы.
- `safety` - явные запреты live-отправки и внешних записей.

## Safety gates

- `live_send=false`
- `send_email=false`
- `send_messenger=false`
- `write_crm=false`
- `write_tallanto=false`
- `write_runtime_db=false`
- `stable_runtime_writes=false`
- `network_calls=false`
- `subprocess_calls=false`
- `llm_calls=false`
- `rag_used=false`

## Следующий шаг

Следующий продуктовый слой - рабочее окно менеджера: показать входящее сообщение, проверенный контекст клиента, черновик ответа и кнопки "принять", "изменить", "отклонить".
Даже там реальная отправка должна включаться только отдельным этапом.
