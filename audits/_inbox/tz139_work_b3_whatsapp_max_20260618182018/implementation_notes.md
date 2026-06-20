# TZ139 Work B3 WhatsApp/MAX

Дата: 2026-06-18
Ветка: `codex/tz139-customer-timeline`
HEAD до стадии: `fc4edef`

## Что сделано

- Подтвержден источник WhatsApp: сырой архив `all_whatsapp_chats.txt` найден в канонической папке, git-ignored; `product_data/transcripts/whatsapp_chats.sqlite` существует, но без таблиц, как источник для B3 не использовался.
- Подтверждено отсутствие MAX-архива переписок в проверенных `product_data` путях. Для MAX новых данных и импортера не добавлено.
- Добавлены контрактные типы `whatsapp_message`, `whatsapp_user_id`, `whatsapp_phone`.
- `ChannelMessageNormalizer` теперь мапит WhatsApp/Wappi в `whatsapp_message` и `whatsapp_user_id`, а MAX оставлен на существующем `max_message`.
- `whatsapp_phone` нормализуется как телефон.
- Существующий WhatsApp txt-импортер доработан:
  - уникальный phone match линкуется к существующему customer_id;
  - неоднозначный phone match создает отдельную ambiguous identity и конфликт `whatsapp_phone_ambiguous`;
  - для ambiguous phone не создается обычный `phone`-линк, чтобы общий resolver не схлопнул семью;
  - для уникальных/новых phone-chat сохраняются `phone` и `whatsapp_phone` links;
  - все bot-context chunks остаются `allowed_for_bot=False`.
- Расширен raw scrub для WhatsApp/Wappi payload-ключей.

## Read-only подтверждение

- Не было live-записей в AMO/CRM/Tallanto.
- Не было отправки сообщений.
- Не запускались ASR/RA/analyze.
- Реальный WhatsApp прогон выполнен только в `dry_run_preview`; probe SQLite не был создан.
- Реальная timeline DB читалась через read-only lookup; `apply=False`.

## MAX статус

B3 по MAX заблокирован: архив переписок не найден. Есть только контрактный `max_message`/`max_user_id` слой и упоминания MAX в KB/профильных данных, но не message archive.
