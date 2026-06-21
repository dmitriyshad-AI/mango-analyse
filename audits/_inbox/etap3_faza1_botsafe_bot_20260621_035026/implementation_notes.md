# Etap 3 Faza 1 Bot-Safe Context

Дата: 2026-06-21.

Цель: подключить безопасную выжимку клиента из боевой customer timeline памяти к черновику Telegram-бота без передачи полного customer_profile и без автоответа клиенту.

Что сделано:

- Добавлен read-only слой `bot_safe_runtime_context`, который читает только `CustomerTimelineReadApi.bot_context(..., allowed_only=True)`.
- Добавлен флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, по умолчанию выключен.
- Wappi draft-loop теперь при включенном флаге резолвит клиента по AMO lead/contact id и подмешивает только brand-scoped bot-safe context.
- Dynamic simulator теперь умеет брать bot-safe context из боевой памяти по `bot_safe_customer_id`/AMO id для M1-замера.
- Direct path prompt получил отдельный блок "Безопасная выжимка клиента", включаемый только под флагом и только для `active_brand in {foton, unpk}`.
- Добавлена фильтрация по `relevance_tags`: обязательно `bot_safe` + активный бренд канала.
- Добавлен PII/service-id scan до промпта и расширен выходной фильтр служебных идентификаторов `customer:`, `timeline_event:`, `bot_context_chunk:`, `botsafe:`.
- Исправлено read-only открытие SQLite через URI: путь теперь кодируется через `Path.as_uri()` и открывается как immutable read-only snapshot, чтобы M1-бандлы на Яндекс.Диске читали FTS-таблицы без попытки создать служебные файлы.

Что не делалось:

- Не давался `customer_profile` целиком.
- Не читались сырые timeline events для промпта.
- Не включался автоответ клиенту.
- Не менялись live-настройки пилота.
- Не писали в AMO, Tallanto или Wappi.

Боевая память:

- DB: `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Bot-safe chunks: 17,856.
- Все chunks имеют `allowed_for_bot=1`.
- Brand tags в `record_json.relevance_tags`: foton 1,290; unpk 4,017; unknown 12,549.

M1 target set:

- Путь: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/botsafe_crm_context_20260621/target_set.jsonl`.
- SHA256: `1385c2c05fcd838531a27dc62d1f91190128a3012fd02eaf99505da25d60e4e3`.
- Содержит 10 персон + simulator/judge spec: позитивные кейсы памяти, cross-brand NEG, no-customer NEG, P0 NEG.
