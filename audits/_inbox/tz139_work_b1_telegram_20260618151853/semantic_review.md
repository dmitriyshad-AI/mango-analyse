# Semantic review

- Бренды: импорт запускается с одним `brand` за прогон; в событиях/chunks ставится `brand:<brand>`, смешивания Фотон/УНПК внутри одного чанка не добавлено.
- Клиентский текст: Telegram message text попадает только в timeline event preview/summary и `bot_context_chunks.allowed_for_bot=false`; это manager-only слой.
- P0/сырьё: raw Telegram update/message/callback structures scrubbed before SQLite; тест проверяет физический dump.
- Коммерческие факты/цены/условия: B1 не генерирует клиентские ответы и не извлекает цены; semantic gate ограничен хранением/разметкой переписки.
