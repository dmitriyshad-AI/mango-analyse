# Backward compatibility

## OFF-поведение

Если `TELEGRAM_BOT_SAFE_CRM_CONTEXT` выключен, prompt block bot-safe памяти не добавляется.

## Старые данные

Старые items без `next_step_status` продолжают проходить как раньше: поле статуса просто отсутствует.

## Контракт данных

Добавлено только новое безопасное поле `next_step_status`. Существующие поля не удалялись и не переименовывались.

## Runtime

`customer_profile` не передаётся в бот. Используется только `bot_context(allowed_only=True)`.
