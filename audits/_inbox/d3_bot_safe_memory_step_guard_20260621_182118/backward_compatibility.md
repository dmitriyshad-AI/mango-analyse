# Backward Compatibility

- При выключенном `TELEGRAM_BOT_SAFE_CRM_CONTEXT` новый гейт полностью инертен.
- При `next_step.status=active` текст и маршрут не меняются.
- При отсутствии bot-safe памяти поведение не меняется.
- Подключение сделано только в direct path после LLM-верификатора и до финального output gate.
