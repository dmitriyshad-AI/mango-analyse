# Backward Compatibility

- Старые записи TelegramPilotSQLiteStore не требуют миграции.
- Новые поля сохраняются только в JSON `context` и `draft_metadata`.
- Если `funnel_state` отсутствует, prompt/metrics/guards работают по старой логике.
- API `SubscriptionDraftResult` не менялся.
- Существующие tests по KB, prompt, LLM guards, runtime, store и metrics проходят.
- Runtime env vars не менялись.
