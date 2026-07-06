# Обратная совместимость

- Форматы `SubscriptionDraftResult` и telemetry не менялись.
- Новых флагов нет.
- Для child-safety route только ужесточается до `manager_only`; автономность не расширяется.
- Existing complaint P0 получает чуть более эмпатичный безопасный текст, без изменения route/safety flags.
