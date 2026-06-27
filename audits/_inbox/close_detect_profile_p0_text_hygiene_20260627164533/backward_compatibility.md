# Обратная совместимость

- Форматы: формат `SubscriptionDraftResult` не менялся; новый слой использует существующие поля `draft_text`, `safety_flags`, `manager_checklist`, `metadata`.
- Потребители: `TELEGRAM_TONE_CLOSE_DETECT` теперь входит в pilot profile; явные context/env override сохраняют приоритет и могут выключить флаг через `0`.
- `TELEGRAM_DIRECT_P0_TEXT_HYGIENE` экспортирован через `subscription_llm_parts.__init__`, но не включён в профиль и не меняет default-поведение.
- Legacy P0, бренд-разделение и числовая верификация не изменялись.
