# Backward Compatibility

- Флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT` остается default OFF.
- Без флага prompt не меняется.
- `allowed_for_bot`, `requires_manager_review`, PII scan и service-id sanitizer не менялись.
- Production DB не мигрировалась в этом блоке.
- Поведение меняется только при включенном bot-safe context: раньше показывался только exact active brand, теперь exact active brand + unknown.

Rollback:

- Снять `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`.
- Или вернуть helper видимости на exact active brand only.
