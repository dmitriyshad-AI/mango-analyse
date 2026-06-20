# Риски

- Клиентский риск: Telegram text сохраняется как manager-only context (`allowed_for_bot=false`), поэтому бот не получает переписку как bot-safe факт.
- Данные/записи: источники и canonical DB использовались read-only/dry-run; live AMO/Tallanto/CRM не трогались, сообщений не отправлялось.
- Identity risk: username matching работает только по уже существующим `telegram_username` identity links; direct Tallanto telegram id не используется.
- Family/shared phone risk: ambiguous phone не мержится в первого клиента, но real-data dry-run на текущем sidecar не дал ambiguous кандидатов; нужен регрейд Клода по сырью семейных телефонов.
- Откат: один коммит B1; runtime/live state не менялся.
