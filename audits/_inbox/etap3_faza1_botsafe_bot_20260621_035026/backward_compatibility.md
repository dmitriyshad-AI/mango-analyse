# Backward Compatibility

Default behavior:

- `TELEGRAM_BOT_SAFE_CRM_CONTEXT` по умолчанию выключен.
- При выключенном флаге direct prompt не содержит блок "Безопасная выжимка клиента".
- При выключенном флаге Wappi draft-loop продолжает строить контекст без read-only customer timeline памяти.
- Dynamic simulator без флага работает как раньше.

Runtime compatibility:

- Если DB отсутствует, customer не найден, identity ambiguous или brand unsupported, новый слой возвращает пустой context и не ломает черновик.
- `customer_profile` и сырые timeline events не используются, поэтому текущие manager-only контракты customer timeline не расширяются.
- Read-only SQLite URI теперь безопасно открывается из путей с пробелами и из M1-бандлов на Яндекс.Диске как immutable snapshot; write-path не менялся.

Rollback:

- Снять `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`.
- Удалять DB или менять pilot profile не требуется.

Known non-changes:

- Не включается автоответ.
- Не меняются P0/brand/output gates.
- Не меняется политика автономии бота.
