# Обратная совместимость

- Existing CLI остаётся dry-run по умолчанию; `--apply` по-прежнему нужен для записи в локальную `customer_timeline.sqlite`.
- Добавлен необязательный `--identity-export-dir`; если не задан, используется соседний `*_max` / `*_with_contacts`, когда он есть.
- `telegram_message` source_system теперь `telegram_history`, а не общий `channel_snapshot`; dedupe key для новых B1 записей стабильный по `telegram:{dialog}:{message}`.
- Existing channel/email records не ломаются: ключ `message` не добавлен в forbidden scrub, запрещены только Telegram/raw-specific payload keys.
