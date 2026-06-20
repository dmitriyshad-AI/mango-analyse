# Риски

- Запись: live `0`; внешнего rollback не требуется.
- Среда: AMO read-only контекст недоступен, поэтому реальный diff/snapshot по живым AMO сущностям не получен.
- Политика multi-deal: текущий D7 guard блокирует `open_deal_count>1`; если архитектор хочет писать в каждую открытую сделку одного бренда, нужен отдельный policy change.
- Источник бренда: read_api не отдаёт `product_context.brand`, поэтому dry-run читает это из `customer_opportunities.record_json` в SQLite read-only.
- LLM: default `rule`, чтобы не делать неявные LLM-вызовы; при необходимости можно запускать `--history-summary-provider codex_cli`.
