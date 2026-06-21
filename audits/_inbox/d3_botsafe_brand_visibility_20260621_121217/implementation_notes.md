# D3 Bot-Safe Brand Visibility

Дата: 2026-06-21.

Задача: привести видимость bot-safe памяти к решению Дмитрия от 21.06: бот видит выжимки активного бренда канала и выжимки `unknown`; выжимки точно чужого бренда не подмешиваются.

Что сделано:

- В `bot_safe_runtime_context.py` фильтр chunks изменен с `active_brand only` на `active_brand OR unknown`, с явным исключением чужого известного бренда.
- В direct path добавлен тот же helper видимости, чтобы prompt не подмешивал чужой бренд даже если верхний контекст уже содержит mixed items.
- В отчет Фазы 0 добавлен `brand_source_counts`.
- Добавлен тест, что event-brand используется, когда deal-brand неизвестен.
- Добавлены тесты, что unknown chunk виден для активного бренда, а explicit foreign brand скрыт.

Что не делалось:

- Production DB не запускалась с `--apply` из этого worktree.
- AMO/Tallanto/Wappi/live не трогались.
- `allowed_for_bot=0` не менялся.
- PII scan сохранен.

Важная сверка по текущему коду:

- В актуальной ветке Фаза 0 уже умеет строить brand-scoped chunks по `customer_opportunities.product_context.brand` и `timeline_events.metadata/record.brand`.
- Поэтому покрытие known brand уже находится на уровне примерно 5.3k клиентов; новая правка в основном меняет runtime-видимость `unknown` chunks.
