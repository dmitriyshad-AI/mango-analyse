# Customer Timeline Integration Decision 2026-05-15

## Решение

`customer_timeline` остается в проекте и становится целевым read-only слоем истории клиента.

Он нужен не как временная демонстрация, а как единый слой чтения истории по клиенту: звонки, AMO, Tallanto, почта, Telegram и будущие каналы. В текущем этапе он подключается аккуратно: через общий context provider, отчет покрытия, явные флаги и fallback.

## Жесткие границы

- `customer_timeline` не является live-write базой.
- Он не пишет в AMO, Tallanto, CRM и мессенджеры.
- Он не запускает ASR и R+A.
- Он не меняет runtime DB и `stable_runtime`.
- Старые deal-aware CSV/snapshot пути остаются рабочими.
- Live AMO writeback не зависит от customer timeline как обязательного источника.

## Порядок включения

1. `timeline_available` - база существует, read API работает, safety contract зеленый.
2. `timeline_coverage_verified` - отчет покрытия подтвердил, что deal-aware телефоны есть в timeline.
3. `timeline_preview_enabled` - preview показывает timeline-контекст и предупреждения.
4. `timeline_primary_read_enabled` - основные чтения истории идут через context provider.
5. `timeline_live_write_context_allowed` - timeline разрешен как один из источников для live AMO writeback, но только с fallback и quality gate.

Перепрыгивать сразу к live-контексту нельзя.

## Что сделано в блоке E

- Добавлен read-only adapter `src/mango_mvp/customer_timeline/context_provider.py`.
- Добавлен coverage-аудит `scripts/audit_customer_timeline_coverage.py`.
- Stage 4 deal-aware preview может добавлять timeline-контекст только по явному флагу.
- Timeline-контекст не добавляется в `DEAL_AI_FIELDS` и не попадает в AMO payload.
- Добавлены тесты на fallback, safety, coverage и staged promotion.

## Что остается дальше

- Прогнать coverage-аудит на реальном deal-aware наборе без записи во внешние системы.
- По результатам принять решение, достаточно ли покрытие для `timeline_primary_read_enabled`.
- После этого отдельно подключать context provider к рабочему месту менеджера и будущему боту.
