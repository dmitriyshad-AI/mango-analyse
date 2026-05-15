# Block E: Customer Timeline Context

Дата: 2026-05-15

## Что сделано

- Зафиксирована роль `customer_timeline` как целевого read-only слоя истории клиента.
- Добавлен `src/mango_mvp/customer_timeline/context_provider.py`.
- Добавлен read-only coverage-аудит `scripts/audit_customer_timeline_coverage.py`.
- Stage 4 deal-aware preview получил явное опциональное включение timeline-контекста.
- Timeline-контекст остается только в preview/report полях и не попадает в AMO payload.
- Добавлена staged-promotion проверка: `timeline_available` -> `timeline_coverage_verified` -> `timeline_preview_enabled` -> `timeline_primary_read_enabled` -> `timeline_live_write_context_allowed`.

## Важное ограничение

Live AMO writeback не зависит от customer timeline и не требует его наличия. Customer timeline можно будет использовать как контекст для live writeback только после coverage-аудита, fallback и отдельного разрешения.
