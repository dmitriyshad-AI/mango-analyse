> DONE 2026-06-20 23:46 | ветка codex/tz-email-timeline-bridge | codex

> TAKE 2026-06-20 23:04 | ветка codex/tz-email-timeline-bridge | codex

Ветка: codex/tz-email-timeline-bridge
Зоны: src/mango_mvp/customer_timeline/, scripts/, tests/, tasks/, docs/worktrees_registry.md, audits/_inbox/, product_data/customer_timeline/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_store.py tests/test_customer_timeline_import_cli.py
Семантический-аудит: да

# ТЗ D4 — мост письма+звонки в тестовую customer_timeline

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_D4_most_pisma_v_pamyat.md`, v2.

Уточнение Дмитрия: авторитетную привязку брать из свежего `fresh-only` relink `bacdd96f` по `message_sha256`; `union` relink не использовать, потому что он даёт ложные дубликаты из-за старых/новых ID на одних людей.

## Цель

Загрузить сводки писем (корпус + дельта) и звонки в отдельную тестовую `customer_timeline` SQLite-БД. Источники только read-only. Боевую timeline-БД, AMO, Tallanto, CRM, YAML и `stable_runtime` не менять.

## Требования

1. JOIN email-события с fresh-only relink по `message_sha256`; inline `customer_id` из jsonl считать интеримным и не использовать.
2. Починить `MailMessageNormalizer`: если передан resolved `customer_id`, нормализатор обязан использовать его; если resolved id нет, не чеканить synthetic customer id.
3. Matched email-события писать в клиентскую timeline; unmatched писать в `pending_attribution`, не создавая клиента.
4. Построить реальный загрузочный гейт: email/channel events и bot chunks не могут быть `allowed_for_bot=True`; нарушение отклоняется.
5. Звонки грузить существующим read-only путём.
6. Запись делать через `bulk_write`, идемпотентно. Повторный запуск должен дать 0 новых записей.
7. Flock: либо реализовать, либо явно зафиксировать в отчёте, что механизм записи ограничен `WAL+busy_timeout`, без отдельного `flock`.

## Приёмка

- matched events лежат под customer_id из fresh-only relink;
- unmatched идут в `pending_attribution`;
- повторная загрузка не плодит дубли;
- событие письма с `allowed_for_bot=True` отклоняется гейтом;
- inline `customer_id` из jsonl не используется;
- тестовая БД лежит в `product_data/customer_timeline/`, git её игнорирует;
- `read_api` отдаёт ленты 3-5 клиентов по дате;
- отчёт в `tasks/_done/` и audit pack в `audits/_inbox/`.
