# Block5 NEG tests: ночной прогон 2026-06-21

Статус: formal_pass по целевым NEG.

## Что проверено

- Telegram/WhatsApp каналовые чанки не становятся bot-safe:
  - `allowed_for_bot=False`
  - `requires_manager_review=True`
  - FTS-путь бот-ретрива не возвращает каналовые чанки как bot-safe.
- Telegram без матча по телефону не создаёт клиента автоматически.
- Семейный/неоднозначный телефон уходит в ambiguous, не приписывается одному клиенту.
- WhatsApp импорт идемпотентен и не создаёт дублей при повторе.
- Tallanto-статус в карточке вынесен в отдельное поле `Статус оплат и занятий` и не попадает в автоисторию/хронологию.
- Превью-карточка не содержит служебные снимки `Read-only AMO contact snapshot`, `exact_phone_single`, `no_exact_phone_match` и маркер `[сжато]`.

## Команды

Timeline targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_import_telegram_export_to_timeline.py \
  tests/test_import_whatsapp_export_to_timeline.py \
  tests/test_customer_timeline_ingestion.py \
  tests/test_customer_timeline_read_api.py
```

Результат: `34 passed in 0.78s`.

Card targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py
```

Результат: `7 passed in 0.42s`.

Full pytest, timeline worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `3365 passed, 5 skipped, 1 warning in 50.18s`.

Full pytest, card worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `3377 passed, 5 skipped, 1 warning in 50.21s`.

## Машинные проверки артефактов

Timeline DB:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/customer_timeline.sqlite`

- `bad_channel_chunks=0`
- `bot_safe_channel_fts_hits=0`
- `telegram_chunks=12000`
- `whatsapp_chunks=40034`

Card preview:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_with_channels/crm_cards_preview.xlsx`

- `Статус оплат и занятий`: 200/200 строк
- старое поле `AI-Tallanto статус по сделке`: 0/200 строк
- Tallanto-текст в `Авто история общения`: 0 строк
- Tallanto-текст в `AI-история по сделке`: 0 строк

Остаточный риск: это formal_pass и машинная проверка; смысловой регрейд карточек остаётся за утренним архитектором.
