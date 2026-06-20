# Этап 3 Фаза 0: bot-safe слой customer_timeline

Дата: 2026-06-21

Ветка: `codex/etap3-botsafe-layer`

Тестовая БД:
`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_email_bridge_seed_full5_20260621/customer_timeline.sqlite`

Audit pack:
`audits/_inbox/etap3_botsafe_20260621_003243/`

## Seed Importer

Seed залит импортёром `canonical_readonly_timeline_import`:

- `source_system`: `canonical_readonly_customer_timeline`
- `source_ref`: `canonical_readonly_20260521_v5`
- почтовая дельта добавлена отдельно через `mail_archive` / `mail_stage2_fresh_relink_bridge_20260621`

Бренд берётся только по клиенту из `customer_opportunities.record_json.product_context.brand`.
У `mail_archive` opportunities brand пустой, поэтому почтовый title не используется для определения бренда.

## Что Сделано

- Добавлен `src/mango_mvp/customer_timeline/bot_safe_summary.py`.
- Добавлен CLI `scripts/build_customer_timeline_bot_safe_summary.py`.
- Для каждого клиента с историей создан ровно один allowed chunk:
  - `chunk_type=bot_safe_summary`
  - `source_ref=botsafe:{customer_id}`
  - `allowed_for_bot=1`
  - `requires_manager_review=0`
- Выжимка строится только из структурных полей:
  - бренд: `customer_opportunities.product_context.brand`
  - стадия: `customer_opportunities.status`
  - интерес: `products_of_interest/title` после фильтров
  - следующий шаг: D8 `resolve_customer_next_step` на лету
- Raw `bot_context_chunks.text/summary` не используются для построения текста.
- Добавлены фильтры title-фрагментов:
  - `redact_text` для email/телефонов;
  - маска 8+ подряд цифр;
  - отсечение чужого бренда;
  - отсечение unsafe interest: скидки, договоры, оплаты, квитанции, счета, документы, пропуски;
  - отсечение имён вложений (`.pdf`, `.jpg/.jpeg`, `.png`, `image-*` и т.п.);
  - очистка `Re/Fwd`.

## Счётчики

- Клиентов с историей: `17 189`
- Создано `bot_safe_summary`: `17 189`
- Покрытие клиентов с историей: `100.0%`
- Бренды по клиентам:
  - `foton`: `402`
  - `unpk`: `562`
  - `unknown`: `16 225`
- D8 next step:
  - `empty`: `17 189`
- Финальная идемпотентность:
  - `created=0`
  - `updated=0`
  - `duplicate=17189`

## NEG

Файл: `audits/_inbox/etap3_botsafe_20260621_003243/validation_report.json`

- `PRAGMA quick_check`: `ok`
- email grep по allowed summaries: `0`
- phone grep по allowed summaries: `0`
- 8+ подряд цифр: `0`
- exact-copy raw `text/summary`: `0`
- unsafe interest markers: `0`
- allowed raw chunks кроме `bot_safe_summary`: `0`
- bot-safe chunks с `requires_manager_review=1` или `allowed_for_bot!=1`: `0`
- bad `source_ref != botsafe:{customer_id}`: `0`
- duplicate `botsafe` source_ref groups: `0`
- брендовые протечки:
  - `foton_contains_unpk=0`
  - `unpk_contains_foton=0`
  - `unknown_contains_brand_marker=0`

## Примеры `bot_context(allowed_only=True)`

1. `customer:00d2cc9635664d249ca7025eff000160`

   `Бренд: Фотон. Стадия: open. Интерес: Обучение на очных курсах в 2025-26 уч.г.; Старт курсов ФОТОН. Следующий шаг: Активный следующий шаг не найден.`

2. `customer:01dea550af144553677a4abf638f732a`

   `Бренд: Фотон. Стадия: open. Интерес: Летняя Выездная школа 2026; Летняя выездная школа с 16 августа по 24 августа; Предложение ЛВШ 26 от Фотон. Следующий шаг: Активный следующий шаг не найден.`

3. `customer:00cb36b66430c12c340f333fbba16efd`

   `Бренд: УНПК. Стадия: Закрыто и не реализовано. Интерес: ЛВШ-26 август физмат. Следующий шаг: Активный следующий шаг не найден.`

4. `customer:00f6a72ba6bb11731f628b5a7010420e`

   `Бренд: УНПК. Стадия: open. Интерес: Очная летняя школа с 28 июля по 8 августа; Летняя очная школа УНПК МФТИ (6 – 17 июля), ФМ. Следующий шаг: Активный следующий шаг не найден.`

5. `customer:0000ca58a9d601a0711914ea80545dda`

   `Стадия: не определена. Интерес: не определён. Следующий шаг: Активный следующий шаг не найден.`

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_bot_safe_summary.py`
  - `6 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3466 passed, 5 skipped, 1 warning`

## Остаточный Риск

- `next_step` сейчас везде `empty`, потому что в текущем seed D8 не видит явных структурных next-step в событиях. Это безопасно, но малоинформативно.
- Большинство клиентов остаётся с `brand=unknown`, потому что fresh mail bridge не несёт бренд, а AMO brand есть не у всех.
