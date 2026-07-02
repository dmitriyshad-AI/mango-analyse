> DONE 2026-07-02 04:04 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 04:01 | ветка codex/adr003-semanticframe-migration | codex

> DONE 2026-07-02 03:59 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 03:49 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/knowledge_base/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_product_existence_axes_catalog.py
Семантический-аудит: да

# TZ ADR-003 F2d: product existence axes catalog

## Контекст

F2c подтвердил: быстрый route-only active не готов. Реальный рычаг автономности лежит в проверке факта "курс/формат существует", отдельно от live availability / записи / мест.

Текущая telemetry (`bot_confirmed_facts`, `missing_facts`, `freshness`) полезна для отчёта, но не является строгим proof-контрактом.

## Цель

Добавить детерминированный модуль `product_existence_axes_catalog`, который строит из KB snapshot производный каталог существования продуктов/форматов с явными осями и функцией:

`verify_product_format_exists(catalog, brand, grade, subject, format, program_kind/product_family) -> exists | not_offered | unknown | needs_slot`

## Scope

- Новый модуль `src/mango_mvp/knowledge_base/product_existence_axes_catalog.py`.
- Тесты `tests/test_kb_product_existence_axes_catalog.py`.
- Никакой runtime-проводки в direct path/provider/profile.
- Никаких новых флагов.

## Инварианты

- `unknown` по умолчанию. Отсутствие факта не значит "курса нет".
- `exists` только при свежем client-safe факте и совпадении заявленных осей.
- `not_offered` только при явном свежем client-safe отрицательном факте.
- Live availability / свободные места / запись / бронирование не считаются доказательством автономного ответа.
- Бренды не смешивать.
- Не трогать P0 floor/preblock, профиль, live/Wappi/AMO/Tallanto/CRM.

## Acceptance

- Тесты доказывают:
  - UNPK online олимпиадная физика 9 класс -> `exists` по snapshot.
  - UNPK летняя/городская школа для 5 класса -> `exists`.
  - UNPK химия -> `not_offered` по явному отрицательному факту.
  - Wrong brand не матчится.
  - Недостаточные слоты -> `needs_slot`.
  - Нет факта -> `unknown`, а не `not_offered`.
- `pytest` по новому тесту зелёный.
- Audit pack объясняет, что это библиотечный proof-layer, не включение поведения.
