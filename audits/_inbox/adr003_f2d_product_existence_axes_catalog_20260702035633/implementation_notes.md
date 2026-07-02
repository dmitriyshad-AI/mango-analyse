# ADR-003 F2d Product Existence Axes Catalog

## Что сделано

Добавлен библиотечный proof-layer `product_existence_axes_catalog.py`, который строит производный каталог существования продукта/формата из KB snapshot.

Новый публичный контракт:

- `build_product_existence_axes_catalog(facts)` -> каталог entries/issues;
- `verify_product_format_exists(catalog, brand, grade, subject, format, program_kind/product_family)` -> `exists | not_offered | unknown | needs_slot`.

## Что это НЕ делает

- Не подключается к direct path/provider/profile.
- Не меняет route/text.
- Не включает автономные ответы.
- Не проверяет живые места, бронирование, запись, оплату или статус сделки.
- Не делает вывод `not_offered` из отсутствия факта.

## Проверенные инварианты

- `unknown` является дефолтом, если нет точного fresh client-safe факта.
- `not_offered` возвращается только при явном отрицательном fresh client-safe факте.
- Бренды не смешиваются: отрицательный факт УНПК по химии не превращается в отрицание для Фотона.
- Недостаточные слоты возвращают `needs_slot`, а не произвольный матч.
- Диапазоны классов разворачиваются в полный набор значений.

## Сырой снимок

См. `catalog_summary.json`:

- entries: 299;
- issues: 55;
- exists: 293;
- not_offered: 6.

## Исправления после независимого аудита

Аудитор нашёл 4 смысловых пробоя, они закрыты кодом и регрессионными тестами:

- отменённая смена больше не делает весь лагерь `not_offered`;
- `raw_value=False` и текст "не проводится" больше не становятся положительным `exists`;
- payment/enrollment/manager-action факты не попадают в positive proof;
- шумная строка класса вроде "кабинет 9" возвращает `needs_slot`, а не расширяет запрос до 9 класса.
