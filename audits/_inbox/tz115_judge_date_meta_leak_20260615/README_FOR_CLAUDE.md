# TZ-115 review scope

Проверь по сырью:

1. `_META_FACT_PHRASE_RE` не добавлен в `_META_CLIENT_MARKERS`.
2. `has_meta_leak()` ловит «в фактах нет» / «нет в данных» / «не указана в фактах».
3. NEG-фразы из ТЗ не блокируются.
4. `_fact_window_date_keys(fact)` парсит только `before_YYYY_MM_DD` из `fact_key`/`fact_id`.
5. `valid_until` сам по себе не индексируется.
6. `snapshot_number_index()` добавляет window-date keys узко и не трогает regex текста бота.
7. Полный pytest зелёный.
