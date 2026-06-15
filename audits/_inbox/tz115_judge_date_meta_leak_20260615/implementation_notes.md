# TZ-115 implementation notes

## Часть 2: meta leak guard

- Добавлен `_META_FACT_PHRASE_RE` отдельно от `_META_CLIENT_MARKERS`.
- Окно осталось узким: `{0,20}` между «не указано/не уточнено/отсутствует» и якорем «факт/база/данные».
- `has_meta_leak()` теперь возвращает `True` для служебных фраз:
  - «в фактах нет»
  - «нет в данных»
  - «не указана в фактах»
- `meta_markers_present()` добавляет `fact_phrase_leak`.

## Часть 1: judge date grounding

- Добавлен `_fact_window_date_keys(fact)`.
- Источник даты строго `before_YYYY_MM_DD` в `fact_key` / `fact_id`.
- `valid_until` сверяется с датой из ключа, но не индексируется сам по себе.
- В `snapshot_number_index()` добавлена узкая строка добавления window-date keys.
- Regex текста бота не менялся.
