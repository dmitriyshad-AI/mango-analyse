> DONE 2026-07-31 18:47 | ветка codex/regex-to-understanding-map-20260731 | codex

> TAKE 2026-07-31 18:08 | ветка codex/regex-to-understanding-map-20260731 | codex

Ветка: codex/regex-to-understanding-map-20260731
Зоны: tests/test_adr003_regex_understanding_moratorium.py, tests/fixtures/adr003_direct_path_text_patterns_snapshot.json, docs/adr003_understanding_map.yaml, artifacts/, audits/_inbox/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ТЗ №200 D1 v2: ведро 2 и ведро 3 по канону 832

Полное исходное ТЗ:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-31_TZ_200_regex_to_ponimanie_karta.md`.

Решение владельца и архитектора после STOP:

- канон карты — 832 фактические строки
  `adr003_direct_path_text_patterns_snapshot.json@main-ca1c9ce5`;
- каждой строке добавить детерминированный `row_id`, `path`, `lineno`,
  `col_offset`, `node_kind`, `symbol` и существующий content hash;
- в карту включить `regex_call`, `marker_helper_call`, `string_contains`
  и `text_table`;
- `255` — только потолок `CHANNEL_MARKER_HELPER_BUDGET@main-ca1c9ce5`,
  не знаменатель;
- таблицы считать машинно по строкам канонического снапшота, без цели
  `75/1507`;
- каждое число в отчёте сопровождается именем/хешем снимка;
- Д1 размечает только ведро 2 и ведро 3; ведро 1/4 не присваивает;
- на шагах 1-3 `src/**` не менять.

## Приёмка

- генератор и фикстура дают ровно 832 уникальных `row_id`;
- все 832 строки имеют `lineno` и `col_offset`;
- YAML содержит только строки ведра 2/3 и не имеет неизвестных `row_id`;
- ведро 2 содержит не менее 110 записей, спорные деньги/P0 подтверждены
  воспроизведением;
- строки ведра 3 являются только форматом/гигиеной;
- бюджеты моратория не выросли, `src/**` не изменён;
- ломатель и ревьюер дали независимые файлы.

## СТОП

- число строк после обогащения не равно 832;
- `row_id` не уникален или дрожит между двумя генерациями;
- любая правка `src/**` или защитного поведения;
- пересечение с файлами параллельного исполнителя;
- ведро 2 меньше 110 после полного аудита;
- рост любого бюджета моратория.
