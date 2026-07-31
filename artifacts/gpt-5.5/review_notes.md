# Review notes GPT-5.5 для ТЗ №200

Дата: 2026-07-31
Режим: read-only по `src/**`; проверены `tests/test_adr003_regex_understanding_moratorium.py`, `tests/fixtures/adr003_runtime_channel_regex_snapshot.json`, `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json`, полное ТЗ `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-31_TZ_200_regex_to_ponimanie_karta.md`.

## Итоговый вердикт

ТЗ в текущем виде нельзя исполнять как однозначную карту `190 + 233 + 255 + 75`: эти числа относятся к разным измерителям и разным наборам файлов. Главная правка ТЗ: сначала зафиксировать машинную единицу записи и стабильный `row_id`/`file:line`, затем уже размечать `bucket`. Без этого параллельная разметка Д1/Клода будет конфликтовать и часть записей невозможно однозначно привязать.

## Точная семантика контрольных чисел

1. `snapshot 197`
   - Факт: `tests/fixtures/adr003_runtime_channel_regex_snapshot.json` содержит 197 строк.
   - Генератор: `_regex_snapshot()` в `tests/test_adr003_regex_understanding_moratorium.py:308-350`.
   - Что считает: все `re.compile(...)` под `src/mango_mvp/channels/**/*.py`.
   - Это фактический снимок, не бюджет.
   - В строках нет `lineno`, `col_offset`, `row_id` и `node_kind`.

2. `live regex 190`
   - Факт: 197 минус 7 строк `src/mango_mvp/channels/telegram_pilot_reporting.py` = 190.
   - Это производная ручная метрика "боевое замыкание без отчётности", а не отдельное поле фикстуры.
   - Важно: `CHANNEL_REGEX_BUDGET` в тесте имеет сумму 200, а не 197; без `telegram_pilot_reporting.py` сумма бюджетов 193, а не 190. Расхождение из-за потолков: `direct_path.py` бюджет 11 при факте 10, `post_layers.py` бюджет 72 при факте 70.

3. `inline 233`
   - Факт сходится только при таком определении: все inline-вызовы `re.search/match/fullmatch/findall/finditer/split/sub/subn` в 15 файлах из regex-снимка, если исключить `telegram_pilot_reporting.py`.
   - Разбивка: `output_verification_floor.py` 61, `post_layers.py` 40, `direct_path.py` 34, `policy_routing.py` 32, `support.py` 17, `fact_claim_audit.py` 15, `few_shot_reference.py` 11, `dialogue_memory.py` 7, `p0_recall_spec.py` 6, `text_hygiene.py` 4, `contracts.py` 3, `reliable_answerer.py` 3.
   - Текущий генератор `adr003_direct_path_text_patterns_snapshot.json` считает другое: 215 inline-вызовов в своём списке `DIRECT_PATH_PATTERN_FILES`, потому что исключает часть файлов с regex-снимком и включает другие direct-path файлы.

4. `marker 255`
   - Это не факт количества записей. Это сумма `CHANNEL_MARKER_HELPER_BUDGET` в `tests/test_adr003_regex_understanding_moratorium.py:84-95`.
   - Фактический счётчик теста `_channel_marker_helper_call_counts()` сейчас даёт 172 вызова по всем `src/mango_mvp/channels/**/*.py`.
   - Фикстура `adr003_direct_path_text_patterns_snapshot.json` содержит только 130 `marker_helper_call`, потому что `_direct_path_text_pattern_snapshot()` добавляет запись только для вызовов с литеральными marker-аргументами. Нелитеральные вызовы из бюджета туда не попадают.
   - Поэтому формулировка "255 маркерных вызовов" в приёмке смешивает budget ceiling с фактическим количеством.

5. `tables 75/1507`
   - В текущем тесте нет такого измерителя.
   - Фикстура `adr003_direct_path_text_patterns_snapshot.json` содержит 154 `text_table`, а не 75.
   - Генератор `text_table` в `tests/test_adr003_regex_understanding_moratorium.py:433-461` берёт любое uppercase-имя с `ACTION/ALIAS/CUE/.../TOPIC`; он не проверяет `>=8`, не проверяет русский текст и не считает элементы.
   - Мои read-only проверки простыми AST-фильтрами не воспроизводят `75/1507`: для `DIRECT_PATH_PATTERN_FILES` получилось 29 таблиц с >=8 кириллическими строками и 656 таких строк; для всех `src/mango_mvp/channels` получилось 48/989. Значит `75/1507` сейчас не имеет машинно закреплённой семантики в проверяемом генераторе.

6. `snapshot 832`
   - Факт: `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` содержит 832 строки.
   - Разбивка фактической фикстуры: 381 `regex_call`, 167 `string_contains`, 154 `text_table`, 130 `marker_helper_call`.
   - `381 regex_call` = 166 `re.compile` + 215 inline `re.*` в `DIRECT_PATH_PATTERN_FILES`; это не `190 + 233`.
   - В формуле приёмки `190 + 233 + 255 + 75 = 753`, но текущий снимок 832 содержит ещё 167 `string_contains` и другие определения таблиц, поэтому эта формула не может быть проверкой полноты текущего генератора.

## Найденные проблемы ТЗ

1. Блокирующее: смешаны фактические количества и потолки бюджета.
   - `197` и `190` являются фактами по regex-снимку.
   - `255` является бюджетом, а не фактическим числом записей.
   - `CHANNEL_REGEX_BUDGET` тоже потолок: сумма 200 при снимке 197.
   - Нужно заменить в ТЗ "255 маркерных вызовов" на два отдельных числа: "budget ceiling 255" и "actual marker-helper calls по текущему счётчику 172"; если нужна разметка только строк фикстуры, отдельно указать "literal marker-helper rows 130".

2. Блокирующее: генератор не покрывает заявленную приёмку `190 + 233 + 255 + 75`.
   - `_regex_snapshot()` покрывает все `re.compile` в каналах, но без `line`.
   - `_direct_path_text_pattern_snapshot()` покрывает 19 вручную перечисленных файлов, а не те же 15 live-regex файлов и не весь живой периметр.
   - В текущей фикстуре нет 233 inline как отдельной группы, нет 255 marker calls, нет 75 таблиц и нет 1507 элементов.
   - Нужно в ТЗ явно выбрать один источник карты: либо расширяется текущая фикстура 832 и тогда приёмка считается по её `node_kind`, либо строится новый единый инвентарь с точным определением четырёх типов записей.

3. Блокирующее: требование "каждая запись с `file:line`" не выполняется текущими фикстурами.
   - В обеих фикстурах есть `path`, но нет `lineno`/`col_offset`.
   - Без line/col невозможно доказать, что YAML-разметка покрыла именно эту запись, а не похожую запись рядом.

4. Блокирующее: `bucket` нельзя однозначно добавить без стабильного `row_id`.
   - В `adr003_direct_path_text_patterns_snapshot.json` уже есть 15 групп полностью одинаковых строк-дублей, то есть 15 "лишних" строк невозможно различить по текущим полям.
   - Примеры дублей: `":\" in text` в `dialogue_memory.py::_parse_recent_messages`, `"online" in text` в `telegram_pilot_context_builder.py::_record_matches_requested_format`, `"онлайн|дистанц"` в `required_fact_keys_for_message`.
   - Если два параллельных исполнителя размечают один и тот же natural key, merge не покажет, какую физическую строку они имели в виду.
   - Вывод: без `row_id` на базе `path:lineno:col_offset:node_kind:symbol/hash` или без включения `lineno/col_offset` в запись расширение `bucket` не является однозначным.

5. Существенное: текущий `text_table` слишком широкий и одновременно не считает то, что просит ТЗ.
   - Он ловит 154 uppercase-assignments по имени переменной.
   - Он не считает словари/списки как элементы карты, не фильтрует русские фразы и не отличает таблицу понимания от технического словаря.
   - Разметка "75 таблиц / 1507 элементов" поверх такой фикстуры будет ручной интерпретацией, а не проверяемой приёмкой.

6. Существенное: `string_contains` есть в фикстуре, но исчезает из формулы приёмки.
   - Во втором снимке 167 `string_contains`.
   - Если они считаются пониманием по подстроке, они должны попасть в карту и в приёмочные суммы.
   - Если не считаются, генератор должен объяснить, почему они в snapshot 832, но вне `190+233+255+75`.

## Необходимые правки ТЗ до реализации

1. Ввести каноническую единицу карты:
   - `compiled_regex`: `re.compile`.
   - `inline_regex`: inline `re.search/match/fullmatch/findall/finditer/split/sub/subn`.
   - `marker_helper_call`: отдельно `literal_marker_helper_call` и `all_marker_helper_call`, если нужны оба слоя.
   - `text_table`: точное правило таблицы и правило подсчёта элементов.
   - `string_contains`: либо включить как отдельный тип, либо явно исключить с причиной.

2. Добавить в шаг 1 обязательный первый подшаг: генератор должен выдавать `row_id`, `path`, `lineno`, `col_offset`, `node_kind`, `symbol`, `stable_hash`.
   - `row_id` должен быть детерминированным и уникальным в пределах полного снимка.
   - YAML-разметка должна ссылаться на `row_id`, а не на preview/hash без позиции.

3. Развести факт и бюджет в тексте приёмки:
   - `regex snapshot actual = 197`.
   - `live compiled regex actual = 190`.
   - `regex budget ceiling = 200`, live без reporting ceiling = 193.
   - `inline actual = 233` только для явно названного набора 15 файлов.
   - `marker helper budget ceiling = 255`, текущий actual по тестовому счётчику = 172, rows in 832 snapshot = 130.
   - `text_table rows in 832 snapshot = 154`; `75/1507` оставить только после добавления машинного счётчика, который это воспроизводит.

4. Исправить приёмку "Карта: 190 + 233 + 255 + 75 записей".
   - Сейчас это не сумма текущего генератора.
   - Нужно заменить на проверку по фактическому output генератора: `0 unclassified rows` по выбранному набору и отдельные totals по каждому `node_kind`.

5. Уточнить разделение параллельной работы.
   - Нельзя делить только по "ведро 1/4" и "ведро 2/3", пока `bucket` ещё не размечен.
   - Без `row_id` безопасное деление должно быть по непересекающимся `path` или по заранее сгенерированному списку `row_id`.

## Что не надо делать

Не надо "подгонять" фикстуры под числа из ТЗ. Правильный ход: сначала сделать воспроизводимый инвентарь с однозначными идентификаторами и настоящими totals, затем обновить ТЗ/приёмку под этот измеритель.
