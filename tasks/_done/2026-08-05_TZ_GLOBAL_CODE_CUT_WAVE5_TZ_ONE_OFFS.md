> DONE 2026-08-05 00:32 | ветка main | codex

> TAKE 2026-08-05 00:13 | ветка main | codex

Ветка: main
Зоны: scripts/build_tz16_profiles_v7.py, scripts/compute_tz16_rerun_tail.py, scripts/build_tz19_calls_review_table.py, scripts/build_tz19_tail_bundle.py, scripts/import_tz19_analyze_tail_results.py, scripts/run_tz121_brand_e_followup_real.py, scripts/run_tz121_brand_e_micro_shadow.py, scripts/run_tz121_deal_a_gold_measure.py, scripts/run_tz121_outcome_b_micro_shadow.py, scripts/run_tz121_question_catalog_c_hybrid_shadow.py, tests/test_tz16_profiles_v7_build.py, tests/test_tz16_rerun_tail.py, tests/test_tz19_analyze_tail_import.py, tests/test_tz19_calls_review_table.py, tests/test_tz19_tail_bundle.py, tests/test_tz121_brand_e.py, tests/test_tz121_brand_e_followup_real.py, tests/test_tz121_deal_a_gold_measure.py, tests/test_tz121_outcome_b.py, tests/test_tz121_question_catalog_c_hybrid_shadow.py, tests/fixtures/tz121_brand_e_micro_gold.csv, tests/fixtures/tz121_outcome_b_micro_gold.csv, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_profile_builder.py tests/test_refresh_customer_profiles.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_outcome_linker.py tests/test_question_catalog_classifier_v2.py tests/test_question_catalog_calibration_v2.py tests/test_analyze.py
Семантический-аудит: да

# Глобальная уборка, волна 5: завершённые TZ16/TZ19/TZ121 one-off

## Проблема

В корне `scripts/` остаются десять одноразовых построителей, миграторов и
shadow-измерителей июня 2026 года. Их задания завершены и переведены в primary
либо поглощены нынешним Customer Timeline, но код и тесты продолжают выглядеть
как действующие механизмы. Скрипты занимают 3806 строк, их собственные тесты и
две микро-фикстуры — ещё 1484 строки.

## Образ результата и бизнес-польза

В проекте остаются текущие владельцы поведения: Customer Profile builder,
Customer Timeline import/read, outcome linker, question catalog classifier и
brand inference. Завершённые миграции и старые shadow-ворота не создают второй
способ пересборки данных или принятия решения. Разработчик видит текущую
архитектуру, а не историю пяти старых ТЗ.

## Перед удалением доказать

1. Ни один файл не вызывается runtime, runbook, launchd/cron, текущим CLI или
   незавершённым ТЗ.
2. Каждый тест из scope проверяет удаляемый runner/migration, а не уникальный
   контракт текущего `src/`.
3. Основные свойства остаются покрыты текущими тестами владельцев.
4. Одноразовые импорты TZ19 уже применены либо больше не являются штатным путём;
   удаление не меняет сами БД и сохранённые результаты.
5. Graphify используется как карта; все выводы подтверждаются сырьём.

## Изменение

- Удалить ровно 10 перечисленных скриптов, 10 привязанных тестовых модулей и 2
  фикстуры, если независимый аудит не найдёт живой контракт.
- Не переносить их функции, не создавать архивный package, wrappers, флаги или
  новые тесты только ради старых CLI.
- Исторические отчёты, Git-историю, runtime и данные не удалять.

## Приёмка

1. Удалено не менее 5290 строк, добавлено 0 строк рабочего/тестового кода.
2. Новых файлов кода, флагов, зависимостей и LLM-вызовов: 0.
3. Прямой collect фиксирует точное число удаляемых тестов; полный collect после
   удаления уменьшается только на это число.
4. Текущие профильные/Timeline/outcome/question-catalog/analyze тесты зелёные.
5. Полный pytest не имеет новых падений против базы `7e2648bc`.
6. Архитектор, ломатель, бизнес-аудитор и уборщик дают независимые вердикты.

## Стоп

- Найден живой потребитель или незаменённый текущий бизнес-контракт.
- Незавершённая задача ссылается на конкретный runner как обязательный путь.
- Удаление требует правки `src/` или нового кода.
- Нужен запуск live, запись в БД/AMO/Tallanto или удаление данных.

## Бритва

Варианты: оставить; перенести helpers в `src/`; удалить завершённый пакет.
Выбрано удаление: перенос сохранил бы историческую архитектуру без живого
потребителя, а Git уже является достаточным архивом.

## Результат 2026-08-05

- Удалено: 10 scripts, 10 test modules, 2 fixtures, `5290` строк.
- Добавлено рабочего/тестового кода: `0` строк.
- Профильные тесты: `126 passed`; import-bomb: `126 passed`.
- Полный collect: `5379 -> 5346`, ровно минус 33 теста.
- Полный pytest: `5335 passed, 8 failed, 3 skipped`; новых падений нет.
- Graphify и сырой поиск не нашли runtime/CLI/runbook/launchd/cron потребителей.
- Архитектор, ломатель, бизнес-аудитор, уборщик и финальный ломатель подтвердили
  удаление. Уникальный hardcode TZ121 C не перенесён в runtime.
- Runtime, БД, AMO, Tallanto, CRM, Wappi и клиентские сообщения не менялись.
