# Промт для Кодекса: прогон тест-пака бизнес-инвариантов (READ-ONLY)

Этот пакет написан Claude как независимым аудитором. Тесты проверяют **бизнес-смысл**
четырёх областей, а не «зелёность» CI. Твоя задача — прогнать их в своей среде (с БД и
зависимостями) и собрать падения. **Ничего не чинить и не править в `src/`.**

## РЕЖИМ

- READ-ONLY по продуктовому коду. НЕ менять `src/`, `product_data/`, `Mango_Bot_KB_FINAL*`,
  `stable_runtime/`. Никаких `git reset/checkout/clean`.
- Разрешено: запускать тесты, читать код, создавать ТОЛЬКО свой файл с находками
  (см. ниже) и, при необходимости, временные фикстуры в `/tmp` или в
  `D1_audit_backlog/codex_tests/_fixtures/` (новая папка, не трогать существующее).
- Запрещено по политике проекта: запуск ASR, Resolve+Analyze, запись в AMO/Tallanto/CRM,
  отправка клиентам, live-write скрипты.

## ЧТО ПРОГНАТЬ

Из корня проекта, с зависимостями (как обычно у тебя — uv/venv с pandas, sqlalchemy и т.д.):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_matching.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_deal_writeback.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_customer_timeline.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_productization.py
```

Каждый файл сам печатает `PASS=N FAIL=M` и список `[FAIL] ...`. Exit code 0 = всё зелёное.
Файлы написаны как самостоятельные скрипты (стиль `D1_semantic_roles_reference/tests`),
pytest не обязателен; но `pytest D1_audit_backlog/codex_tests/` тоже сработает, если
переименовать функции под discovery (не требуется).

В среде Claude (без БД-зависимостей) три файла дали полностью зелёный результат, один —
после правки фикстуры. Ожидаемая база при первом прогоне у тебя:
`test_matching` 36/0, `test_deal_writeback` 27/0, `test_customer_timeline` 81/0,
`test_productization` 42/0. Если у тебя цифры расходятся — это сигнал.

## ЧТО ПОКРЫВАЮТ ТЕСТЫ (по областям)

1. **`test_matching.py`** — сопоставление клиента/сделки.
   Инвариант: разные люди не слипаются; один человек в разных форматах телефона матчится;
   пустой/мусорный ключ не даёт ложный матч; неоднозначность блокируется, а не выбирается наугад.
   Модули: `tallanto_matching`, `tallanto_deal_ranking`, `crm_entity_resolver`,
   `phone_context` (скелет).
2. **`test_deal_writeback.py`** — запись сделок.
   Инвариант: всегда dry_run без approval; низкое качество блокирует; rollback восстанавливает
   только при совпадении текущего значения, иначе skipped/manual; идемпотентность по resume-ключам.
   Модули: `deal_writeback`, `deal_quality_gate`, `amo_rollback`.
3. **`test_customer_timeline.py`** — история клиента.
   Инвариант: импорт read-only (контракты безопасности, нет сети/subprocess); PII не попадает
   в слой бота и маскируется в слое менеджера; один телефон у двух customer_id → конфликт,
   у одного → нет конфликта.
   Модули: `safety`, `read_api`, `ingestion`, `canonical_readonly_import`, `read_api` (скелет на живой БД).
4. **`test_productization.py`** — ASR-гейт, изоляция арендаторов, идемпотентный capture.
   Инвариант: исполнение ASR невозможно без approval-record (execution_allowed всегда False,
   hard_guards.run_asr=False); tenant без tenant_id блокируется; capture не дублирует event_key.
   Модули: `asr_execution_approval_gate`, `tenant_isolation` (скелет на БД), `capture`.

## КАКИЕ ФИКСТУРЫ ДОКРУТИТЬ (помечены `# TODO(codex)` в коде)

- **`test_matching.py` → `test_phone_context_readonly_skeleton`**: собрать временную
  экспортную папку с `master_contacts_ru.csv` и `master_calls_ru.csv` (колонка
  `Телефон клиента`), замокать `phone_context._latest_export_dir` (или
  `settings.source_workspace_root`), сбросить `phone_context._CACHE`. Проверить:
  пустой/мусорный телефон → `None`; один человек в двух форматах → один и тот же `contact_row`.
- **`test_deal_writeback.py`**: фикстур БД не требует (всё на чистых функциях и фейках).
  Если имена AI-полей в `deal_text_builder` изменятся — обновить `_good_row`/каталог.
- **`test_customer_timeline.py` → `test_read_api_read_only_skeleton`**: собрать временную
  `customer_timeline.sqlite` через `CustomerTimelineSQLiteStore`, открыть
  `CustomerTimelineReadApi.open(...)`, проверить `health()['read_only'] is True` и что
  `bot_context(audience='bot')` не содержит телефон/email/customer_id/raw payload.
- **`test_productization.py` → `test_tenant_isolation_db_skeleton`**: собрать временную
  `product.sqlite` (таблицы `tenants`, `product_calls`: две строки с разными `tenant_id`
  + одна без `tenant_id`), вызвать `build_tenant_isolation_report(...)`, проверить
  `summary['blocked'] >= 1` и `validation_ok is False`; отдельно — что выборка по `tenant_id='t1'`
  не возвращает строки `t2`.
- **`test_productization.py` → `_event`**: если конструктор `TelephonyCallEvent` изменится,
  дополнить обязательные поля из `productization/contracts.py`.

## ФОРМАТ ОТЧЁТА (твой отдельный файл, НЕ менять чужие)

Создай ровно один файл:

```
audits/_inbox/codex_testpack_findings_<TS>.md
```

где `<TS>` — текущий timestamp вида `20260525_092834`. Для каждого падения — строка-блок:

- **Гипотеза**: что тест ожидал (бизнес-инвариант) и почему.
- **Факт**: фактический вывод (`[FAIL] ...`, exit code, трейс если был).
- **Это баг продукта или теста?** — отметь, похоже ли на реальный смысловой дефект
  (`product`) или на хрупкую фикстуру/устаревшее имя поля (`test_fixture`).
- **Severity**: `P0` (нарушен инвариант безопасности: запись без approval, утечка PII,
  ложный матч разных людей, смешение арендаторов) / `P1` (нарушен бизнес-инвариант, но не
  безопасность) / `P2` (косметика/хрупкость теста).

В конце отчёта — сводка: PASS/FAIL по каждому файлу, список оставшихся `TODO(codex)` фикстур,
и явный вывод: «есть ли среди падений хоть один `product`+`P0/P1`». **Не чинить.** Решение о
правках принимает Дмитрий, исправления вносит Кодекс отдельным заходом.
