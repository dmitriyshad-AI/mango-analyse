> DONE 2026-07-29 11:04 | ветка codex/timeline-canonical-identity | codex

> TAKE 2026-07-29 10:35 | ветка codex/timeline-canonical-identity | codex

Ветка: codex/timeline-canonical-identity
Зоны: src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/canonical_readonly_import.py, src/mango_mvp/customer_timeline/ingestion.py, src/mango_mvp/customer_timeline/next_step_resolver.py, src/mango_mvp/customer_timeline/source_policy.py, src/mango_mvp/customer_timeline/stage4b_bot_opening.py, scripts/import_telegram_export_to_timeline.py, tests/test_bot_safe_runtime_context.py, tests/test_customer_timeline_bot_safe_summary.py, tests/test_customer_timeline_canonical_readonly_import.py, tests/test_customer_timeline_ingestion.py, tests/test_customer_timeline_stage4b_bot_opening.py, tests/test_import_telegram_export_to_timeline.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_stage4b_bot_opening.py tests/test_import_telegram_export_to_timeline.py
Семантический-аудит: да

# ТЗ: восстановить полезную память Customer Timeline в живом пути бота

## Цель

Устранить доказанные причины потери полезного контекста без ослабления защиты персональных данных, бренда и надёжности личности.

## Изменения

1. Телефонная маска не должна принимать дату-время, хеш, номер дома, диапазон классов или режим `24-7` за телефон. Настоящие телефоны по-прежнему маскируются.
2. Сохранить правильный порядок `маскировка -> проверка остаточной утечки`. Не принимать предложение сканировать исходный текст как запрещающий гейт: оно выбрасывает корректно обезличиваемые фрагменты.
3. Несодержательные звонки не должны открываться боту. Решение только по структурному полю источника (`record.contentful`/канонический тип), не по словам текста.
4. Проверить сырьём текущий контракт `brand_context_authorized`. Не создавать второй механизм, если ключ уже пишет канонический модуль и проверяет store/opening.

## Приёмка

- граничные тесты телефона зелёные;
- настоящий телефон и имя не проходят в итоговый контекст;
- дата-время сохраняется;
- звонок с `contentful=Нет` закрыт, с `contentful=Да` доступен при остальных зелёных условиях;
- измерено на текущей staging-базе: сколько фрагментов проходит каждый слой и сколько реальных сильных клиентов получают непустой контекст;
- есть отрицательные контроли и независимый смысловой/ломающий аудит;
- полный pytest зелёный;
- нет новых флагов, зависимостей и внешних записей.

## СТОП

Остановиться без изменения кода, если структурный признак содержательности звонка отсутствует в реальном событии, бренд-проверку нельзя доказать сырьём или правка требует ослабить защиту личности/бренда/персональных данных.

## Запреты

Не менять рабочую Timeline, не публиковать staging, не писать во внешние системы, не ослаблять identity/brand/PII floors.
