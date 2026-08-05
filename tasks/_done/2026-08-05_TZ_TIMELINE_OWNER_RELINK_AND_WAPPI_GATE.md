> DONE 2026-08-05 15:09 | ветка codex/timeline-owner-relink-fix | codex

> TAKE 2026-08-05 14:06 | ветка codex/timeline-owner-relink-fix | codex

Ветка: codex/timeline-owner-relink-fix
Зоны: src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/nightly_service.py, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_store.py tests/test_customer_timeline_nightly_service.py
Семантический-аудит: да

# Customer Timeline: перенос владельца события и честный Wappi gate

## Проблема

1. `upsert_event()` при смене `customer_id` снимает производные зависимости, но
   не переоценивает уже записанную пару `duplicate.superseded_by=canonical`.
   После mail-link enrichment один и тот же внешний email оказался связан с
   разными клиентами: красная линия «один клиент не видит данные другого» может
   нарушаться через устаревшую дедупликацию и bot-context.
2. Nightly считает Wappi-источник успешным по status/count, даже когда его отчёт
   явно содержит `attribution_complete=false` или `publish_ready=false`.

## Образ результата и бизнес-польза

- После переноса события к другому клиенту ни одна активная связь
  `superseded_by` не пересекает владельцев; соответствующая память не остаётся у
  прежнего клиента.
- Точная связь нескольких детей в одной семье не ломается: правило касается
  владельца конкретного события, а не общего телефона.
- Wappi Telegram/MAX не даёт зелёный publish gate, пока привязка не полна.
- Исправление использует существующие store/gate helpers, не создаёт новый
  контроллер, полную копию БД или новый feature flag.

## Минимальное решение

1. В общем `upsert_event()` при реальной смене владельца переоценить только
   затронутые связи, где событие является дублем или canonical. Несовместимые
   cross-customer ссылки снять и синхронно убрать их из bot-context/FTS.
2. Расширить существующий required-source proof для Wappi двумя уже имеющимися
   полями `attribution_complete` и `publish_ready`.
3. Не делать полный rebuild FTS, если существующий точечный helper достаточен.

## Приёмка

1. Сквозной тест: событие было у A и superseded, затем переехало к B; после
   upsert оно не скрыто чужим canonical, память A его не выдаёт, B видит только
   своё событие.
2. Отрицательный контроль: смена несмыслового поля при том же customer_id не
   снимает корректную дедупликацию.
3. Семейный сценарий: общий телефон разных детей не вызывает слияние или снятие
   корректной связи сам по себе.
4. Wappi `status=ok`, но любой из двух флагов false => nightly `partial/blocked`;
   оба true и закрытые balances => прежний PASS.
5. На копии свежей staging-БД число cross-customer superseded links становится
   нулём без изменения исходной БД; `quick_check=ok`, FK=0.
6. Точечные тесты и полный pytest не дают новых падений.

## Ограничения

- Только код, тесты и временная копия staging-БД.
- Не писать в рабочую Timeline, Tallanto, AMO, CRM или Wappi.
- Не публиковать `latest_published`.
- Новый runtime-код: до 50 строк; новых файлов, флагов и зависимостей — ноль.

## СТОП

- Исправление требует записи в рабочую Timeline или внешнюю систему.
- Связь нельзя снять адресно и требуется полный rebuild FTS/БД без отдельного
  измерения времени и доказательства отсутствия более узкого helper.
- Появились чужие изменения в заявленных файлах либо актуальный код уже решает
  тот же инвариант другим способом.
