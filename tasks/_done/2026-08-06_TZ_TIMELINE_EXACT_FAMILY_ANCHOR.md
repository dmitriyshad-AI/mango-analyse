> DONE 2026-08-06 08:53 | ветка codex/timeline-exact-family-anchor-20260806 | codex

> TAKE 2026-08-06 07:08 | ветка codex/timeline-exact-family-anchor-20260806 | codex

Ветка: codex/timeline-exact-family-anchor-20260806
Зоны: scripts/import_tallanto_payments_to_timeline.py, src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/customer_timeline/family_graph.py, tests/test_import_tallanto_payments_to_timeline.py, tests/test_bot_safe_runtime_context.py, tests/test_customer_timeline_family_graph.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_import_tallanto_payments_to_timeline.py tests/test_customer_timeline_store.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_family_graph.py
Семантический-аудит: да

# Customer Timeline: пересчёт exact Tallanto identity после D-110

## Проблема

На HEAD `85facd09` после закрытия устаревших
`tallanto_identity_ambiguous` из conflict-block вышли 237 exact-владельцев.
У 235 профиль сохранил исторический `identity_status=ambiguous`: merge обязан
держать неоднозначность, пока конфликт открыт, но после D-110 отдельного
пересчёта нет. Поэтому `family_graph` и рабочий AMO reader продолжают честно
блокировать уже доказанного владельца.

Два клиента дополнительно имеют слабую строку `status=excluded`.
Bot-safe reader сейчас считает такую уже исключённую улику полноценным
неуверенным ребёнком и блокирует всю семью.

## Образ результата

После закрытия D-110 профиль становится `strong` только при одном Tallanto
student ID, единственном exact/manual владельце, совпадающем активном событии и
отсутствии другого открытого identity/family-конфликта. `family_graph` не
переопределяет профиль. Строка `excluded` хранится для аудита, но не блокирует
соседнего уверенного ребёнка; `excluded-only` остаётся закрытым.

Менеджер получает семейный контекст и исторический признак оплаты только своего
клиента. Никаких client-send, AMO/Tallanto write или смены production pointer.

## Рассмотренные варианты

1. Подавить ambiguity внутри `family_graph`: отвергнуто после полного прогона,
   потому что открыло 999 строк вместо целевых 235.
2. Удалять `excluded` из семейного графа: отклонено, теряется улика.
3. Выбранный корень: переиспользовать строгую проверку D-110 для открытых и
   закрытых конфликтов, затем штатным `upsert_customer` пересчитать профиль.

## Реализация

- Вынести существующую проверку D-110 в один helper без N+1 запросов.
- После conflict reconciliation идемпотентно повысить только строгих
  resolved-владельцев `ambiguous -> strong` через существующий store.
- Пометить владельца изменения в metadata и обратимо вернуть только этот
  `strong` в `ambiguous`, если exact-доказательство исчезло или открылся конфликт.
- Читать proof и blockers под штатным writer lock; промежуточный сбой может
  оставить профиль только в более строгом `ambiguous`, но не открыть его ошибочно.
- Не ослаблять `shared_family_phone`, direct/duplicate Tallanto ID,
  несколько валидных детей и runtime conflict gate.
- В bot-safe family projection исключать `excluded` из кандидатов, сохраняя
  fail-closed для `excluded-only`.
- Раскладывать сохранённые составные классы на атомарные значения, чтобы
  повторный family rebuild не наращивал собственный прошлый результат.
- Не добавлять модуль, флаг, таблицу, зависимость или новый resolver.

## Приёмка

1. `1`: exact owner с положительным `customer_purchases_v1` меняется
   `family blocked -> single`; в prompt появляется только исторический
   `payment_history=fact_confirmed`.
2. `10`: пять положительных и пять NEG; прогноз записан до прогона,
   фактический prompt проверен по тексту.
3. `всё`: на свежем APFS-клоне до D-110 закрываются 1 902 TIA и ровно 235
   профилей меняются `ambiguous -> strong`; после family rebuild ровно 235
   основных строк меняются `ambiguous/low -> confident/high`, а две отдельные
   `excluded`-улики сохраняются. В рабочей проекции 122 семьи становятся
   `single` при совпавшем бренде, 113 честно остаются `needs_clarification`.
   Exact AMO-опора есть только у 7 из 235: в контрольном прогоне 5 дали полезный
   контекст и 2 потребовали уточнение. Этот блок не решает покрытие AMO и бренда.
   Первичные события, identity links и purchases не меняются; повтор идемпотентен.
4. NEG: direct ID conflict, два владельца одного ID, несколько Tallanto ID,
   weak-only evidence, shared family phone, несколько валидных детей и
   открытый конфликт остаются закрыты.
5. `quick_check=ok`, FK=0, точечные и смежные тесты зелёные.
6. Отрицательный контроль: отключить profile reconcile либо вернуть `excluded`
   в selector - соответствующий сквозной тест краснеет.
7. После открытия direct ID conflict профиль с D-110-меткой возвращается в
   `ambiguous`; настоящий `strong` без этой метки не меняется.

## Границы

- Только APFS-клон staging в write-режиме.
- Production Timeline, внешние системы, сообщения и runtime pointer не менять.
- Полный nightly/cutover не входят в этот блок.

## СТОП

- Exact Tallanto owner не единственный или не совпадает с владельцем события.
- После правки открывается direct/duplicate/shared-family NEG-кейс.
- Меняется число первичных событий, identity links, purchases или конфликтов.
- Сквозной prompt содержит данные другого клиента.
