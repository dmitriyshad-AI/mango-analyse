> DONE 2026-07-29 05:43 | ветка main | codex

> TAKE 2026-07-29 05:12 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/contracts.py, src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/customer_timeline/ingestion.py, src/mango_mvp/customer_timeline/store.py, tests/test_customer_timeline_nightly_service.py, tests/test_customer_timeline_ingestion.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_ingestion.py
Семантический-аудит: да

# ТЗ: восстановить уникальность точных идентификаторов Timeline

## Сырой факт

- в staging 669 значений `tallanto_student_id` одновременно имеют класс
  `strong_unique` у нескольких клиентов;
- текущий resolver замечает новый конфликт, но оставляет старые ложные связи;
- один и тот же точный идентификатор не может быть надёжным у двух семей;
- сырьё и события удалять нельзя, склейка по похожести ФИО запрещена.

## Образ результата

1. Публикация блокируется, если любой точный идентификатор класса
   `strong_unique/manual` связан с несколькими клиентами.
2. При следующей полной карточке Tallanto старые конфликтные точные ссылки
   понижаются до `ambiguous`, а не удаляются.
3. Текущая полная карточка может повысить ровно одну ссылку только при
   непротиворечивом точном контакте; спор остаётся конфликтом.
4. Сырые события не удаляются и не склеиваются по имени.
5. Второй одинаковый проход не создаёт новых изменений.

## Минимальная реализация

- добавить в store один запрос точных конфликтующих ссылок;
- переиспользовать существующий механизм обновления IdentityLink и conflict;
- исключить уже пониженные exact-ссылки из множества авторитетных владельцев;
- добавить invariant в snapshot manifest и publish gate;
- закрыть тестами два клиента с одним Tallanto ID, один ребёнок в семье с двумя
  детьми, противоречащий телефон и повторный проход.

## СТОП

- не менять prod/staging/runtime в рамках кода и тестов;
- не удалять события, клиентов или ссылки;
- не объединять по ФИО;
- не создавать второй resolver, таблицу или флаг.

## Приёмка: готово, когда

- точечные и полные CPU-тесты зелёные;
- отрицательный контроль доказывает, что publish gate живой;
- конфликтные ссылки сохраняются как ambiguous;
- свежая непротиворечивая карточка восстанавливает одну strong-связь;
- создан audit pack, один коммит и push в оба зеркала.
