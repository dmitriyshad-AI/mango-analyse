> DONE 2026-07-29 06:18 | ветка main | codex

> TAKE 2026-07-29 06:04 | ветка main | codex

Ветка: main
Зоны: scripts/import_tallanto_payments_to_timeline.py, src/mango_mvp/customer_timeline/stage5_money_ingest.py, tests/test_import_tallanto_payments_to_timeline.py, tests/test_customer_timeline_stage5_money_ingest.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_import_tallanto_payments_to_timeline.py tests/test_customer_timeline_stage5_money_ingest.py
Семантический-аудит: да

# ТЗ: точно привязывать оплаты и абонементы Tallanto

## Сырой факт

- конфликт владельца оплаты закрывался по наличию contact_id, даже если в Timeline не было единственного надёжного владельца;
- ручная подтверждённая связь `manual` не учитывалась как надёжная при импорте и пересчёте покупок;
- десятки тысяч старых оплат/абонементов ещё ждут повторного прохода после свежих карточек.

## Образ результата

1. Оплата привязывается только к единственному владельцу `strong_unique/manual`.
2. Конфликт закрывается только после такой связи.
3. Два владельца одного Tallanto ID остаются ambiguous, первый не выбирается.
4. Поздняя полная карточка исправляет событие без дубля.
5. Полный пересчёт покупок учитывает manual и удаляет старую ложную агрегацию.

## СТОП

- не писать в Tallanto/AMO и не менять реальную staging/prod базу;
- не создавать второй importer/resolver/таблицу/флаг;
- не склеивать по ФИО.

## Приёмка: готово, когда

- strong и manual проходят одинаковые положительные сценарии;
- ambiguous не закрывает конфликт;
- повторный проход идемпотентен;
- целевые и полные CPU-тесты зелёные;
- audit pack, один коммит, push в оба зеркала.
