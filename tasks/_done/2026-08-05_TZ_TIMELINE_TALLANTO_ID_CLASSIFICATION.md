> DONE 2026-08-05 15:25 | ветка codex/timeline-tallanto-id-classification | codex

> TAKE 2026-08-05 15:21 | ветка codex/timeline-tallanto-id-classification | codex

Ветка: codex/timeline-tallanto-id-classification
Зоны: src/mango_mvp/customer_timeline/family_graph.py, tests/test_customer_timeline_family_graph.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_family_graph.py
Семантический-аудит: да

# Timeline: не считать технический master-contact ID учеником Tallanto

## Проблема

Старый canonical import создавал `tallanto_student_snapshot` с техническим
`source_id=tallanto:<customer_id>` и точным
`source_ref=master_contact:<customer_id>:tallanto`. Семейный граф принимает
любой `source_id` такого события за реальный ID ученика. На свежей staging-БД
это создаёт 7298 ложных `tallanto_student_id_not_in_family` из 7793.

## Образ результата и бизнес-польза

- Техническая сводка остаётся в Timeline, но не выдаёт себя за ребёнка.
- Реальные Tallanto student snapshot, оплаты, абонементы и посещения продолжают
  связываться по точному student ID.
- После пересчёта производного семейного графа ложный класс уменьшается на 7298;
  остающиеся 495 случаев становятся реальной очередью качества данных.
- Новый импортёр, таблица, regex, feature flag или зависимость не создаются.

## Минимальное решение

В существующей `_event_tallanto_student_id()` распознать точную структурную пару,
которую создаёт canonical import, сравнением с `customer_id`, и вернуть пустой ID.
Другие события не менять.

## Приёмка

1. Тест на техническую пару: ID пуст, причина атрибуции
   `missing_exact_tallanto_student_id`, а не `not_in_family`.
2. Контроли: реальный `tallanto:student:<id>` и реальные payment/abonement/
   attendance ID по-прежнему атрибутируются точно.
3. На read-only staging запрос независимо подтверждает исходные 7298.
4. Точечные и полные тесты не получают новых падений.
5. Runtime-дифф до 10 строк; новых файлов/флагов/зависимостей нет.

## Ограничения

- Не изменять рабочую или staging-БД в этом блоке.
- Не удалять исходные события и не переписывать настоящие Tallanto ID.
- Не добавлять эвристику по имени, телефону или похожести.

## СТОП

- Техническую пару нельзя отличить точными полями без эвристики.
- Правка требует изменения схемы, полного импорта или записи во внешние системы.
- Найден другой существующий канонический классификатор этого же класса.
