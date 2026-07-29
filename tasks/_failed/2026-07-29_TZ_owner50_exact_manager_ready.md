> FAIL 2026-07-29 07:18 | ветка main | codex | причина: formal exact-identity pass; semantic/business READY blocked until model-led P0, opt-out, interest and name review on fresh data

> TAKE 2026-07-29 06:44 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/manager_dossier.py, src/mango_mvp/customer_timeline/store.py, tests/test_customer_timeline_manager_dossier.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# ТЗ: Owner50 показывает только реально готовые семьи

## Сырой факт

- текущая витрина не использует пакет точных AMO/Tallanto ID при доказательстве личности;
- подготовленный патч принимал один exact ID у двух customer как два READY;
- открытый Tallanto identity conflict не блокировал READY;
- технический display_name мог попасть в «Кому» и следующий шаг.

## Образ результата

1. Точный AMO/Tallanto ID подтверждает семью только при единственном владельце `strong_unique/manual`.
2. Любой открытый identity-конфликт блокирует READY.
3. Доказательство клиента и доказательство человеческого имени разделены; техническое имя не выходит менеджеру.
4. Все члены семьи и дети входят в одну строку без N+1 запросов.
5. Квота 50 заполняется только READY; CANDIDATE/EXCLUDED показаны отдельно и не маскируют недобор.

## СТОП

- не выбирать первого владельца точного ID;
- не повышать имя по regex или похожести;
- не писать в AMO/Tallanto и не менять реальные базы;
- не создавать вторую витрину/классификатор.

## Приёмка

- регрессии на duplicate exact ID, open conflict, техническое имя и многодетную семью;
- отрицательный контроль живого пути;
- целевые и полные CPU-тесты зелёные;
- смысловой аудит, audit pack, один коммит, push в оба зеркала.
