> DONE 2026-07-28 01:39 | ветка main | codex

> TAKE 2026-07-28 01:27 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/freshness.py, scripts/audit_owner_gate_semantic_sample.py, tests/test_customer_timeline_manager_dossier.py, tests/test_audit_owner_gate_semantic_sample.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_manager_dossier.py tests/test_audit_owner_gate_semantic_sample.py
Семантический-аудит: нет

# ТЗ: честная свежесть Timeline и fail-closed приёмка семей

## Проблема

Менеджерский гейт не требует источники оплат и посещений Tallanto. Импорт посещений
использует отдельный `run_kind`, который текущий запрос не видит. В `acceptance`
ошибка классификации Owner50 превращается в пустой лист и общий `OK`. Также CLI
позволяет построить реальную приёмку с пропущенной проверкой свежести.

## Сделать

1. Сделать `tallanto_crm_call` и `tallanto_attendance_api` обязательными источниками.
2. Карточки признавать свежими только по API-циклу `tallanto_cards_daily`, оплаты — только по API-source_ref, посещения — по завершённому increment и его курсору.
3. Требовать непустую границу данных для карточек, оплат и посещений.
4. Убрать CLI-обход свежести из режимов owner50/dossiers/acceptance.
5. При ошибке Owner50 в acceptance вернуть ненулевой код, blocked-манифест и не писать XLSX.
6. Если популяция содержит оплаты/посещения, выборка должна содержать их примеры.
7. Новые Excel выпускать под уникальным именем; манифест должен явно отличать текущий файл от старого.

## Приёмка

- пропавшие/пустые оплаты или посещения блокируют freshness;
- успешный attendance increment с курсором виден гейту;
- ошибка Owner50 не завершается `OK` и не создаёт XLSX;
- `--skip-freshness-gate` больше не принимается;
- существующие тесты Timeline зелёные.

## СТОП

- требуются live-запись, внешняя сеть или рабочая база;
- нужен новый флаг, файл механизма или зависимость;
- меняется клиентский текст, маршрутизация или P0-пол.

## Не делать

- не запускать ночной сбор;
- не читать и не менять рабочую Customer Timeline;
- не писать в AMO, Tallanto или Wappi;
- не добавлять LLM-вызовы.
