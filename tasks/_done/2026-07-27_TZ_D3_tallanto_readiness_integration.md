> DONE 2026-07-27 12:30 | ветка main | codex

> TAKE 2026-07-27 12:09 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/customer_timeline/tallanto_cards_sync.py, src/mango_mvp/customer_timeline/tallanto_attendance_import.py, tests/test_customer_timeline_nightly_service.py, tests/test_customer_timeline_tallanto_cards_sync.py, tests/test_customer_timeline_tallanto_attendance_import.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_tallanto_cards_sync.py tests/test_customer_timeline_tallanto_attendance_import.py
Семантический-аудит: нет

# Tallanto: честная готовность ночного импорта

## Цель

Закрыть доказанные разрывы готового ночного контура Tallanto без второго импортера и без live-запуска:

1. Не импортировать неполную вселенную карточек, если хотя бы у одного контакта нет стабильного Tallanto ID.
2. Не продвигать курсор первого API-импорта посещений, пока не получено ни одного события.
3. Не публиковать ночную базу при нуле событий оплат/абонементов или посещений Tallanto.
4. При падении денежного импортера сохранять диагностический артефакт без сырого текста, ПДн и секретов.

## Границы

- Только staging и код/тесты.
- Не обращаться к live Tallanto и не менять env.
- Не писать в AMO/Tallanto/CRM или боевую Customer Timeline.
- Не переносить предложенный raw stdout/stderr: сохранять только тип, код возврата, размеры и SHA-256.
- Переиспользовать существующие импортёры, курсоры и publication gate.

## Приёмка

- Карточки: `checked_with_id == checked`, иначе apply заблокирован до записи.
- Посещения: пустая первая выборка оставляет курсор на месте; пустой инкремент при существующей истории допустим.
- Publication gate: нулевые Tallanto payments/attendance дают `missing/stale`, не `ok`.
- Диагностика ошибки имеет режим 0600 и не содержит исходный stdout/stderr.
- Целевые и полный pytest зелёные.

## СТОП

- Любая попытка записи в live Tallanto, AMO, CRM или боевую Customer Timeline.
- Невозможность удержать курсор при пустом первом импорте либо неполной идентификации.
- Попадание исходного stdout/stderr, телефона, email или секрета в диагностический отчёт.
- Красный целевой или полный pytest.
