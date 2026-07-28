> DONE 2026-07-28 17:28 | ветка main | codex

> TAKE 2026-07-28 17:23 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/amocrm_runtime/amo_integration.py, src/mango_mvp/channels/night_funnel_shadow.py, tests/test_exact_runtime_dedup_contract.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_telegram_public_pilot_bots.py tests/test_crm_card_amo_writeback.py
Семантический-аудит: нет

# ТЗ: схлопнуть точные локальные дубли без смены поведения

## Цель

Сохранить все существующие имена, но убрать повторные тела там, где функции в
одном модуле имеют одинаковые сигнатуры и побайтово эквивалентную логику.

## Разрешённые пары

- `_contact_entity_endpoint` использует `_contact_update_endpoint`.
- `_flatten_lead_field_item` использует `_flatten_contact_field_item`.
- `append_lead_card` и `append_inbound_tee_record` используют
  `append_shadow_log`.

## СТОП

Остановиться, если сигнатуры, возвращаемые значения, исключения, кодировки,
сортировка JSON или ограничения путей различаются; если до правки красны
целевые тесты; если появляется чужая грязь в рабочем дереве.

## Приёмка

- Все прежние имена импортируются и дают прежний результат.
- Алиасы/обёртки используют одну физическую реализацию.
- Целевой и полный безопасный pytest зелёные.
- Новых флагов, зависимостей и рабочих файлов нет; рабочий код уменьшается.
