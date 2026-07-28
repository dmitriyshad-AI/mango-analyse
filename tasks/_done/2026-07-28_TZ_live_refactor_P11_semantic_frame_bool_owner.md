> DONE 2026-07-28 23:03 | ветка main | codex

> TAKE 2026-07-28 22:48 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/contracts.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, tests/test_exact_runtime_dedup_contract.py, tests/test_single_owner_registry.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_direct_p0_text_hygiene.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# P11: один владелец широкого преобразователя SemanticFrame bool

## Цель

Удалить две одинаковые реализации преобразования `must_handoff` в `provider`
и `policy_routing`, сохранив прежние приватные имена как прямые ссылки на один
владелец в листовом `contracts.py`.

## Граница

- Не менять `text_hygiene._semantic_frame_bool`: он намеренно не принимает
  числовые `0/1` и строки `y/n/on/off` и защищает клиентский P0-текст.
- Не менять P0-маршрутизацию, регулярные выражения, флаги и внешние данные.
- Не менять цепочку реэкспорта `subscription_llm_parts.__init__`.

## Приёмка

- Два прежних имени являются объектно тем же callable одного владельца.
- Таблица входов `None/bool/0/1/float/строки/мусор` полностью совпадает с
  поведением до правки.
- Строгий преобразователь `text_hygiene` остаётся отличающимся на `1`, `y`,
  `on` и это закреплено тестом.
- Целевые и полные тесты зелёные, мораторий на смысловые regex зелёный.
- Нет новых флагов, файлов кода и зависимостей; нетестовый diff отрицательный.

## СТОП

- Любое изменение результата или исключения на зафиксированной таблице входов.
- Любое изменение `text_hygiene`, P0-маршрута или реэкспортной цепочки.
- Красный мораторий или непонятная новая грязь вне зон ТЗ.

## Результат

- Один владелец в `subscription_llm_parts/contracts.py`; два прежних имени ссылаются на него.
- Поведение на граничной таблице совпадает; строгий P0-парсер не изменён.
- Полный `pytest`: 5039 passed, 2 skipped, 2 известных предупреждения.
- Независимый архитектор: PASS.
- Нетестовый код: −15 строк; новых файлов, флагов и зависимостей нет.
