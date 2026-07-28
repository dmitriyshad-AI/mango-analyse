> DONE 2026-07-28 23:23 | ветка main | codex

> TAKE 2026-07-28 23:17 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/post_layers.py, src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/approved_context_pack.py, src/mango_mvp/customer_timeline/channel_preview_from_pack.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_approved_context_pack.py tests/test_customer_timeline_channel_preview_from_pack.py tests/test_customer_timeline_bot_safe_summary.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# P13: удалить невызванные функции и точные копии

## Цель

- Удалить `_deal_action_unknown` и `_latest_status`, для которых нет ни одного вызова,
  строковой ссылки, экспорта или динамической регистрации.
- Удалить копии `load_json_object`, `file_sha256`, `stable_unique` из
  `channel_preview_from_pack`; модуль уже зависит от `approved_context_pack`.

## Приёмка

- Глобальный поиск подтверждает ноль ссылок на две мёртвые функции.
- Три помощника второго модуля ссылаются на те же объекты одного владельца.
- Поведение по граничным входам совпадает.
- Целевые, P0/бренд, мораторий и полные тесы зелёны.
- Нет новых файлов кода, флагов, зависимостей и regex.

## СТОП

- Найден живой или динамический вызов двух кандидатов.
- Помощники различаются на любом граничном входе.
- Любая смысловая регрессия.

## Результат

- Две невызванные функции удалены; ссылок не осталось.
- Три копии помощников заменены импортами одного владельца; личность объектов закреплена тестом.
- Полный `pytest`: 5041 passed, 2 skipped, 2 известных предупреждения.
- Нетестовый код −44 строк; новых файлов, флагов и зависимостей нет.
