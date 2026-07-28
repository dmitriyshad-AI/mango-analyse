> DONE 2026-07-28 21:59 | ветка main | codex

> TAKE 2026-07-28 21:50 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/answer_contract.py, src/mango_mvp/channels/conversation_intent_plan.py, src/mango_mvp/channels/draft_prompt_builder.py, src/mango_mvp/channels/telegram_pilot_context_builder.py, tests/test_exact_runtime_dedup_contract.py, tests/test_single_owner_registry.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_single_owner_registry.py
Семантический-аудит: нет

# P04: один владелец нормализации активного бренда

## Цель

Заменить четыре байт-идентичных тела в channels совместимыми алиасами на
`pilot_context.normalize_active_brand`, сохранив прежние локальные имена.

## Границы

- Не трогать `output_verification_floor`: другой контракт.
- Не трогать `few_shot_reference`: поддерживает `any`.
- Не трогать `knowledge_base/kc_context.py`: перенос создаст инверсию слоя.
- Не трогать `knowledge_base/answer_registry.py`: возвращает неизвестное значение.

## Приёмка

- Старые локальные имена существуют и являются ссылками на владельца.
- Граничные значения дают прежний результат.
- Реестр физических определений снижен с 8 до 4 и остаётся зелёным.
- Целевой и полный pytest зелёные.
- Рабочий diff отрицательный; флагов, зависимостей и новых файлов кода нет.

## СТОП

- Не менять динамический реэкспорт, P0, ПДн, факты или маршруты.
- Любое расхождение граничного результата останавливает конкретный алиас.
