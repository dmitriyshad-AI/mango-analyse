# ТЗ MAIN — правка 4b: включить переворот умолчания (под щитом). 2026-05-30.

Автор: Claude #1. Исполнитель: MAIN Кодекс. 4a закрыта (decide_route + щит 18 вето зелёные, 432 теста,
проверено Claude #1 в песочнике; коммиты a695d391 → 431ee0e6). Это финальный шаг этапа и самый опасный:
бот отвечает клиенту напрямую, переворот меняет умолчание для всех «осторожных» уходов.

## Зачем

Механика переворота уже в `decide_route` (`subscription_llm.py:6593-6599`): при `allow_default_autonomy=True`
И `route=="draft_for_manager"` И `autonomy_ready` И `has_covering_fact` → `bot_answer_self_for_pilot`.
Сейчас guard (`subscription_llm.py:1434`) вызывает `decide_route(...)` БЕЗ `allow_default_autonomy` →
дефолт `False` → переворот выключен. 4b его включает.

## Что сделать

1. **Управляемый источник флага (стоп-кран, НЕ хардкод `True`):** функция типа
   `_default_autonomy_flip_enabled(context)`, читающая флаг из политики/контекста (напр.
   `autonomy_policy.get("allow_default_autonomy")` или `context.get("allow_default_autonomy")`),
   **default `False`**. Совместимо со стоп-краном `apply_public_autonomy_kill_switch`: при выключенной
   автономии переворот тоже выключен.
2. **Проброс:** `decide_route(result, client_message=..., context=...,
   allow_default_autonomy=_default_autonomy_flip_enabled(context))` (subscription_llm.py:1434).

Условия переворота НЕ ослаблять (уже в `decide_route` 6591-6599): `autonomy_ready` =
`_autonomy_enabled` + тема в матрице `AUTONOMY_MATRIX_SAFE_TOPIC_IDS` (23 безопасные); `has_covering_fact`
= client-safe факт покрывает ответ ИЛИ verified-шаблон. Если хоть одно нет → к менеджеру.

## Обязательный негативный контроль (правило #4) — СЕРДЦЕ 4b

При `allow_default_autonomy=True` ВСЕ 18 вето-категорий щита 4a остаются маршрутом к менеджеру:
- прогнать `test_pravka4_router_veto_shield_keeps_all_manager_routes` с включённым флагом → все 18 зелёные;
- ни одно вето не проскакивает в автоответ: P0 / pregate / refund-без-факта / бренд / выдумка /
  unsupported_entity / forbidden_scope / meta / ai_disclosure / p0_promise / unsupported_promise /
  operational / unstated / payment / unknown_brand / force_manager_only / semantic_unavailable / no_draft_fn.

Это сердце 4b: переворот меняет ТОЛЬКО случай «route осел в draft_for_manager как осторожное умолчание,
вето нет, факт есть» — и не трогает ни одно вето.

## Тест выхода (новый)

- Вход: `route="draft_for_manager"` (осторожное умолчание), тема в матрице, `has_covering_fact`, нет
  вето, флаг включён → ВЫХОД `bot_answer_self_for_pilot`.
- Контроль: тот же без покрывающего факта → `draft_for_manager`; с любым вето → к менеджеру; тема вне
  матрицы → `draft_for_manager`; флаг выключен → `draft_for_manager` (поведение 4a).

## После 4b

Поведение изменилось → юнит-щита мало. Нужен прогон на M1 (большой eval) для замера реальной
автономии / over_handoff / тона / hard_gate. Claude #1 соберёт финальный bundle (код 4b + v6.4-снимок)
после проверки 4b в песочнике.

## Ограничения

- Тесты не гонять (лимит main) — Claude #1 прогонит щит (флаг True) + smoke + тест автоответа. В main
  не мержить до зелёного И прогона M1.
- Флаг управляемый, default off (стоп-кран). Правило #1: источник флага и проброс подтвердить чтением.
- Маршрутные условия (autonomy_ready / has_covering_fact) НЕ менять — только включить флаг.
- Отчёт: где флаг и как пробрасывается; тексты негативного теста (18 вето при True) и позитивного
  (автоответ при покрытии).
