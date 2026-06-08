# ТЗ-фикс MAIN — коммит 0 (гард пустых фактов): route → bot_answer_self. 2026-05-30.

Автор: Клод 1. Фикс проверен в песочнике на 8e18c1a8.

## Проблема

Гард `empty_facts_no_fabrication` (коммит 8e18c1a8) механически верен — перехватывает пустой розыск
ДО `draft_fn`, убирая выдумку. Но `route="draft_for_manager"` (это была ошибка в моём ТЗ) сломал 2
теста, фиксирующих правильное поведение:
- `test_pipeline_missing_fact_uses_narrow_handoff` — ждёт route `bot_answer_self`;
- `test_phase1_coverage_noop_when_no_rfk` — ждёт route `bot_answer_self`.

Эти тесты правы: при пустых фактах бот должен сам честно ответить «уточню у менеджера» (narrow
handoff, автономия — цель проекта), а НЕ уходить в `draft_for_manager` (это лишний over-handoff). Гард
всё равно убирает выдумку (safe_fallback вместо draft_fn), но клиенту отвечает сам бот.

## Правка

В блоке гарда `empty_factual_answer_self` (pipeline.py ~1050) поменять route и reason оставить:
```python
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="bot_answer_self",          # было draft_for_manager
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="empty_facts_no_fabrication",
        )
```
И в trace_event выше route поправить на `"bot_answer_self"` для консистентности.

## Тест

Переписать новый гард-тест `test_empty_facts_guard_blocks_answer_self_fact_question_without_rfk`
(test_dialogue_contract_pipeline.py:~1359): ожидать `route == "bot_answer_self"` (не draft_for_manager),
draft_fn НЕ вызван (его перехватили), текст = safe_fallback (содержит «менеджер»), выдумки нет.

## Проверено Клодом 1 в песочнике (8e18c1a8 + этот фикс)

- 2 старых теста (narrow_handoff, coverage_noop) → зелёные;
- полный pipeline + subscription + dialogue_memory + smoke = 442 passed;
- остаётся только переписать сам гард-тест Кодекса под bot_answer_self (1 строка ассерта).

## Смысл

Гард = «при пустых фактах не вызывать модель-черновик (чтобы не выдумала), отдать честный safe_fallback,
но route остаётся bot_answer_self» — выдумки нет, автономия сохранена. Это и есть правильный баланс:
не выдумывать ≠ уходить к менеджеру.
