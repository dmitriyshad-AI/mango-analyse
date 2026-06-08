# ТЗ MAIN — финальная правка over-handoff: semantic-recover в МЕСТАХ рождения хендоффа. 2026-05-31.

Автор: Клод 1. Глубокий разбор v2B (ed6c1791) показал точный корень. Это правка в правильное место,
после 3 промахов (draft-калибровка, key_coverage A, semantic B — все стояли ДО места рождения хендоффа).

## Корень (доказано по сырью)

Гейты A (key_coverage 1586) и B (semantic 1595) стоят ДО `_hard_check`+repair (1638+). На момент гейтов
draft ещё факт-ответ → не хендофф → гейты не входят (attempted=False на foton_05/unpk_02). Хендофф
рождается ПОЗЖЕ, в ДВУХ точках:
1. **fail-close критик-недоступен** (строка 1648 `if not semantic_available: return safe_fallback`,
   fallback_reason="semantic_check_unavailable") — 6/21 хендоффов нити (parallel 8 → faithfulness timeout).
2. **repair-fail** (строка ~1685, `if findings or unsupported` после repair → safe_fallback
   draft_for_manager) — ~15/21. Модель дала факт-ответ → критик пометил → repair не починил → шаблон
   SAFE_FALLBACK «Спасибо за сообщение. Передам менеджеру».

## Решение — ОДИН хелпер, применён в ОБЕИХ точках рождения хендоффа

Хелпер (рядом с _semantic_match): пытается отдать cite-only ответ из фактов вместо хендоффа, если факты
покрывают вопрос ПО СМЫСЛУ (тот продукт). Возвращает результат-ответ или None (тогда хендофф остаётся).
```python
def _semantic_recover_or_handoff(
    *, contract: AnswerContract, retrieval: RetrievalResult, draft: str,
    semantic_match_fn, faithfulness_fn, client_words: str,
    conversation, context, toggles, previous_bot_texts,
) -> DialogueContractPipelineResult | None:
    if (semantic_match_fn is None or not retrieval.facts
            or contract.answerability != "answer_self" or contract.is_p0):
        return None
    verdict = _semantic_match(semantic_match_fn, contract=contract, retrieval=retrieval,
                              client_words=client_words, draft=draft)
    if not (_truthy(verdict.get("covers")) and _truthy(verdict.get("same_product"))):
        return None
    # cite-only из проверенных retrieved-фактов: безопасен даже если faithfulness-критик недоступен
    replacement = _verified_empty_handoff_replacement(
        draft, contract=contract, retrieval=retrieval, client_words=client_words,
        faithfulness_fn=faithfulness_fn, toggles=toggles, context=context,
        previous_bot_texts=previous_bot_texts, allow_key_coverage=True)
    if not replacement:
        return None
    trace_event(context, "semantic_recover", {"replaced": True})
    return DialogueContractPipelineResult(
        draft_text=_avoid_repeating_text(replacement, conversation=conversation, contract=contract, facts=retrieval.facts),
        route="bot_answer_self", manager_only=False, contract=contract,
        facts=retrieval.facts, missing=retrieval.missing, repaired=True,
        fallback_reason="semantic_recover",
        semantic_match_attempted=True, semantic_match_replaced=True)
```

### Применение 1 — fail-close (критик недоступен), строка 1648
```python
    if not semantic_available:
        recovered = _semantic_recover_or_handoff(
            contract=contract, retrieval=retrieval, draft=draft, semantic_match_fn=semantic_match_fn,
            faithfulness_fn=None,  # критик недоступен; cite-only из фактов не нуждается в нём
            client_words=client_words, conversation=conversation, context=context,
            toggles=toggles, previous_bot_texts=previous_bot_texts)
        if recovered:
            return recovered
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        return DialogueContractPipelineResult(... как сейчас, "semantic_check_unavailable" ...)
```

### Применение 2 — repair-fail, строка ~1685 (где `if findings or unsupported` → safe_fallback)
ПЕРЕД возвратом safe_fallback вставить:
```python
        recovered = _semantic_recover_or_handoff(
            contract=contract, retrieval=retrieval, draft=draft, semantic_match_fn=semantic_match_fn,
            faithfulness_fn=faithfulness_fn, client_words=client_words, conversation=conversation,
            context=context, toggles=toggles, previous_bot_texts=previous_bot_texts)
        if recovered:
            return recovered
        # иначе — текущий safe_fallback draft_for_manager
```

Старый semantic-блок (1595) можно УБРАТЬ (он 0 раз сработал — стоит до места рождения хендоффа). Логику
он передал в хелпер. Если убирать рискованно — оставить, не вредит.

## Тесты + НЕГАТИВНЫЙ контроль

- ПОЗИТИВ: foton_05 «в августе?» + факт «3-14 августа», хендофф родился в repair-fail → semantic
  covers+same_product → cite-only ответ из факта, route=bot_answer_self.
- ПОЗИТИВ: критик недоступен (semantic_available=False) + факт покрывает → cite-only ответ, не хендофф.
- НЕГАТИВ (критично): вопрос про СМЕНУ + факт про регулярный курс → same_product=false → хендофф остаётся.
- НЕГАТИВ: P0 → recover не входит (is_p0); нет факта → хендофф; semantic_match_fn=None → хендофф как сейчас.
- НЕГАТИВ: composer не собрал валидный cite-only → хендофф (не выдумывать).

## Замер (ВСЕГДА --parallel 4)

ПРАВИЛО (Дмитрий 31.05): все прогоны симулятора --parallel 4, не выше. При высоком parallel критик
timeout-ит → fail-close маскирует реальную картину (на v2B parallel 8 дал 6 ложных хендоффов от
недоступности критика). 4 = чистый замер.
- Кодекс гонит pytest+smoke сам. Клод 1 проверяет.
- Перепрогон v2 + semantic ON, --parallel 4, нить+батч: semantic_recover/replaced > 0; foton_05/unpk_02
  отвечают из факта; hard_gate (выдумки) = 0 (смена≠курс не подставлен); over-handoff упал.

## Ограничения

- Не трогать P0/бренд/факт-границы. cite-only composer строго из retrieved-фактов (не выдумка).
- Правило #1: точные строки return хендоффа (fail-close 1648, repair-fail ~1685), сигнатуры
  _semantic_match/_verified_empty_handoff_replacement, поля DialogueContractPipelineResult — подтвердить
  чтением.
- Отдельный коммит. Это правка в доказанное место — критерий успеха: semantic_recover replaced > 0.
