# ТЗ MAIN (Кодекс) — Решение A: cite-only восстановление во ВСЕХ точках рождения ухода. 2026-05-31.

Автор: Клод 1. Точки подтверждены чтением HEAD aef24d36 (`dialogue_contract_pipeline.py`). Это
СТРУКТУРНАЯ правка горячего пути «ответить/уйти» — самый большой рычаг из полного разбора 412 диалогов
(`AUDIT_full_both_runs_deep_roots_2026-05-31.md`). Отдельный коммит. Правило #1: всё подтвердить чтением.

## Что чиним (по сырью)

400 ходов (~33% всех) — открывашка-уход; **в 348 (87%) у бота на руках были retrieved-факты**. То есть
слой проверки душит уже найденный ответ. Симптомы: «где вы находитесь?» → 3× «Спасибо за сообщение»
(адрес В БАЗЕ); маткапитал/документооборот/индив.занятия/уровень — факт есть, бот уходит.

## Почему так (корень в коде, прочитано)

Восстановление уже есть, но УЗКОЕ, по двум причинам:
1. **Источник дословный по темам.** `_verified_empty_handoff_replacement` (1706) собирает замену из
   `_composition_answer` (1693, только n-предметов/смена/цена-формат/рассрочка) → `_hard_failure_exact_fact_fallback`
   (только цена/формат/запись) → `_coverage_cite_only_answer` (1669, общий cite, НО гейтится сверху).
   Для адреса/маткапитала/документооборота спец-композера нет.
2. **Гейт уважает осторожный флаг понимания.** `_should_replace_empty_handoff` требует
   `answerability == "answer_self"` И `_has_exact_retrieved_answer_part`. Если понимание из осторожности
   пометило подвопрос `answerable="manager"` (адрес!), восстановление не входит. А расширять понимание
   НЕЛЬЗЯ — это приносило выдумки (Task A откатывали).

Плюс точки `contract_manager_only` (1046) и `hard_verification_failed` (1232) возвращают уход вообще
без попытки cite-only (1232 пробует только узкий `verified_fallback`, гейтнутый
`_can_autonomously_replace_failed_draft` = «только fact_grounding»).

## Идея решения (безопасная замена дословного на семантическое)

Не трогаем понимание. Вводим ОДИН строгий хелпер cite-only-восстановления, применяемый во ВСЕХ точках
рождения ухода. Безопасность держится на ДВУХ замках, а не на осторожности понимания:
- **Замок 1 — scope:** ответ строится только если `_has_exact_retrieved_answer_part` (ключ совпал с
  ТЕКУЩИМ подвопросом и его scope через `_retrieved_keys_match_question_scope`). Это исключает дрейф на
  смежный факт (смена≠курс, предоплата≠рассрочка).
- **Замок 2 — валидация выхода:** собранный cite-only ПЕРЕПРОВЕРЯЕТСЯ через `_hard_check` (бренд/мета/
  числа/p0/faithfulness). Уходит клиенту только если чисто.

Этого достаточно: scope-замок + перепроверка выхода делают ненужным осторожный флаг понимания. Старая
understanding-калибровка падала именно потому, что НЕ имела этих двух замков.

## Правки

### Правка 1 — единый хелпер
Рядом с `_verified_empty_handoff_replacement` (1706):
```python
def _cite_only_recover_before_handoff(
    *, contract, retrieval, draft, client_words, faithfulness_fn, toggles, context,
) -> str:
    # P0/возврат/жалоба/спор — НИКОГДА не восстанавливаем
    if contract.is_p0 or _asks_refund_policy(contract):   # + complaint, если есть отдельный признак
        return ""
    # Замок 1: точное scope-совпадение факта с текущим подвопросом
    if not _has_exact_retrieved_answer_part(contract, retrieval):
        return ""
    candidate = (
        _composition_answer(contract, retrieval, current_draft=draft)
        or _hard_failure_exact_fact_fallback(contract, retrieval)
        or _coverage_cite_only_answer(contract, retrieval)   # общий cite client_safe — закрывает адрес и т.п.
    )
    if not candidate:
        return ""
    # Замок 2: перепроверка выхода
    cand_facts = _facts_with_derived_answer(retrieval.facts, candidate)
    findings, unsupported, semantic_available = _hard_check(
        candidate, facts=cand_facts, contract=contract, client_words=client_words,
        faithfulness_fn=faithfulness_fn, toggles=toggles, context=context)
    if findings or unsupported:
        return ""
    # если критик недоступен (fail-close) — cite-only из client_safe всё равно безопасен:
    # детерминированные проверки (бренд/мета/числа/p0) ВНУТРИ _hard_check уже отработали;
    # faithfulness не требуется, т.к. текст дословно из retrieved client_safe.
    return candidate
```
Подтвердить чтением: `_hard_check` при `faithfulness_fn=None` возвращает `semantic_available=False`, но
детерминированные findings всё равно считает (если нет — добавить отдельный детерминированный про-ход
для fail-close, без faithfulness). НЕ отдавать candidate, если сработал хоть один детерминированный finding.

### Правка 2 — применить хелпер во всех точках рождения ухода (ПЕРЕД возвратом safe_fallback)
Перечень точек (подтвердить строки):
- `contract_manager_only` — строка 1046.
- `hard_verification_failed` — строка 1232.
- `semantic_check_unavailable` (fail-close) — строки 1151, 1188, 1290 (faithfulness_fn=None).
- `draft_error` (1128), `no_draft_fn` (1077) — по возможности.
Шаблон в каждой точке:
```python
    recovered = _cite_only_recover_before_handoff(
        contract=contract, retrieval=retrieval, draft=draft, client_words=client_words,
        faithfulness_fn=faithfulness_fn,  # None там, где критик недоступен
        toggles=toggles, context=context)
    if recovered:
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(recovered, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="bot_answer_self", manager_only=False, contract=contract,
            facts=retrieval.facts, missing=retrieval.missing,
            fallback_reason="cite_only_recover", repaired=True)
    # иначе — текущий safe_fallback (как сейчас)
```

### Правка 3 (минимальная) — не дублировать логику
Существующий `_verified_empty_handoff_replacement` пересекается с новым хелпером. Не удалять, но новый
хелпер сделать единой точкой; если просто — переключить `_verified_empty_handoff_replacement` на вызов
нового (источник+валидация общие), оставив его сигнатуру. «Простота вперёд»: не плодить две ветки.

## Тесты + НЕГАТИВНЫЙ контроль (без него структурная правка не закрыта)

ПОЗИТИВ:
- «где вы находитесь?» + факт `locations_foton.addresses` scope-совпал → cite-only адрес,
  route=bot_answer_self (воспроизводит V3_address_02/03, сейчас 3× уход).
- маткапитал «можно?» + `matkap.client_safe_text` → cite-only ответ (unpk_docs_03).
- «нужно подъехать в корпус?» + `electronic_document_flow` → cite-only «всё электронно» (unpk_program_08).
- hard_verification_failed, где факт scope-совпал → cite-only вместо ухода.
- fail-close (faithfulness_fn=None) + client_safe факт scope-совпал → cite-only (детерминир. проверки чисты).

НЕГАТИВ (критично):
- P0/жалоба/возврат → хелпер возвращает "" на первом замке, уход остаётся (НЕ ослаблять).
- смена≠курс: вопрос про смену + факт про регулярный курс → `_has_exact_retrieved_answer_part`=False → уход.
- предоплата vs рассрочка (дрейф аспекта): ключ подвопроса «предоплата» не совпал с фактом рассрочки →
  scope-замок не пускает → уход/уточнение, НЕ выдаёт рассрочку.
- чужой бренд в cite-only → `_hard_check` brand_leak → "" → уход.
- число вне факта в собранном тексте → fact_grounding → "" → уход.
- understanding пометил manager по БЕЗОПАСНОСТИ (p0) → первый замок (is_p0) держит.

## Замер (ВСЕГДА --parallel 4)

Кодекс гонит pytest+smoke сам. Клод 1 проверяет дифф в песочнике. Критерий выхода:
- юнит: 5 позитивов зелёные, 6 негативов держат (особенно P0 и смена≠курс);
- существующий набор не покраснел; бренд/мета остаются 0.
- M1-прогон позже (в батч): ждём ↓over_handoff (348 «уходов при фактах» должны резко упасть), ↑автономии,
  тон подрастёт частично (полностью — после Решения B про человечный путь ухода). Выдумки НЕ должны
  вырасти (оба замка). `cite_only_recover` replaced > 0 и заметно.

## Ограничения

- Не трогать понимание (understanding-промпт), P0-ветку (946), refund-различение, бренд-гарды.
- cite-only строго из retrieved client_safe фактов; никакой генерации чисел/сущностей.
- Это правка в доказанное место (87% уходов при фактах). Один коммит, с обоими замками и NEG-контролем.
