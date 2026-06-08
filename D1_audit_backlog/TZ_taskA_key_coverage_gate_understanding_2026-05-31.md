# ТЗ MAIN — задача A: гейт покрытия ключей + understanding-калибровка (over-handoff). 2026-05-31.

Автор: Клод 1. Точки проверены чтением 9e874ccf. Цель — закрыть остаток over-handoff надёжно
(детерминированно), а не промпт-инструкцией (она стохастит: модель то связывает «август»=«3-14 августа»,
то нет). Две парные правки, отдельные коммиты.

## Проблема (по сырью v2cal)

Калибровка draft помогла частично (хендоффы 47→38%, FAIL 2→0), но походово over-handoff остался:
foton_05 t2 («в августе?» + факт «3-14 августа» в retrieved) и unpk_02 («олимпиада по физике?» + факт
«Физтех 9/11») всё равно «передам менеджеру». Два корня:
1. draft (модель) выдаёт хендофф при покрытых фактах (route=bot_answer_self, текст хендофф) — стохастика.
2. understanding ставит answerability=manager_only «при низкой уверенности» (route=manager до розыска).

## КОММИТ 1 — гейт покрытия ключей (детерминированный, надёжный)

Идея: понимание уже связало вопрос→needed_fact_keys, розыск нашёл факты по ним (matched_keys/facts).
Значит факт ПОКРЫВАЕТ вопрос. Если при этом draft = чистый хендофф и не P0 — это ЛОЖНЫЙ over-handoff:
надо ответить из факта. Механизм сборки ответа УЖЕ есть — `_verified_empty_handoff_replacement` (2070,
собирает из retrieval.facts + проверяет критиком); сейчас он зовётся только при пустом/ошибочном draft.

Точка: `run_pipeline`, ПОСЛЕ того как `draft_fn` вернул `draft` и ДО финализации/`_hard_check`
(найти чтением — рядом со строками, где draft принимается; есть `_is_pure_handoff_text` 2495).

Хелпер покрытия:
```python
def _key_coverage_ok(contract: AnswerContract, retrieval: RetrievalResult) -> bool:
    needed = contract.all_needed_fact_keys()
    if not needed:
        return False
    return any(
        any(mk in retrieval.facts for mk in retrieval.matched_keys.get(key, ()))
        for key in needed
    )
```

Гейт:
```python
    # Гейт покрытия: модель ушла в хендофф, но факты по запрошенным ключам найдены → ответить из факта.
    if (_is_pure_handoff_text(draft)
            and contract.answerability == "answer_self"
            and not contract.is_p0
            and _key_coverage_ok(contract, retrieval)):
        covered = _verified_empty_handoff_replacement(
            draft, contract=contract, retrieval=retrieval, client_words=client_words,
            faithfulness_fn=faithfulness_fn, toggles=toggles, context=context)
        if covered:
            trace_event(context, "key_coverage_gate", {"replaced": True})
            draft = covered   # ответ из фактов вместо ложного хендоффа; route остаётся bot_answer_self
```

Двойная защита от выдумок: (1) `_key_coverage_ok` — нужный ключ реально найден в facts; (2)
`_verified_empty_handoff_replacement` собирает ТОЛЬКО из retrieval.facts и проверяет критиком (faithful).
Если факт чужой/не покрывает — `_key_coverage_ok`=False ИЛИ composer вернёт "" → хендофф остаётся.

Тесты:
- покрытие: «олимпиада по физике?» + retrieved содержит факт олимпиады + draft=хендофф → заменён на
  ответ из факта, route=bot_answer_self.
- NEG: факт чужой (нужный ключ НЕ в facts) → `_key_coverage_ok`=False → хендофф остаётся.
- NEG: P0 (is_p0) → гейт не трогает.
- NEG: composer не собрал валидный ответ → хендофф остаётся (не выдумывать).

## КОММИТ 2 — understanding-калибровка (manager_only только по делу)

Точка: `build_understanding_prompt` (pipeline.py:342). Заменить инструкцию:
```
-  "- Если факта нет или уверенность низкая: answerability=manager_only, но current_question всё равно заполни.\n"
+  "- answerability=manager_only ТОЛЬКО если: по теме реально нет факта, это P0 (возврат/жалоба/спор "
+  "оплаты/юр-угроза), или вопрос вне сферы учебного центра.\n"
+  "- НИЗКАЯ уверенность в ФОРМУЛИРОВКЕ — НЕ повод для manager_only: ставь answer_self, заполни "
+  "current_question и needed_fact_keys, пусть розыск проверит наличие факта. Неуверенность отражай в "
+  "поле confidence, а не в answerability.\n"
```

Тесты:
- «олимпиада по физике есть?» (формулировка ≠ «Физтех») → answerability=answer_self (не manager).
- NEG: «верните деньги» / жалоба → answerability=manager_only (P0 держится).
- NEG: вопрос вне сферы («почини айфон») → manager_only/off-topic.

## Замер

- Кодекс гонит pytest+smoke сам перед каждым коммитом (особенно NEG — выдумки 0).
- Клод 1 проверяет каждый коммит в песочнике.
- Перепрогон v2 (флаг ON), нить+батч на 2 мака: over-handoff (хендоффы по сырью, FAIL «факт был, ушёл»)
  ДОЛЖЕН упасть заметно; hard_gate (выдумки) ДОЛЖЕН остаться 0. Если выдумки выросли — корень в
  composer/coverage, откат гейта.

## Ограничения

- Гейт НЕ трогает P0/бренд/факт-границы; composer уже faithfulness-verified.
- understanding-калибровка не ослабляет P0 (только убирает «низкая уверенность→manager»).
- Правило #1: точку вставки гейта (где draft принимается), сигнатуры `_verified_empty_handoff_replacement`,
  `_is_pure_handoff_text`, поле строки 342 — подтвердить чтением.
- Это детерминированный фундамент (вариант A). Модельный semantic-match (B) — следующим слоем, если
  перепрогон покажет непокрытый остаток.
