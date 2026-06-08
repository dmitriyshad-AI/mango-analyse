# ТЗ MAIN — память диалога: полное единое ТЗ (гард пустых фактов + шаги 1a-1b-2-3-4). 2026-05-30.

Автор: Клод 1. Все точки проверены чтением кода 20956e8e. Полуфабрикаты — Кодексу проверить и
перенести. Реализовать ПОСЛЕ мержа батча 5.1+5.2 (тот же боевой путь). Решение Дмитрия 30.05: полная
глубина (как ChatGPT), одно ТЗ, реализация коммитами по шагам.

## Почему это (регрейд батча 5.1+5.2 по сырью, 30.05)

Все 11 выдумок прогона имеют `retrieved_facts=ПУСТО` и `faithfulness=None` (критик не вызывается без
фактов). Механика: розыск фактов возвращает пусто (вопрос эллиптичный, тема не достроена) → критик не
запускается → бот всё равно генерит (выбирает «онлайн», «олимпиада 9/11», «в будни», тестирование
вместо цены). Значит калибровка критика (правки 5/5.1) на эти кейсы не распространяется — корень в
розыске фактов не той темы и в том, что бот выдумывает при пустых фактах. Лечит: гард пустых фактов
(коммит 0) + память, особенно достройка вопроса темой (коммит 2 / шаг 1b).

## Боевой путь (подтверждено)

`subscription_llm.build_draft` → `_build_dialogue_contract_pipeline_draft` (subscription_llm.py:1154) →
`run_dialogue_contract_pipeline` (pipeline.py). Промпт черновика — `pipeline.build_draft_prompt`
(pipeline.py:521), память `dialogue_memory_view` лежит в `context`, но в промпт не попадает. Понимание
`build_understanding_prompt` (pipeline.py:300) уже видит conversation+known_slots. `decide_route`
(subscription_llm.py:6578) уже получает context. Наполнение памяти — `update_dialogue_memory_after_answer`
(dialogue_memory.py:323), вызывается ПОСЛЕ ответа (раннер: run_telegram_dynamic_client_sim.py:690).

## Правила реализации

- 6 ОТДЕЛЬНЫХ коммитов по порядку: 0 → 1a → 1b → 2 → 3 → 4. Каждый самодостаточен.
- Тесты НЕ гонять (лимит) — после каждого коммита Клод 1 гоняет pytest+smoke в песочнике (ловит
  регрессию до конкретного шага). В main не мержить до зелёного.
- Один M1-прогон в конце: набор нити (`thread_keeping_memory_set_20260530.jsonl`) + контроль на
  батч-наборе. Плохо → откат конкретного коммита-шага.
- Правило #1: каждую точку и имена подтвердить чтением. Каждый защитный шаг — с НЕГАТИВНЫМ контролем
  (не ослабить P0/бренд/факт-гарды и 4 типа правки 5.1).

---

## КОММИТ 0 — гард пустых фактов: бот не выдумывает, когда фактов нет

Точка: `run_dialogue_contract_pipeline` (pipeline.py), сразу ПОСЛЕ блока `force_draft_for_manager`
(~строка 1063, перед `if draft_fn is None:`). Проблема: `force_draft_for_manager` срабатывает только при
`contract.answerability != "answer_self"`; на выдумках понимание ставит `answer_self`, факты пусты —
бот проскакивает в генерацию. Добавить параллельный гард для `answer_self`:

```python
    # Гард пустых фактов: при answer_self нельзя выдумывать, если фактов под фактологический вопрос нет.
    needs_facts = bool(contract.all_needed_fact_keys())
    empty_factual_answer_self = (
        contract.answerability == "answer_self"
        and needs_facts
        and not retrieval.facts
        and not exact_answer_available
        and not _has_retrieved_self_answer_part(contract, retrieval)
    )
    if empty_factual_answer_self:
        fallback = _safe_fallback_text(contract, facts=retrieval.facts, context=context)
        trace_event(context, "build_draft", {"route": "bot_answer_self", "fallback_reason": "empty_facts_no_fabrication"})
        return DialogueContractPipelineResult(
            draft_text=_avoid_repeating_text(fallback, conversation=conversation, contract=contract, facts=retrieval.facts),
            route="bot_answer_self",
            manager_only=False,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            fallback_reason="empty_facts_no_fabrication",
        )
```

ВАЖНО про route (исправлено после прогона коммита 0): route = `bot_answer_self`, НЕ `draft_for_manager`.
Гард перехватывает ДО `draft_fn` (модель не выдумает), отдаёт честный safe_fallback, но отвечает сам бот
(«уточню у менеджера») — выдумки нет, автономия сохранена. `draft_for_manager` здесь сломал бы 2 теста
(narrow_handoff, coverage_noop) и лишне поднял over-handoff. К этой точке `_single_missing_slot_question`
уже обработан выше (~998), уточняющий вопрос сюда не дойдёт.

Тесты:
- answer_self + needed_fact_keys непусто + retrieval.facts пусто → route=bot_answer_self, текст =
  safe_fallback (честный «уточню у менеджера»), draft_fn НЕ вызван, выдумки нет.
- НЕГАТИВНЫЙ КОНТРОЛЬ: retrieval.facts НЕ пусто → бот отвечает как раньше (гард не перехватывает);
  болтовня без needed_fact_keys → не перехватывается; уточняющий slot-вопрос → bot_answer_self как раньше.

Эффект: убирает выдумки про формат/смену/расписание сразу (бот честно «уточню» вместо «Это онлайн»),
БЕЗ роста over-handoff (route остаётся self). Память коммита 1b дальше заменит «уточню» на реальный
ответ, найдя факты по теме.

---

## КОММИТ 1 — шаг 1a: подать готовую память в боевой промпт черновика

Подробно — `TZ_memory_wave1_step1a_feed_memory_to_prompt_2026-05-30.md`. Кратко: добавить параметр
`dialogue_memory_view` в `build_draft_prompt` (pipeline.py:521) + хелпер `_format_memory_block`
(формулировки из `draft_prompt_builder.py:210-216`: open_question, known_slots, do_not_ask_again,
commitments, с пометкой «P0/бренд/факт-гарды важнее») + прокинуть из `context` в месте вызова (~1101):

```python
        prompt = build_draft_prompt(
            conversation=conversation, contract=contract, facts=retrieval.facts,
            missing=retrieval.missing, tone_guide=tone_guide, style_examples=style_examples,
            toggles=toggles,
            dialogue_memory_view=(context or {}).get("dialogue_memory_view"),   # NEW
        )
```

Тест: с памятью → блок в промпте; без памяти → как раньше (NEG).

---

## КОММИТ 2 — шаг 1b: достройка вопроса темой перед розыском фактов (ядро лечения выдумок)

Понимание (`build_understanding_prompt`, pipeline.py:300) уже видит `conversation` и `known_slots`, но
не `topic_focus` и слабо инструктировано на эллипсис. `topic_focus` (dialogue_memory.py:904) содержит
brand, grade, subject, format, goal, product, city, location, question_kind, product_family
(camp/regular_course) — идеально для достройки.

Правка 2.1 — в `build_understanding_prompt` подать `topic_focus` (рядом с known_slots, ~310) и
инструкцию:
```
"- Если реплика — уточнение/эллипсис (короткий вопрос про класс/формат/цену/срок без названия предмета
  или продукта), ВОССТАНОВИ тему из истории, known_slots и topic_focus: в current_question и
  needed_fact_keys укажи ПОЛНУЮ тему (предмет+формат+класс+продукт), а не только новую деталь.\n"
"- product_family из topic_focus важен: если тема была 'camp' (лагерь/смена), уточнение остаётся про
  смену, НЕ подменяй обычным курсом или олимпиадой.\n"
"- Если клиент ЯВНО назвал другой предмет/продукт — это switched_topics, НЕ склеивай со старой темой.\n"
```

Правка 2.2 — детерминированная страховка после `parse_contract`, перед `retrieve_facts`. Если вопрос
эллиптичный (нет темы в current_question) а в памяти есть topic_focus.subject — дополнить
`needed_fact_keys`/`current_question` темой:
```python
def _augment_contract_with_memory_topic(contract: AnswerContract, *, context) -> AnswerContract:
    memory = (context or {}).get("dialogue_memory_view") or {}
    focus = memory.get("topic_focus") or {}
    subject = focus.get("subject")
    if not subject or _contract_has_topic(contract):   # тема уже есть — не трогаем
        return contract
    # дополнить current_question и needed_fact_keys полем темы (subject [+ format + grade + product_family])
    enriched_q = _compose_topic_question(contract.current_question, focus)
    enriched_keys = tuple(dict.fromkeys((*contract.all_needed_fact_keys(), *_keys_for_topic(focus))))
    return replace_contract_topic(contract, current_question=enriched_q, needed_fact_keys=enriched_keys)
```
(имена `_contract_has_topic`, `_compose_topic_question`, `_keys_for_topic`, `replace_contract_topic` —
реализовать; ключи темы брать из каталога fact_key, как делает understanding.)

Вызов: в `build_contract`/перед `retrieve_facts` в run_pipeline, только если understand не достроил сам.

Критерий: «информатика 10» на t1 → «а онлайн?» на t2 находит факт информатики онлайн 10.
NEG: явная смена («а по физике?») НЕ склеивается; product_family=camp не подменяется курсом; одиночный
вопрос без истории — как раньше.

---

## КОММИТ 3 — шаг 2: маршрут и критик читают память

2a — `decide_route` (subscription_llm.py:6578) уже получает `context`. Добавить: повторный уточняющий
ход по уже безопасно отвеченной теме (по `route_history`/`answered_questions` из `dialogue_memory_view`)
— не уходить к менеджеру при наличии факта (понизить лишний veto). НЕ трогать P0-вето
(force_manager_only/high_risk/unknown_brand — жёсткие).
NEG: P0/смена на жалобу → manager_only по-прежнему.

2b — критик `build_faithfulness_prompt` (pipeline.py:575) получает только `draft, facts, client_words`.
Добавить необязательный `established_topic` (из topic_focus/known_slots) + инструкцию: «если класс/формат
уточняют УЖЕ установленную тему (тот же предмет/продукт), не помечай wrong_scope только из-за смены
класса/формата — проверяй по факту той же темы». Прокинуть из context в месте вызова критика
(`_dialogue_contract_faithfulness_runner` → faithfulness_fn).
NEG (обязателен): реальный wrong_scope (лагерь vs курс, другой предмет) и contradicted (онлайн vs очно)
по-прежнему ловятся — контрольные тесты, что 4 типа правки 5.1 НЕ ослаблены.

---

## КОММИТ 4 — шаг 3: модельное наполнение памяти (low effort, мелкая модель, ПОСЛЕ ответа)

Точка — `update_dialogue_memory_after_answer` (dialogue_memory.py:323), вызывается ПОСЛЕ ответа
(раннер 690) → готовит память для СЛЕДУЮЩЕГО хода, латентность текущего ответа не растёт.

Заменить/дополнить regex-извлечение (`_extract_slots_from_text` 499, `_detect_open_question` 546,
`_topic_focus` 904, `_detect_commitments` 573) модельным шагом
`update_memory_llm(recent_turns, prev_memory) -> {slots, topic, open_question, commitments, summary}`:
- ОТДЕЛЬНЫЙ лёгкий вызов: low reasoning effort; по возможности мелкая/быстрая модель (haiku-класс),
  НЕ основная модель ответа;
- regex остаётся FALLBACK, если модельный вызов недоступен/упал (надёжность);
- промпт: «извлеки из последних реплик предмет/класс/формат/продукт, открытый вопрос, что бот обещал;
  обнови, не выдумывай; строгий JSON» — детали по образцу understanding-промптов.
Критерий: на перефразировках (где regex промахивался) слоты/тема извлекаются верно.
NEG (обязателен): модельный шаг НЕ меняет active_brand (бренд задаётся каналом — см.
`test_dialogue_memory_never_changes_active_brand_from_client_text`).

---

## КОММИТ 5 — шаг 4: смысловая сводка диалога

`_conversation_summary_short` (dialogue_memory.py:985) — сейчас склейка слотов (grade/subject/format/
product). Заменить на сжатую смысловую сводку (1-2 фразы: кто клиент, что обсуждали, на чём
остановились) — тем же модельным вызовом коммита 4 (поле `summary`). Regex-склейка — fallback.
Критерий: на диалоге 6+ ходов сводка отражает суть без потери ключевого факта/обещания.

---

## Замер (Клод 1, после всех коммитов)

- После КАЖДОГО коммита: pytest pipeline + dialogue_memory + smoke в песочнике (регрессий 0).
- Один M1-прогон: набор нити (14 персон) — доля ходов без потери нити; контроль на батч-наборе
  (выдумки ↓ от коммита 0 и 1b; автономия не должна упасть «по-настоящему» — рост за счёт найденных
  фактов, не выдумок). Регрейд по сырью; просадка → откат коммита-шага.

## Чего НЕ делать

- НЕ выбрасывать `DialogueMemory` и regex (regex = fallback).
- НЕ ослаблять P0/бренд/факт-гарды и 4 типа правки 5.1 ради нити (везде NEG-контроль).
- НЕ ставить модельное наполнение в синхронный путь ответа (только после ответа, для следующего хода).
- НЕ менять лимиты (MAX_TURNS) — вне этой цепочки.

## Порядок и зависимость коммитов

Коммит 0 (гард) самостоятелен и режет выдумки сразу — можно мержить даже отдельно. 1a самостоятелен.
1b — ядро (находит факты, снимает over-handoff, который добавил коммит 0). 2 опирается на topic_focus
(есть). 3-4 — модельное наполнение, делают память качественной для 1a/1b/2. Откат любого коммита не
ломает предыдущие.
