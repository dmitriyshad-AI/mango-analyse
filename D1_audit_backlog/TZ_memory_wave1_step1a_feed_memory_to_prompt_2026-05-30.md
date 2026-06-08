# ТЗ MAIN — память, волна 1 / шаг 1a: подать готовую память в боевой промпт (полуфабрикат). 2026-05-30.

Автор: Клод 1. Read-only разбор 20956e8e. Реализовать ПОСЛЕ замера и мержа батча 5.1+5.2 (трогает тот
же боевой путь — не смешивать). Решение Дмитрия 30.05: идём в полную глубину памяти (как ChatGPT),
но волнами с замером. Это первая, самая дешёвая волна.

## Зачем (подтверждено чтением кода)

Боевой/eval путь генерации = `subscription_llm.build_draft` → `_build_dialogue_contract_pipeline_draft`
→ `run_dialogue_contract_pipeline`. Промпт черновика — `dialogue_contract_pipeline.build_draft_prompt`
(pipeline.py:521) — принимает только `conversation, contract, facts`. Параметра `dialogue_memory_view`
у него НЕТ — модель видит лишь сырую ленту реплик. При этом раннеры (симулятор и боевой пилотный бот)
УЖЕ строят богатую `DialogueMemory` и кладут `dialogue_memory_view` в context (open_question,
known_slots, do_not_ask_again, recent_turns, last_bot_commitments, topic_focus, conversation_summary).
Её просто никто не передаёт в боевой промпт. Шаг 1a — прокинуть готовую память в промпт. Самый дешёвый
шаг: нет новых вычислений и вызовов модели, память уже посчитана.

Образец, как память подаётся правильно, уже есть — `draft_prompt_builder.py:210-216` (другая, не боевая
ветка). Берём оттуда формулировки.

## Правка 1 — параметр и блок в `build_draft_prompt` (pipeline.py:521)

Добавить необязательный параметр и блок памяти (вставить ПЕРЕД историей диалога, после фактов):

```python
def build_draft_prompt(
    *,
    conversation: Sequence[Mapping[str, str]],
    contract: AnswerContract,
    facts: Mapping[str, str],
    missing: Sequence[str],
    tone_guide: str = "",
    style_examples: Sequence[str] = (),
    toggles: Toggles | None = None,
    dialogue_memory_view: Mapping[str, Any] | None = None,   # NEW
) -> str:
    ...
    memory_block = _format_memory_block(dialogue_memory_view)   # NEW, см. ниже
    ...
    # в сборку промпта добавить memory_block перед "История диалога:"
```

Хелпер (рядом, формулировки — из draft_prompt_builder.py:210-216):

```python
def _format_memory_block(view: Mapping[str, Any] | None) -> str:
    if not view:
        return ""
    import json
    open_q = (view.get("open_question") or {}).get("text") or ""
    known = view.get("known_slots") or {}
    do_not_ask = view.get("do_not_ask_again") or []
    commitments = view.get("last_bot_commitments") or []
    topic = view.get("topic_focus") or {}
    summary = view.get("conversation_summary_short") or ""
    lines = ["Рабочая память переписки (используй, но P0/бренд/факт-гарды важнее):"]
    if summary:    lines.append(f"- кратко: {summary}")
    if topic:      lines.append(f"- фокус темы: {json.dumps(topic, ensure_ascii=False)}")
    if open_q:     lines.append(f"- открытый вопрос клиента (закрой первым, если безопасно): {open_q}")
    if known:      lines.append(f"- уже известно (НЕ переспрашивай): {json.dumps(known, ensure_ascii=False)}")
    if do_not_ask: lines.append(f"- не спрашивай заново: {', '.join(map(str, do_not_ask))}")
    if commitments:lines.append(f"- бот уже обещал (не меняй без факта): {'; '.join(map(str, commitments))}")
    return "\n".join(lines) + "\n\n"
```

## Правка 2 — прокинуть память из context в месте вызова (pipeline.py:~1101)

В `run_dialogue_contract_pipeline`, где вызывается `build_draft_prompt(...)` (около 1101), добавить:

```python
        prompt = build_draft_prompt(
            conversation=conversation,
            contract=contract,
            facts=retrieval.facts,
            missing=retrieval.missing,
            tone_guide=tone_guide,
            style_examples=style_examples,
            toggles=toggles,
            dialogue_memory_view=(context or {}).get("dialogue_memory_view"),   # NEW
        )
```

Подтвердить чтением: точные имена аргументов вызова (retrieval.facts/missing — как в коде) и что
`context` доступен в этой точке (run_dialogue_contract_pipeline принимает context=context).

## Что НЕ трогать в этой волне

- Розыск фактов (эллипсис «а для 10?») — это шаг 1b, СЛЕДУЮЩАЯ волна (память в промпт не чинит подбор
  фактов не той темы; это отдельно).
- Маршрут/критик читают память — шаг 2, следующая волна.
- Модельное наполнение памяти + смысловая сводка — шаги 3-4, после замера 1a+1b+2.
- `draft_prompt_builder` (else-ветка) — не трогать.

## Тесты

- `build_draft_prompt` с `dialogue_memory_view` → промпт содержит open_question, known_slots,
  do_not_ask_again, commitments.
- Без памяти (None/пусто) → промпт как раньше (блок пуст), НИЧЕГО не сломано — НЕГАТИВНЫЙ КОНТРОЛЬ.
- Память не перебивает гарды: текст инструкции явно «P0/бренд/факт-гарды важнее» (как в образце).

## Замер (Клод 1, после реализации)

- pytest pipeline + smoke в песочнике (регрессий 0).
- Прогон набора удержания нити (`thread_keeping_memory_set_20260530.jsonl`, 14 персон) ДО и ПОСЛЕ 1a:
  сравнить долю ходов без потери нити (особенно не-переспрашивание известного, удержание темы).
- Контроль: автономия и выдумки на батч-наборе не просели (память в промпт не должна ломать факты).

## Ограничения

- Тесты не гонять (лимит) — Клод 1 прогонит. В main не мержить до зелёного и до замера батча 5.1+5.2.
- Правило #1: имена аргументов и точку вызова подтвердить чтением.
- Хирургично: 1 параметр + 1 хелпер + 1 проброс. Не расширять на 1b/2 в этой волне.
