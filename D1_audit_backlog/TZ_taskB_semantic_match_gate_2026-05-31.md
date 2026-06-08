# ТЗ MAIN — вариант B: модельный semantic-match гейт против over-handoff. 2026-05-31.

Автор: Клод 1. Точки проверены чтением 9e874ccf. Это правильное решение over-handoff после провала
детерминированной задачи A (гейт мёртв из-за хромого детектора хендоффа; key-coverage не отличает
смену от курса). B доказанно необходим.

## ШАГ 0 — откат understanding-калибровки (СНАЧАЛА, отдельный коммит)

understanding-калибровка (коммит 5758e1fa) вернула выдумки (батч fabrication 0→2: бот отвечает соседним
продуктом — регулярный курс вместо смены). Откатить: `git revert 5758e1fa` ИЛИ вернуть инструкцию в
`build_understanding_prompt` (строка 342) к оригиналу «Если факта нет или уверенность низкая →
answerability=manager_only». Гейт 31a73976 НЕ трогать (0 срабатываний, безвреден). Прогнать тесты,
убедиться, что understanding-тесты вернулись к ожиданию manager_only при сомнении.

## Принцип B (почему так)

Детерминированные проверки провалились на двух вещах, которые умеет только модель:
1. «бот собрался уйти к менеджеру» — детектор по словам промахивается;
2. «факт отвечает на вопрос по смыслу И это ТОТ продукт» (смена ≠ регулярный курс; «август» = «3-14
   августа») — ключи-покрытие не отличает.
Решение: один модельный вызов **gpt-5.5, medium reasoning** (это задача на ПОНИМАНИЕ смысла, а не
извлечение — поэтому НЕ мелкая модель: качество semantic-match определяет «выдумка vs верный ответ»,
экономить тут нельзя), который смотрит вопрос + факты + черновик и решает, ушёл ли бот ЗРЯ. Вызов
редкий — только когда есть подозрение на хендофф при фактах, поэтому medium-цена приемлема.

## ШАГ 1 — параметр semantic_match_fn в run_pipeline

`run_pipeline` (pipeline.py:1226) уже принимает understand_fn/draft_fn/repair_fn/faithfulness_fn/
warmth_fn. Добавить так же:
```python
    semantic_match_fn: Callable[[str], object] | None = None,
```

## ШАГ 2 — триггер + вызов + действие

Точка: в run_pipeline ПОСЛЕ формирования draft+route, перед финализацией результата (рядом с гейтом
31a73976 / pure_handoff ~1841). Логика:
```python
    # B-гейт: бот, похоже, уходит к менеджеру, но по теме есть факты — спросить модель, не зря ли.
    looks_handoff = _looks_like_handoff(draft) or route in ("draft_for_manager", "manager_only")
    if (semantic_match_fn is not None
            and looks_handoff
            and contract.answerability == "answer_self"
            and not contract.is_p0
            and retrieval.facts):
        verdict = _semantic_match(semantic_match_fn, contract=contract, retrieval=retrieval, client_words=client_words)
        # verdict: {"covers": bool, "same_product": bool, "answer": "<краткий ответ из факта или ''>"}
        if verdict.get("covers") and verdict.get("same_product"):
            composed = _verified_empty_handoff_replacement(
                draft, contract=contract, retrieval=retrieval, client_words=client_words,
                faithfulness_fn=faithfulness_fn, toggles=toggles, context=context)
            if composed:
                trace_event(context, "semantic_match_gate", {"replaced": True})
                draft = composed   # ответ из факта; route → bot_answer_self
        # covers=False ИЛИ same_product=False → хендофф остаётся (бот ушёл правильно)
```
`_looks_like_handoff` — широкий грубый детектор (включая «передам менеджеру», «уточню у менеджера»,
«спасибо за сообщение», «не могу точно ответить»). НЕ полагаться на узкий `_is_pure_handoff_text` (он
и промахнулся в A). Грубый детектор только СУЖАЕТ круг вызовов модели — финальное решение за моделью.

## ШАГ 3 — промпт semantic-match (gpt-5.5, medium reasoning)

```python
def build_semantic_match_prompt(*, question: str, facts: Mapping[str, str], draft: str) -> str:
    facts_block = "\n".join(f"- {v}" for v in facts.values())
    return (
        "Клиент спросил: " + question + "\n"
        "У нас есть подтверждённые факты:\n" + facts_block + "\n"
        "Черновик ответа бота: " + draft + "\n"
        "Вопрос: отвечают ли эти факты на вопрос клиента ПО СМЫСЛУ, и про ТОТ ЖЕ продукт/тему?\n"
        "Правила: «олимпиадная подготовка Физтех» = ответ на «олимпиада по физике» (covers=true). "
        "«в августе» покрывается фактом «3-14 августа» (covers=true). "
        "Но летняя СМЕНА/ЛАГЕРЬ ≠ обычный регулярный курс: если спросили про смену, а факт про "
        "регулярный курс — same_product=false. Другой предмет/способ оплаты/формат — same_product=false.\n"
        "Верни строго JSON: {\"covers\": true|false, \"same_product\": true|false}.\n"
    )
```

## ШАГ 4 — подключить semantic_match_fn в боевом пути + раннере

- В `subscription_llm._build_dialogue_contract_pipeline_draft` (где вызывается run_pipeline ~1160)
  передать `semantic_match_fn=self._dialogue_contract_semantic_match_runner` (по образцу
  faithfulness_runner 1167/1248). Runner — `CodexJsonModel(model="gpt-5.5", reasoning_effort="medium")`.
- В симуляторе `run_telegram_dynamic_client_sim`: добавить флаг подключения semantic-модели (как
  --memory-mode), модель gpt-5.5 / medium (аргументы --semantic-mode/--semantic-model/--semantic-reasoning,
  дефолт medium).

## Тесты + НЕГАТИВНЫЙ контроль (критично)

- ПОЗИТИВ: «олимпиада по физике?» + факт «Физтех 9/11» + draft=хендофф → semantic covers+same_product →
  ответ из факта, route=bot_answer_self.
- ПОЗИТИВ: «в августе?» + факт «3-14 августа» → ответ.
- НЕГАТИВ (главное): вопрос про СМЕНУ + факт про регулярный курс → same_product=false → хендофф остаётся
  (НЕ подставлять курс — это и была выдумка A).
- НЕГАТИВ: другой предмет/способ/формат → same_product=false → хендофф.
- НЕГАТИВ: P0 → гейт не входит; semantic_match_fn=None → поведение как сейчас (хендофф).

## Замер

- Кодекс гонит pytest+smoke (с мок semantic_match_fn) сам. Клод 1 проверяет.
- Перепрогон v2 с semantic-моделью ON: over-handoff (foton_05/unpk_02 должны ответить из факта);
  hard_gate (выдумки) = 0 (смена≠курс не подставляется); semantic_match_gate срабатывания > 0.

## Ограничения

- +1 вызов модели (gpt-5.5 medium), но РЕДКИЙ (только при подозрении на хендофф при фактах).
- Не трогать P0/бренд/факт-границы. composer уже faithfulness-verified — двойная защита.
- Правило #1: точку финализации draft/route, сигнатуры run_pipeline/composer/faithfulness_runner —
  подтвердить чтением.
