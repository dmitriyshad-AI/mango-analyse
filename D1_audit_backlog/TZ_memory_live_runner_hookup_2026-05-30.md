# ТЗ MAIN — live-подключение memory_llm в раннере (коммиты 3-4 работают в прогоне). 2026-05-30.

Автор: Клод 1. Точки проверены чтением раннера 9e874ccf. Отдельный коммит. Это правка ТОЛЬКО раннера
(`scripts/run_telegram_dynamic_client_sim.py`) — код бота (коммиты 0-4) не трогать.

## Зачем

Коммиты 3-4 (модельное наполнение памяти + смысловая сводка) сделаны как hook через `memory_llm_fn`,
но в раннере live-вызов не подключён — сейчас память наполняется regex. Чтобы прогон измерил 3-4,
нужно подать в `update_dialogue_memory_after_answer` живую модель: отдельную, мелкую/быструю, low effort.

## Точки (проверено)

- Класс модели: `CodexJsonModel(*, model, reasoning_effort, timeout_sec, codex_bin)`, метод
  `.generate(prompt: str) -> Mapping` (run_telegram_dynamic_client_sim.py:104). Это ровно сигнатура
  `memory_llm_fn: Callable[[str], object]`.
- Аргументы моделей: `--model` (gpt-5.5), `--bot-reasoning` medium, `--judge-reasoning` high и т.п.
  (строки 182-188). Режимы codex/fake.
- Вызов наполнения: `update_dialogue_memory_after_answer(...)` (~строка 690) — сейчас БЕЗ `memory_llm_fn`.

## Правка

1. Аргументы (рядом с --bot-reasoning):
```python
    parser.add_argument("--memory-mode", choices=("codex", "fake", "off"), default="codex")
    parser.add_argument("--memory-model", default="gpt-5.5")   # заменить на мелкую/быструю, если доступна
    parser.add_argument("--memory-reasoning", default="low")    # low — обязательно (извлечение, не рассуждение)
```
Если в инфраструктуре есть отдельная мелкая модель (haiku-класс) — указать её дефолтом в `--memory-model`;
если доступна только gpt-5.5 — оставить gpt-5.5, но reasoning ОБЯЗАТЕЛЬНО low (это и даёт экономию).

2. Сборка модели (по образцу build_judge_model/build_bot_model):
```python
def build_memory_model(args):
    if args.memory_mode == "off":
        return None
    if args.memory_mode == "fake":
        return FakeMemoryModel()
    return CodexJsonModel(model=args.memory_model, reasoning_effort=args.memory_reasoning,
                          timeout_sec=args.timeout_sec, codex_bin=args.codex_bin)
```
`FakeMemoryModel.generate` — вернуть разумный payload (slots/topic/open_question/commitments/summary)
для smoke без codex.

3. Передать в вызов наполнения (~690):
```python
        updated_memory = update_dialogue_memory_after_answer(
            context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {},
            answer_text=bot_text,
            route=str(turn.get("bot_route") or ""),
            fact_refs=(),
            safety_flags=tuple(turn.get("bot_safety_flags") or ()),
            memory_llm_fn=(memory_model.generate if memory_model is not None else None),   # NEW
        )
```
`memory_model` создать один раз рядом с judge_model/bot_provider и прокинуть в функцию хода.

## Стоимость (важно)

Это +1 codex-вызов на КАЖДЫЙ ход (после ответа). low reasoning смягчает, но прогон станет заметно
дольше/дороже. Поэтому: `--memory-reasoning low` обязательно; режим `off` оставить для прогонов, где
3-4 мерить не нужно (тогда поведение = коммиты 0-2 на regex). Concurrency — как сейчас (2-3 для codex).

## Проверка (Клод 1)

- smoke с `--memory-mode fake` — память наполняется из FakeMemoryModel, тесты раннера зелёные;
- в codex-режиме memory_model подключён (memory_llm_fn не None в вызове наполнения).

## Ограничения

- Правка ТОЛЬКО раннера; код бота (коммиты 0-4) не трогать.
- Тесты не гонять (лимит) — Клод 1 прогонит smoke в песочнике. Правило #1: точку вызова и сигнатуру
  CodexJsonModel подтвердить чтением. Отчёт: что изменено, как настроены memory-model и low reasoning.
