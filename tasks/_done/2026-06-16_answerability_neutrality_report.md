# 2026-06-16: answerability shadow neutrality check

## Scope

Проверка перед постоянным включением `TELEGRAM_ANSWERABILITY_SHADOW` в профиль `pilot_gold_v1`.

Рабочее дерево: `/Users/dmitrijfabarisov/Projects/Mango_answerability_neutrality`  
Ветка: `codex/answerability-shadow-neutrality`  
База ветки: `bb9bb39`

Основную папку `/Users/dmitrijfabarisov/Projects/Mango analyse` не менял: там есть чужие tracked-изменения TZ119.

## Initial replay before fix

Набор: `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`  
Replay source: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_final_smoke18_predpilot/dynamic_dialog_transcripts.jsonl`

Результат первичной OFF/ON пары:

| metric | value |
| --- | ---: |
| turns_compared | 40 |
| route_same | 38/40 = 95.0% |
| draft_text_same | 2/40 = 5.0% |
| route_and_text_same | 2/40 = 5.0% |
| trace_off_turns | 0 |
| trace_on_turns | 40 |

Вывод: старая реализация была не нейтральна. Причина в коде: при ON блок answerability добавлял поля `can_answer_self`, `self_missing_facts`, `supporting_facts`, `why_manager` прямо в основной direct-path prompt и JSON-схему ответа.

## Fix

Сделано:

- удалено добавление answerability-инструкции и answerability-полей из `_build_direct_path_prompt`;
- direct draft runner больше не парсит answerability-поля из основного JSON ответа;
- `answerability_trace` остаётся post-layer метаданными в `_direct_path_finalize_metadata`;
- добавлен NEG: при `TELEGRAM_ANSWERABILITY_SHADOW=1` direct prompt байт-в-байт совпадает с prompt при `TELEGRAM_ANSWERABILITY_SHADOW=0`.

Важно: это перенос датчика из prompt ответа в post-prompt слой. Модельная самооценка как отдельный LLM-проход здесь не добавлялась; если нужна именно модельная самооценка после ответа, это отдельная следующая правка.

## Replay after fix

OFF run:

`/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260616_answerability_neutrality_postpass_off`

ON run:

`/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260616_answerability_neutrality_postpass_on`

Оба прогона:

| metric | OFF | ON |
| --- | ---: | ---: |
| dialogs | 18 | 18 |
| turns | 40 | 40 |
| fail | 0 | 0 |
| hard_gate_failures | 0 | 0 |
| pass_with_notes | 18 | 18 |
| config_validity.invalid | false | false |
| llm_calls.total | 116 | 116 |
| bot_direct_draft | 38 | 38 |
| bot_semantic_output_verifier | 39 | 39 |
| answerability_trace turns | 0 | 40 |

Машинное сравнение OFF/ON транскриптов после фикса:

| metric | value |
| --- | ---: |
| turns_compared | 40 |
| route_same | 37/40 = 92.5% |
| draft_text_same | 3/40 = 7.5% |
| route_and_text_same | 3/40 = 7.5% |

Интерпретация: после фикса основной prompt одинаковый, но две независимые replay-попытки всё равно не дают byte-for-byte совпадение, потому что direct-path и verifier заново вызывают модель и этот участок не является детерминированным байт-тестом. Это `measurement_bug` исходной методики проверки, а не доказательство влияния answerability после удаления его из prompt.

## Tests

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "answerability or pilot_gold_v1_enables_full_battle_profile_flags"
9 passed, 463 deselected
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3276 passed, 5 skipped, 1 warning
```

## Decision

Старую реализацию с answerability-полями в основном prompt включать нельзя.

Текущая ветка делает безопасную форму: `TELEGRAM_ANSWERABILITY_SHADOW` остаётся в профиле, но не меняет prompt ответа и добавляет только `answerability_trace` post-layer. Для строгого будущего A/B byte-test нужен детерминированный harness: один и тот же замороженный direct draft/verifier result прогонять через OFF/ON post-layers, либо отдельный LLM self-eval pass после финального ответа с явным запретом менять `route`/`draft_text`.
