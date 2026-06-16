# TZ-122 wrong_intent_fact OFF->ON micro measurement

Дата: 2026-06-16
Ветка: `codex/tz122-wrong-intent-fact`
Коммит кода: `d2e6af1`

## Цель

Проверить калибровку `wrong_intent_fact` перед включением флага:

- `TELEGRAM_WRONG_INTENT_FACT_CALIBRATION=0` против `=1`;
- микросет POS/NEG;
- `--parallel 4`;
- флаг в `pilot_gold_v1` не включать.

## Артефакты

- Сценарии: `runs/tz122_wrong_intent_fact_micro_20260616/input/scenarios.jsonl`
- Replay: `runs/tz122_wrong_intent_fact_micro_20260616/input/replay.jsonl`
- Dynamic OFF: `runs/tz122_wrong_intent_fact_micro_20260616/off/`
- Dynamic ON: `runs/tz122_wrong_intent_fact_micro_20260616/on/`
- Controlled gate transcripts: `runs/tz122_wrong_intent_fact_micro_20260616/controlled/controlled_micro_transcripts.jsonl`
- Controlled summary: `runs/tz122_wrong_intent_fact_micro_20260616/controlled/controlled_micro_summary.json`

## Команды

OFF:

```bash
env TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
  TELEGRAM_DIALOGUE_CONTRACT_PIPELINE=1 \
  TELEGRAM_RULES_ENGINE_PLANNER_INTENT=1 \
  TELEGRAM_WRONG_INTENT_FACT_CALIBRATION=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 scripts/run_telegram_dynamic_client_sim.py \
    --scenarios runs/tz122_wrong_intent_fact_micro_20260616/input/scenarios.jsonl \
    --replay-from runs/tz122_wrong_intent_fact_micro_20260616/input/replay.jsonl \
    --snapshot product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json \
    --out-dir runs/tz122_wrong_intent_fact_micro_20260616/off \
    --parallel 4 --brand all \
    --judge-mode fake --memory-mode fake --semantic-mode fake --semantic-verifier-mode fake --selling-compose-fake \
    --model gpt-5.5 --bot-reasoning medium --timeout-sec 240
```

ON: та же команда, но `TELEGRAM_WRONG_INTENT_FACT_CALIBRATION=1` и `--out-dir .../on`.

## Dynamic result

Оба прогона завершились без инфраструктурных ошибок:

| Режим | Диалогов | Ходов | FAIL | Hard-gate fail |
|---|---:|---:|---:|---:|
| OFF | 8 | 8 | 0 | 0 |
| ON | 8 | 8 | 0 | 0 |

Счётчики по трассам:

| Метрика | OFF | ON | Итог |
|---|---:|---:|---|
| `wrong_intent_fact` | 3 | 1 | снизилось |
| `fact_grounding` | 2 | 1 | не выросло |
| `unsupported_promise` | 2 | 1 | не выросло |
| финальный route `draft_for_manager`/`manager_only` | 3 | 3 | over-handoff не вырос |
| gate-route к менеджеру | 7 | 3 | снизилось |

POS:

- `сколько стоит смена в лагере`: `wrong_intent_fact` исчез, но финальный route остался `draft_for_manager` из-за политики/недостающих фактов, не из-за `wrong_intent_fact`.
- `сколько стоит смена`: `wrong_intent_fact` отсутствует в обоих режимах dynamic, финальный route остался `draft_for_manager` из-за политики/недостающих фактов.
- `где очные курсы 7 класс`: OFF давал `wrong_intent_fact`, ON прошёл `gate_action=pass`, финальный route стал `bot_answer_self_for_pilot`.
- `учебный год/даты старта`: ON всё ещё даёт `wrong_intent_fact` с деталем `Адресный факт нельзя выдавать как ответ на неадресный вопрос`.

NEG:

- `цена -> адрес/расписание`: dynamic не принудил модель дать адрес вместо цены; вместо этого сработали `fact_grounding`/`unsupported_promise`, они не выросли.
- `контактные часы как дни занятий`: dynamic дал безопасный уточняющий ответ, без ложной выдачи контактных часов.
- `residential vs city`: dynamic ON дал корректный факт городского лагеря, поэтому NEG-срабатывания не было в живом тексте.
- `выдуманное число без факта`: dynamic ON не выдумал новое неподтверждённое число; `fact_grounding` не вырос.

Вывод по dynamic: флаг снижает ложные `wrong_intent_fact`, но не полностью закрывает ожидаемый POS-набор. Главный остаточный провал: `учебный год/даты старта`.

## Controlled gate result

Контролируемый слой нужен, потому что dynamic не всегда генерирует ошибочный NEG-текст. Здесь проверяется сам страж на заранее заданных черновиках.

Счётчики:

| Группа | Режим | `wrong_intent_fact` | `fact_grounding` | route-proxy manager |
|---|---|---:|---:|---:|
| POS | OFF | 2 | 0 | 2 |
| POS | ON | 0 | 0 | 0 |
| NEG | OFF | 2 | 1 | 3 |
| NEG | ON | 3 | 1 | 4 |

Что подтвердилось:

- POS `где очные курсы 7 класс`: исправлен.
- POS `сколько стоит смена` без явного слова `лагерь`: исправлен.
- NEG `цена -> адрес`: демоут сохранён.
- NEG `контактные часы как дни занятий`: демоут сохранён.
- NEG `residential vs city`: ON добавил правильный демоут, который OFF пропускал.
- NEG `выдуманное число без факта`: `fact_grounding` сохранён, не ослаблен.

Проверка `OFF == origin/main`:

- отдельный worktree от `origin/main` (`dd00d65`) прогнан тем же controlled-набором;
- результат controlled OFF на ветке ТЗ-122 полностью совпал с `origin/main` по `finding_codes` и деталям findings.

## Verdict

`PASS_WITH_BLOCKER_FOR_ENABLE`.

Флаг работает как калибровка адреса/лагеря и не ослабляет NEG-гейты, но включать его в профиль рано:

1. Dynamic POS `учебный год/даты старта` всё ещё понижается из-за адресного `wrong_intent_fact`.
2. Dynamic POS лагеря перестал падать на `wrong_intent_fact`, но финально всё ещё уходит в `draft_for_manager` по другому слою (`policy_permission` / недостающие факты).
3. Dynamic NEG не покрывает все вредные черновики, поэтому controlled gate остаётся обязательной частью приёмки.

## Рекомендация

Не включать `TELEGRAM_WRONG_INTENT_FACT_CALIBRATION` в `pilot_gold_v1` до следующего узкого фикса:

- добавить отдельную калибровку для `academic_year/start_date`, чтобы адресный факт из расписания не превращал ответ про даты старта в `wrong_intent_fact`;
- отдельно разобрать route-policy для лагерных POS, потому что после снятия `wrong_intent_fact` они всё равно остаются менеджерскими черновиками.
