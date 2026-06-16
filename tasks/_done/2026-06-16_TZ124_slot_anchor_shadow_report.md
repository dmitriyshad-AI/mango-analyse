# TZ-124 Slot Anchor Shadow Report

Дата: 2026-06-16  
Ветка: `codex/tz124-slot-anchor`  
База: fresh `origin/main` после TZ-121, commit `ecefee53f2209158143710b9ade04e5292b95858`

## Что сделано

- Добавлен default-off флаг `TELEGRAM_ANCHORED_BARE_GRADE`.
- За флагом добавлен якорный распознаватель «голого класса» `1..11`.
- Класс извлекается только при якоре в короткой фразе: предмет, формат, `класс/кл`, `ОГЭ/ЕГЭ`.
- Жёсткий стоп-лист: телефон, время, возраст, количество, деньги, дата, диапазон, часть года/номера.
- `dialogue_memory.py` пишет новый класс только через provenance-путь: `source=memory_provenance`, с quote клиентской фразы.
- `new_lead_funnel.extract_grade` получил тот же anchored-вариант.
- При `TELEGRAM_ANCHORED_BARE_GRADE=0` поведение остаётся как `main`.
- `dialogue_contract_pipeline.py` не трогался.

## Доп-страховки

- Мульти-предмет в provenance-слое не превращается в единичный `subject` scope.
- Формат `онлайн и очно сравниваю` при включённом флаге остаётся пустым.
- P0 не ослаблен: извлечение слотов не повышает маршрут и не снимает handoff.

## OFF→ON прогон

Команда:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz124_slot_anchor_pack.py \
  --out-dir audits/_inbox/tz124_slot_anchor_pack_20260616 \
  --parallel 4
```

Результат:

- rows: `30`;
- modes: `off=15`, `on=15`;
- `gate_passed=true`;
- `failed_checks=[]`;
- `false_grade_from_number_trap=false`;
- `format_choice_selected=false`;
- `price_under_extracted_class=false`;
- `off_changed_bare_grade=false`;
- `llm_calls_total=0`.

Grep по транскриптам:

- POS `P1/P2/P3/P4/P5` в ON имеют ожидаемый `MEMORY_GRADE`;
- NEG `N1/N2/N3/N4/N7/N8/N9` в ON не извлекают ложный класс;
- `N5` не создаёт единичный subject scope;
- `N6` не выбирает формат;
- `PRICE_MENTIONS` пустые, неверной цены под извлечённый класс нет.

Артефакты для регрейда:

`audits/_inbox/tz124_slot_anchor_pack_20260616/`

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz124_slot_anchor.py

12 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_dialogue_memory.py \
  tests/test_new_lead_funnel.py \
  tests/test_conversation_intent_plan.py \
  tests/test_tz124_slot_anchor.py

112 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3316 passed, 5 skipped, 1 warning
```

## Semantic Review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Новый класс не извлекается из чисел-ловушек.
- OFF остаётся безопасным дефолтом.
- P0 остаётся сильнее извлечения.
- Нет ценовых обещаний или клиентских текстов, которые могли бы назвать неверную стоимость.

Остаточный риск:

- Это parser-level и compact-pack проверка, не полноценный живой Telegram-прогон. Поэтому блок останавливается на регрейд Дмитрия/Claude перед включением флага.

## Стоп

Остановиться на регрейд. Флаг не включать глобально до PASS.
