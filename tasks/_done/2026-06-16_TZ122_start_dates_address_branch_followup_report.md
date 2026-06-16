# TZ-122 start dates address-branch follow-up

Дата: 2026-06-16
Ветка: `codex/tz122-wrong-intent-fact`

## Цель

Довести адресную ветку `wrong_intent_fact`: вопрос про даты старта/учебный год не должен понижаться только потому,
что в выбранном fact pack есть ключи вида `start_by_location`.

Флаг в профиль не включался: `TELEGRAM_WRONG_INTENT_FACT_CALIBRATION` остаётся opt-in.

## Что изменено

1. В `dialogue_contract_pipeline.py` добавлена узкая калибровка:
   - если вопрос явно про `учебный год` / `даты старта`;
   - и факт имеет `start_by_location`;
   - то адресный `wrong_intent_fact` не выставляется.

2. В `policy_routing.py` добавлена узкая автономная разблокировка под тем же флагом:
   - только `theme:013_schedule`;
   - только вопрос про старт учебного года;
   - только при чистом выходном гейте;
   - только если в direct-path фактах есть `academic_year_2026_27.start` / `start_by_location`;
   - общая матрица расписания, P0, бренд, числа и факт-гейтинг не ослаблялись.

## Микро-замер

Артефакты:

- OFF: `runs/tz122_wrong_intent_fact_micro_20260616_fix3/off/`
- ON: `runs/tz122_wrong_intent_fact_micro_20260616_fix3/on/`
- Controlled: `runs/tz122_wrong_intent_fact_micro_20260616_fix3/controlled/`

Оба динамических прогона завершились без инфраструктурных ошибок:

| Режим | Диалогов | Ходов | FAIL | Hard-gate fail |
|---|---:|---:|---:|---:|
| OFF | 8 | 8 | 0 | 0 |
| ON | 8 | 8 | 0 | 0 |

Динамические счётчики:

| Метрика | OFF | ON | Итог |
|---|---:|---:|---|
| `wrong_intent_fact` | 4 | 0 | исправлено |
| `fact_grounding` | 1 | 1 | не выросло |
| `unsupported_promise` | 1 | 1 | не выросло |
| финальный manager-route | 3 | 2 | не вырос |
| gate-route к менеджеру | 7 | 3 | снизилось |

Ключевой POS:

- `tz122_pos_school_year_start`: ON -> `bot_answer_self_for_pilot`, `wrong_intent_fact=0`,
  есть флаг трассы `tz122_academic_year_start_autonomy_allowed`.

Контролируемый слой:

- POS: ON -> 4/4 `bot_answer_self_for_pilot`.
- NEG: адрес вместо цены, контактные часы как дни занятий, лагерный scope mismatch и выдуманное число остались под демоутом.
- `fact_grounding` не ослаблен.

OFF-паритет:

- Controlled OFF совпал с `origin/main` по кодам/деталям.
- Отдельная проверка автономности с флагом OFF совпала с `origin/main`: маршрут остаётся `draft_for_manager`.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_tz122_wrong_intent_fact.py \
  tests/test_subscription_llm_draft_provider.py

492 passed in 6.88s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3299 passed, 5 skipped, 1 warning in 41.58s
```

Предупреждение: `urllib3`/`LibreSSL`, не связано с ТЗ-122.

## Semantic review

Вердикт: `semantic_pass_for_tz122_followup`.

Смысловая проверка:

- клиент спрашивает про старт учебного года -> бот теперь может дать проверенные даты старта, не уходя к менеджеру;
- клиент спрашивает цену, а черновик отвечает адресом -> защита сохраняется;
- контактные часы офиса не считаются расписанием занятий;
- лагерный scope mismatch не ослаблен;
- флаг не включён в профиль, поэтому изменение инертно без явного opt-in.

Остаток:

- лагерные POS всё ещё иногда остаются менеджерскими из-за слоя политики/ретривера, а не из-за `wrong_intent_fact`.
  Это вынесено в отдельный C2 backlog.

## Verdict

`PASS_FOR_CLAUDE_REGRADE`.

Флаг пока не включать в `pilot_gold_v1`.
