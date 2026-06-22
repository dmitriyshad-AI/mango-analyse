# Фаза 4: M1 stack V2 для чистого direct-path профиля

Дата: 2026-06-22  
Исполнитель: Codex  
Ветка: `codex/tz-profile-selfcheck`  
База ветки: `8ff43b7519802016acbdc35094048569fc6c019c`  
ТЗ: `Foton/2026-06-21_TZ_edinyy_profil_samoproverka_chekera.md`, Фаза 4

## Что сделано

- В `scripts/m1_watcher.py` добавлен `PRODUCTION_ENV_STACK_V2`.
- `PRODUCTION_ENV_STACK_V2` содержит ровно один ключ: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
- Исторический `PRODUCTION_ENV_STACK` не изменён по значениям и помечен deprecated since `2026-06-21`.
- Добавлен явный переключатель `--production-env-stack {legacy,old,v1,v2}`; default остаётся `legacy`, чтобы без отмашки ничего не поменять.
- `M1Watcher(..., production_stack_version="v2")` и отчёты watcher-а теперь фиксируют `production_stack_version`.
- Фаза 5 не выполнялась; состав `pilot_gold_v1` не менялся.

## Чего нет в V2

В V2 намеренно не включены исторические M1-only флаги, включая:

- `TELEGRAM_STEP4_KEEP_ANSWER`
- `TELEGRAM_A_FREE_NUMBER_GATE`
- `TELEGRAM_STEP4_NUMBER_GROUNDING`
- `TELEGRAM_PH2_TONE`
- `TELEGRAM_PH2_OBJECTION`
- `TELEGRAM_PH2_ANXIETY`
- `TELEGRAM_TONE_WARM_FRAME`
- `TELEGRAM_TONE_CLOSE_DETECT`
- `TELEGRAM_TONE_SELL_PROMPT`
- `TELEGRAM_TONE_RICH_FORMAT`
- `TELEGRAM_Q_PARTIAL_YIELD`
- `TELEGRAM_Q_CLARIFY_SCOPE`
- `TELEGRAM_Q_USEFUL_HANDOFF`
- `TELEGRAM_A_TRAVEL_COMPOSE`
- `TELEGRAM_A_ESTIMATE_MODE`
- `TELEGRAM_COMPOSITE_CONTRACT_FIX`
- `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE`
- `TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD`
- `TELEGRAM_HANDOFF_TRACE`
- `TELEGRAM_OUTPUT_SANITIZER`
- `TELEGRAM_RULES_ENGINE_PLANNER_INTENT`
- `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER`
- `DIALOGUE_CONTRACT_DEBUG_TRACE`

## Калибровочная дельта

M1 не запускался и не трогался. Калибровка сделана локально и read-only через материализацию `effective_task_env()` на фиксированных синтетических env-delta.

| case | legacy keys | V2 keys | shared | legacy-only | V2-only | changed shared |
|---|---:|---:|---:|---:|---:|---:|
| `empty_delta` | 23 | 1 | 0 | 23 | 1 | 0 |
| `tone_override` | 23 | 2 | 1 | 22 | 1 | 0 |
| `explicit_profile` | 24 | 1 | 1 | 23 | 0 | 0 |

Вывод: старый стек и V2 метриками несравнимы. Пустая задача на legacy включает 23 исторических флага, V2 — только direct-path профиль.

## Тесты

Команды:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile scripts/m1_watcher.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_m1_watcher.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
git diff --check
```

Результат:

- py_compile: PASS
- focused pytest: `24 passed`
- full pytest: `3496 passed, 5 skipped, 1 warning in 72.99s`
- diff check: PASS

Единственный warning: `urllib3` про LibreSSL, не связан с изменениями.

## Read-only и границы

- M1 не запускался и не опрашивался.
- Live Telegram bot, AMO, Tallanto и `stable_runtime` не трогались.
- Merge/push/live-write не выполнялись.
- V2 не включён по умолчанию.

## ACK

ACK: Фаза 4 реализована кодом и тестами, калибровочная дельта приложена, STOP на регрейд Claude #1.
