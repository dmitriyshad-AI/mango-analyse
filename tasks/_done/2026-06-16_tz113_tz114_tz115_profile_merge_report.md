# 2026-06-16: tz113 + tz114 + tz115 profile merge prep

## Scope

Подготовлена ветка для регрейда перед мержем в `main`.

Ветка: `codex/tz113-114-115-profile`  
База: `main` / `origin/main` = `89d753f`

В основную папку `/Users/dmitrijfabarisov/Projects/Mango analyse` не писал: там активен другой трек (`codex/tz119-assumed-scope-guard-main`) и untracked-служебные файлы.

## Влито

- `codex/tz113-answerability-shadow` (`83c824f`) — answerability trace.
- `codex/tz114-autonomy-topic` (`8c6b2aa`) — autonomy topic fix.
- `codex/tz115-judge-date-meta-leak` (`8442e7d`) — judge date/meta leak fix.
- `codex/answerability-shadow-neutrality` (`fe19db1`) — нейтрализация answerability: датчик больше не добавляет поля самооценки в основной direct-path prompt.

Все merge прошли через `ort` без ручного разрешения конфликтов.

## Профиль pilot_gold_v1

В `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS` теперь включены:

- `TELEGRAM_ANSWERABILITY_SHADOW` — пришёл из принятой ветки нейтрализации.
- `TELEGRAM_DIRECT_PATH_MODEL_P0` — включён этой правкой.
- `TELEGRAM_DEAL_ACTION_DECISION` — включён этой правкой.

Важно: функции `_direct_path_model_p0_enabled()` и `_deal_action_decision_enabled()` теперь реально учитывают профиль `pilot_gold_v1`; раньше они смотрели только явный env/context-флаг.

NEG:

- без `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1` оба новых слоя остаются выключены;
- явный `TELEGRAM_DIRECT_PATH_MODEL_P0=0` поверх профиля выключает model-P0;
- явный `TELEGRAM_DEAL_ACTION_DECISION=0` поверх профиля выключает action decision.

## Tests

Точечно:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "pilot_gold_v1_enables_full_battle_profile_flags or answerability or direct_path_model_p0 or deal_action" tests/test_deal_action_decision.py
25 passed, 458 deselected
```

Полный набор:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3281 passed, 5 skipped, 1 warning
```

## Notes for review

В `main` не мержил и не пушил. Ветка готова под регрейд Дмитрия/Claude #1 перед финальным мержем в канон.
