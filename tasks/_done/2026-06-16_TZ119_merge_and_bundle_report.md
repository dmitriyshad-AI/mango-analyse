# 2026-06-16: TZ119 merge + M1 bundle

## Scope

Регрейд ТЗ-119 PASS. Ветка `codex/tz119-assumed-scope-guard-main` влита в `main`.

Рабочее дерево: `/Users/dmitrijfabarisov/Projects/Mango_tz113_114_115_profile`  
Branch: `main`

## Merge

До merge:

- `main`: `2268f55`
- `codex/tz119-assumed-scope-guard-main`: `aea3674`

Merge commit:

- `dab6d79` / full `dab6d79112be607509e1aa6b5207e3023b76d7d4`

Конфликты были в:

- `src/mango_mvp/channels/subscription_llm_parts/__init__.py`
- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`
- `tests/test_subscription_llm_draft_provider.py`

Разрешение:

- оставлены оба набора экспортов: answerability/model-P0/action-decision + TZ119 assumed-scope helpers;
- сохранён `TELEGRAM_ASSUMED_SCOPE_GUARD` как default OFF;
- `RETRIEVER_MODEL_DRIVEN` включается только при связке `TELEGRAM_ASSUMED_SCOPE_GUARD=1` + `TELEGRAM_RETRIEVER_MODEL_DRIVEN=1`;
- тест contact-trace уточнён: для проверки старого keyword contact path явно выключает `TELEGRAM_LLM_RETRIEVE=0`, чтобы не путать его с боевым LLM-retriever.

Проверка профиля:

```text
ASSUMED_SCOPE_GUARD in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS = False
model_driven default in pilot = False
model_driven explicit without guard = False
model_driven explicit with guard = True
```

## Tests

Точечно:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "template_from_kb_contact_trace_is_visible or assumed_scope or pilot_gold_v1_enables_full_battle_profile_flags or retriever_model_driven"
5 passed, 476 deselected
```

Перед этим профиль/assumed/action набор:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "assumed_scope or pilot_gold_v1_enables_full_battle_profile_flags or retriever_model_driven or answerability or direct_path_model_p0 or deal_action" tests/test_deal_action_decision.py
28 passed, 462 deselected
```

Полный набор:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3288 passed, 5 skipped, 1 warning
```

## Bundle

Собран новый bundle для M1 от merge commit `dab6d79112be607509e1aa6b5207e3023b76d7d4`.

Папка:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/mango_clean_dab6d791`

Проверено:

- `manifest.json` есть;
- `BUNDLE_INFO.txt` есть;
- `branch: main`;
- `head: dab6d79112be607509e1aa6b5207e3023b76d7d4`;
- `kb_snapshot: product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`;
- `pilot_config: pilot_gold_v1`.

## Notes

`TELEGRAM_ASSUMED_SCOPE_GUARD` в боевой профиль не добавлен. Для замера включать только явной связкой с retriever/model-driven флагами.
