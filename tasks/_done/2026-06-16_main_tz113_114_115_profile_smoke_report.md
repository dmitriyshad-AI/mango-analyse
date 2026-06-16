# 2026-06-16: main merge + pilot profile smoke

## Merge

Ветка `codex/tz113-114-115-profile` влита в `main` fast-forward.

До merge:

- `main` / `origin/main`: `89d753f`

После merge и push:

- `main` / `origin/main`: `a7d6271`

Вошло:

- `tz113` answerability trace;
- `tz114` autonomy topic fix;
- `tz115` judge date + meta leak calibration;
- нейтрализация answerability: датчик не меняет основной direct-path prompt;
- включение в `pilot_gold_v1`:
  - `TELEGRAM_DIRECT_PATH_MODEL_P0`;
  - `TELEGRAM_DEAL_ACTION_DECISION`.

## Smoke

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl \
  --bot-mode codex --bot-reasoning medium \
  --memory-mode codex --memory-reasoning low \
  --semantic-mode codex --semantic-reasoning medium \
  --semantic-verifier-mode codex --semantic-verifier-reasoning medium \
  --parallel 4 --judge-prompt-version v9.1 \
  --out-dir "/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260616_main_tz113_114_115_profile_smoke18"
```

Out dir:

`/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260616_main_tz113_114_115_profile_smoke18`

Итог:

| metric | value |
| --- | ---: |
| dialogs | 18 |
| turns | 41 |
| PASS | 7 |
| PASS_WITH_NOTES | 11 |
| FAIL | 0 |
| hard_gate_failures | 0 |
| infra_error_dialogs | 0 |
| completed | 18 |
| config_validity.invalid | false |

Ключевые признаки профиля:

- `run_config.key_flags.profile.effective = true`
- `run_config.key_flags.retriever.effective = true`
- `run_config.key_flags.rubric.effective = true`
- `run_config.key_flags.render.effective = true`
- `run_config.key_flags.memory_provenance.effective = true`
- `answerability_trace.turn_count = 41`
- `action_decision.enabled_turns = 41`
- `semantic_output_verifier.checked_turns = 39`
- `semantic_output_verifier.unavailable_turns = 0`

LLM calls:

| role | count |
| --- | ---: |
| bot_direct_draft | 39 |
| bot_retriever | 39 |
| bot_semantic_output_verifier | 43 |
| bot_semantic_output_regen | 3 |
| judge | 18 |
| client | 41 |
| memory | 0 |
| total | 183 |

## Notes

Smoke выполнен на `main` после push merge-коммита `a7d6271`. Отчётный файл добавлен после smoke; код не менялся.
