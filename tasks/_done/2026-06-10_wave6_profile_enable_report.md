# Wave 6 in pilot profile

Дата: 2026-06-10
Коммит: `5d2aa969` (`Enable Wave 6 retriever in pilot profile`)

## Что изменено

- `TELEGRAM_LLM_RETRIEVE` добавлен в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- `_llm_retrieve_enabled()` теперь работает через общий профильный helper: `pilot_gold_v1` включает Wave 6 по умолчанию, явный `TELEGRAM_LLM_RETRIEVE=0` поверх профиля отключает LLM-ретривер.
- NEG: `test_pilot_gold_v1_llm_retrieve_explicit_zero_keeps_keyword_pack` проверяет, что явный `0` не вызывает retriever и сохраняет старый keyword-pack.

## Проверки

- Targeted: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'wave6_llm_retrieve or pilot_gold_v1'`
  - `11 passed, 427 deselected`
- Full: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2902 passed, 5 skipped, 1 warning`

## Smoke canary10

Набор: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/smoke_v2_acceptance_2026-06-08_canary10.jsonl`

sha256: `62b1c2bcf94c96922f749e93c4b81809b98206bddc6ce90ba0d966a93c898f99`

Валидный прогон:

- `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_wave6_profile_canary10_5d2aa969_fast`
- Команда: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, default KB r3, `--judge-prompt-version v9`, `--parallel 4`, временный `CODEX_HOME=/tmp/mango_codex_home_fast_5d2aa969` с `service_tier="fast"`; основной `~/.codex/config.toml` не менялся.
- `config_validity.invalid=false`
- `bot_direct_draft=11`
- `bot_retriever=11`
- `bot_semantic_output_verifier=16`
- `bot_faithfulness=0`
- Итог: `10` диалогов, `33` хода, `PASS=3`, `PASS_WITH_NOTES=6`, `FAIL=1`.
- Единственный hard fail: `sm_u_p0_complaint`, `violated_gates=["p0_mishandled"]`; разбор за архитектором.

Невалидные попытки до обхода личного конфига:

- `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_wave6_profile_canary10_5d2aa969`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_wave6_profile_canary10_5d2aa969_isolated`

Причина: `codex exec failed rc=1: Error loading config.toml: unknown variant default, expected fast or flex in service_tier`. Ходов в этих попытках `0`, в оценку не брать.

## Принятая пере-пара Wave 6

Сырые прогоны, на которые ссылается решение Дмитрия/архитектора:

- `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/runs/20260610_wave6_repair_base_smoke89_codex`
- `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/runs/20260610_wave6_repair_on_smoke89_codex`

Итог решения из чата: уходы-в-тексте черновиков `-43%`, self `+53%`, derived `0`, fallback `4.3%`, новых жёстких нет. После включения профиль `pilot_gold_v1` содержит: рубрика + Wave 6 + template render + KB r3 + presale.
