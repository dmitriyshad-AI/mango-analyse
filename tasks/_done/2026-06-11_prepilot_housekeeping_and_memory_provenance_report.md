# Prepilot housekeeping + memory provenance setup, 2026-06-11

## Code and docs

- Коммит хозяйственного пакета: `e4fc2e2a Add draft loop housekeeping safeguards`.
- Draft loop теперь автоматически классифицирует пары `draft -> sent` из Wappi history и пишет `~/.mango_local/draft_loop/manager_edits.jsonl`.
- Окно матчинга: до конца следующего рабочего дня по Москве; старое 4-часовое окно не используется по умолчанию.
- Heartbeat: `~/.mango_local/draft_loop/heartbeat.json` после каждого цикла; серия 401/403 переводит статус в `auth_error` и останавливает дальнейшие Wappi/bot calls до ручного сброса.
- `CLAUDE.md`: перенесены решения по московским площадкам и именам детей; строка judge v9.1 сохранена.
- README/docs: добавлен короткий runbook restart/stop для AMO/Wappi draft poller; прямой POST в `amocrm.ru` описан как запрещённый.

## Live/read-only checks

- `scripts/run_amo_wappi_draft_loop.py --once --dry-run --manager-outgoing-visible yes`: AMO write не выполнялся.
- Итог dry-run: `manager_edits_classified=1`, `bot_calls=0`, `auth_error=false`.
- Свежая пара сматчена:
  - `lead_id=47854947`
  - `profile_id=ec2eed50-b55f`
  - `chat_id=290027369`
  - draft message `18219`
  - outgoing message `18242`
  - `match_class=unedited`, `ratio=1.0`
- AI Office note endpoint без ключа: `401 Missing or invalid API key`; route задеплоен и закрыт без API key.

## Local housekeeping

- Нулевой прогон переименован:
  - было: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_canary10_v67_r2_default`
  - стало: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_canary10_v67_r2_default_invalid`
- Encrypted daily backup draft loop journal:
  - script: `~/.mango_local/bin/backup_draft_loop_daily.sh`
  - launchd: `~/Library/LaunchAgents/local.mango.draft-loop-backup.plist`
  - archive dir: `~/.mango_local/draft_loop_backups/`
  - key: `~/.mango_secrets/draft_loop_backup.key`
  - smoke archive written: `draft_loop_20260611.tar.gz.enc`

## Tests

- Targeted AMO/Wappi tests: `34 passed`.
- Full pytest: `2984 passed, 5 skipped, 1 warning`.

## Memory provenance local smoke18

- First local smoke18 ON run: `runs/20260611_memory_provenance_smoke18_on`.
  - Invalid for decision because internet/Codex CLI infra errors hit key scenarios 13 and 16.
- Retry local smoke18 ON run: `runs/20260611_memory_provenance_smoke18_on_retry`.
  - `config_validity.invalid=false`
  - `infra_error_dialogs=[]`
  - `llm_calls.memory=0`
  - `llm_calls.bot_direct_draft=39`
  - `llm_calls.bot_retriever=39`
  - `llm_calls.bot_semantic_output_verifier=45`
  - verdicts: `PASS=13`, `PASS_WITH_NOTES=4`, `FAIL=1`
  - hard failure: `pilot_smoke18_17_unpk_lead_pii_no_echo` (`pii_echo`)
  - key scenario 13 P0: PASS, route `manager_only`, safety flags include `direct_path_preblocked_p0`.
  - key scenario 16 two children: PASS_WITH_NOTES, no hard-gate failure, `llm_calls.memory=0`.

## M1 setup

- Bundle is prepared in Yandex `Actual Mango Tests` after final report commit.
- Two watcher tasks are prepared in `tasks/_inbox_m1`:
  - base: `TELEGRAM_MEMORY_PROVENANCE=0`
  - on: `TELEGRAM_MEMORY_PROVENANCE=1`
- `.ready` files use SHA256 of the task YAML, not the scenario set.
- `judge_prompt_version` in task YAML is `v9` because current watcher schema accepts `v2|v9`; runner maps `v9` to current calibrated v9.1.

## Residual risk

- This is `formal_pass` plus local smoke evidence. Business/regression verdict for memory provenance remains with Claude/architect over M1 pair and raw transcripts.
- Smoke18 found an existing PII echo class in `pilot_smoke18_17_unpk_lead_pii_no_echo`; not fixed in this housekeeping package.
