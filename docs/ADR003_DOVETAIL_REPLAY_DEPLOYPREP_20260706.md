# ADR-003 dovetail, Wappi replay, deploy-prep

Дата: 2026-07-06
Ветка: `codex/adr003-semanticframe-migration`
HEAD до текущих правок: `52af7ffd0ae1f088f66adcab1fbe373378ec789a`

## Вердикт по задаче Claude

GO_WITH_FIXES.

План Claude в целом правильный по приоритету: сначала довесок live-status, потом deploy-prep, потом replay. Но буквальное "удалить stem мест" опасно. По сырью остатки `мест/брон/налич` делятся на разные классы:

- `conversation_intent_plan._asks_live_availability` - legacy-input floor, не должен быть смыслом для модели.
- `policy_routing` live-status guards - fail-closed пол и безопасные manager handoff тексты.
- `reliable_answerer._AVAILABILITY_PROMISE_RE` - output-floor против обещаний мест без live-факта.
- `semantic_reading.sense_seats_reading_decision` - trace/reading слой, не кандидат на широкое удаление.
- safe templates с текстом "наличие мест проверит менеджер" - клиентская безопасная формулировка, не понимание.

Принятое решение: не вырезать все регексы по слову `мест`; отделить legacy floor от prompt-visible understanding и оставить safety floors как верификаторы.

## Что сделано в коде

1. `ConversationIntentPlan.to_prompt_view()` больше не отдаёт `legacy_live_availability_floor_signal` и одноимённую decision note.
2. Добавлен `ConversationIntentPlan.to_internal_view()` для guard-only полей.
3. `telegram_pilot_context_builder` кладёт внутренний view в `conversation_intent_plan_internal`.
4. `policy_routing._conversation_plan_live_availability_floor_result()` читает floor из внутреннего view, а старый prompt-key оставлен только как обратная совместимость.
5. Добавлены регрессии: внутренний floor не попадает в prompt, но fail-closed route сохраняется.

## Wappi replay

Read-only выгрузка выполнена с явным флагом `--allow-live-wappi-read`.

- Raw path: `~/.mango_local/replay_exam/raw/wappi_replay_raw_20260706_080927/`
- Raw dialogs: 100
- Raw messages: 1809
- Raw policy: только локально, не в git, не в Яндекс.

Усилен replay-контур:

- `export_wappi_replay_dialogs.py` теперь делает реальную Wappi GET-выгрузку по 4 профилям из `~/.mango_secrets/amo_wappi_profiles.json`.
- Transport для replay использует `SafeTransportPolicy.wappi_read_only()`: только Wappi GET, без AMO/AI Office.
- `/sync/messages/get` всегда `mark_all=false`.
- `/sync/chats/get` всегда `show_all=false`.
- Raw output принудительно только под `~/.mango_local/replay_exam/raw/`.
- Добавлен `build_wappi_replay_cases.py`: raw manifest -> pseudonymized dialogs -> teacher-forcing cases -> sample для ручной проверки.
- Псевдонимизатор теперь маскирует profile/chat/message/lead/contact/talk/thread/dialog ids; ID-alias использует base32, чтобы не выглядеть как телефон.
- `load_cases()` сохраняет `prefix_messages`, иначе threaded-memory экзамен был бы неверным.

Локальный scrubbed pilot:

- Scrubbed path: `~/.mango_local/replay_exam/scrubbed/replay_cases_wappi_replay_raw_20260706_080927/`
- Dialogs: 10
- Cases: 31
- Segment counts: `chat_only=31`
- Leak count: 0
- Local report with 5 scrubbed tests: `~/.mango_local/replay_exam/scrubbed/replay_cases_wappi_replay_raw_20260706_080927/pilot10_methodology_report.md`
- Report PII check: 0 signals.

Важно: это formal pipeline pass, не semantic quality pass. `run_wappi_replay_exam.py` пока имеет только `--fake-provider`; реальный provider adapter для проверки качества ответа бота ещё не включён.

## Deploy-prep read-only snapshot

Актуальная read-only фактура на этой машине:

- Live Telegram pid: `60227`
- Process start: `Mon Jun 29 01:21:14 2026`
- Command: `scripts/run_telegram_public_pilot_bots.py --env-file /dev/null --mode poll --brand all`
- CWD pid 60227: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`
- Screen sessions: `mango_public_pilot_bots_main_eb6fa0b_reliable_20260629_v2`, `mango_draft_loop_watchdog`
- Heartbeat: `status=polling`, `effective_profile=pilot_gold_v1`, `draft_path=direct_path`, brands list length 2.
- Runtime flags file keys: `CODEX_HOME`, `MANGO_TELEGRAM_KB_SNAPSHOT`, `TELEGRAM_AUTONOMY_SCOPE_PRECISION`, `TELEGRAM_DIRECT_PATH_PILOT_CONFIG`, `TELEGRAM_FACT_VENUE_SCOPE`, `TELEGRAM_P0_MODEL_LED`, `TELEGRAM_PROSE_MODEL_LED`, `TELEGRAM_RELIABLE_ANSWERER_STEP1`.

Worktrees:

- `/Users/dmitrijfabarisov/Projects/Mango analyse` -> `9e8fb3b`, branch `codex/tz135-direct-wow-tone`
- `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` -> `1bb51de`, branch `codex/email-pipeline-restore`
- `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` -> `4caa5eb`, detached
- `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` -> `52af7ffd`, branch `codex/adr003-semanticframe-migration`

Нота: `docs/worktrees_registry.md` и имя screen-сессии могут быть устаревшими относительно факта `ps/lsof`. Перед любым свапом нужно заново снять `ps`, `lsof`, heartbeat и `git rev-parse HEAD`.

Runtime pointer в текущем worktree:

- `stable_runtime/CANONICAL_EXPORT.txt` -> `sales_master_export_20260523_audio_working_store_v1`
- `stable_runtime/CURRENT_RUNTIME.json` в текущем worktree отсутствует.

Draft-loop/Wappi:

- Активного `run_amo_wappi_draft_loop.py` процесса не найдено.
- Watchdog screen жив.
- `~/.mango_local/draft_loop/heartbeat.json`: `status=ok`, `last_cycle_at=2026-06-30T07:51:52.907023+00:00`, `dry_run` проверять в summary перед запуском.
- `~/.mango_secrets/STOP_DRAFT_LOOP` не проверялся изменением; stop-file не создавался.

Secrets inventory только по метаданным:

- `amo_wappi.env`, `ai_office.env`, `foton_crm_readonly_mcp_connector.env`: mode `0600`, ключи прочитаны без значений.
- `amo_wappi_profiles*.json`, `draft_loop_pairs*.json`, `amo_wappi_phase1.json`, `wappi_storage_state.json`: значения не включать в отчёты.
- `server_access_packs/`, `*.bak_tz15_*`, `draft_loop_backup.key`: не раскрывать содержимое.

## Черновик swap/rollback плана

Не выполнено, только проект.

1. Freeze:
   - `git status --short --branch`
   - `git rev-parse HEAD`
   - `ps -p <pid> -o pid=,ppid=,lstart=,stat=,command=`
   - `lsof -a -p <pid> -d cwd`
   - прочитать heartbeat и runtime flag keys без секретов.
2. Smoke на candidate HEAD:
   - targeted pytest
   - локальный direct-path smoke без live-write
   - проверить `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
3. Stop-crane:
   - иметь готовую команду остановки screen/process, но не выполнять без отдельного "да".
   - Wappi draft-loop перед деплоем должен иметь явный dry-run/stop статус.
4. Swap:
   - только после отдельного подтверждения владельца и Claude reggrade.
   - запускать из явно выбранного worktree, не по имени screen.
5. Rollback:
   - вернуть предыдущий worktree/HEAD и env-файл.
   - проверить heartbeat, что pid новый и cwd соответствует rollback.

## Проверки

- Targeted hidden-floor/replay/transport: `67 passed`.
- ADR003 moratorium + semantic reading/direct-path соседние тесты: `682 passed`.
- New scripts py_compile: pass.

## Остаточные риски

1. Replay semantic-pass не получен: fake-provider проверяет pipeline, но не качество ответов бота.
2. `conversation_intent_plan_internal` передаётся в provider-context как hidden key; prompt-builder его не сериализует, но будущие prompt builders должны сохранять whitelist.
3. Live pid/cwd факт конфликтует с устаревшими registry/screen именами; перед деплоем нужен новый freeze.
