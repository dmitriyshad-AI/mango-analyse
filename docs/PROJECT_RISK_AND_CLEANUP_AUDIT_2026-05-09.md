# Project Risk And Cleanup Audit

Дата: 2026-05-09

Режим: read-only аудит с единственной записью этого файла. Файлы проекта, runtime DB, audio, transcripts, AMO/Tallanto/CRM и тяжелые batch/start/run-ui/ASR/R+A сценарии не запускались и не изменялись.

## Executive Summary

Проект в git сейчас выглядит чисто, но рабочая директория фактически является смесью кода, runtime/data lake, секретов, исторических DB, Excel/CSV выгрузок, Telegram/Mango архивов, Codex auth snapshots и опасных операционных скриптов.

Самые критичные риски:

1. Секреты и refresh/auth tokens лежат в ignored, но локально доступных файлах: `.env`, `*.env.private`, `.codex_local/auth.json`, `.codex_workers/*/auth.json`, `stable_runtime/**/codex_home/auth.json`, shell snapshots, SSH private key.
2. Есть реальные AMO writeback entrypoints без достаточного внешнего friction: `scripts/write_amo_ready_contacts.py`, `scripts/write_recent_actionable_deals.py`, `/deals/writeback`, legacy `sync_amocrm`.
3. `scripts/` вырос до 98 файлов, а каталог `docs/CLI_AND_SCRIPTS_CATALOG_2026-05-07.md` описывает 49; safety status для новых Mango/SaaS scripts неполный.
4. `stable_runtime` и корень проекта содержат большие runtime-данные: `stable_runtime` ~14G, source audio ~24G, `.git` ~2.4G, Telegram exports ~1.2G.
5. Тесты есть и их много, но покрытие неровное: `productization` покрыт хорошо; `gui.py`, FastAPI routers, external clients, legacy CRM sync и часть CLI имеют слабое покрытие.

## Git Snapshot

Команды:

```bash
git status --short --branch
git branch --show-current
git log --oneline --decorate -n 12
```

Результат:

- Ветка: `codex/saas-productization-baseline`.
- Tracking: `origin/codex/saas-productization-baseline`.
- `git status --short --branch`: clean, без видимых modified/untracked.
- Последний коммит: `4d0cdba Add pipeline quick actions to GUI`.
- Недавняя история включает housekeeping, runtime preview API, operations audit docs, SaaS baseline.
- Tracked files: 414.

Важно: clean git не означает clean workspace. `git status --ignored` подтверждает, что критичные data/runtime/secret файлы игнорируются, но находятся рядом с кодом.

## Repository Inventory

Текущая структура:

- `src/mango_mvp`: 106 Python files.
- `scripts`: 98 files, примерно 94 Python + 4 shell.
- `tests`: 76 test files, 441 collected tests.
- `docs`: 65 Markdown files.
- `stable_runtime`: tracked scaffolding + large ignored runtime.

Крупнейшие top-level объекты:

| Path | Size | Risk |
|---|---:|---|
| `2026-03-09--26` | ~24G | source audio / PII, держать вне code repo |
| `stable_runtime` | ~14G | DB/audio/transcripts/logs/launchers mixed with tracked scripts |
| `.git` | ~2.4G | локальная object database раздута |
| `telegram_exports (2)` | ~1.2G | Telegram/CRM PII exports |
| `_local_archive_20260424` | ~1.1G | архив лежит внутри проекта |
| `2026-03-05-21-06-49-ч1`, `ч2` | ~985M each | старые source export folders |
| `.venv-asrbench` | ~931M | локальное окружение |
| `_local_archive_mango_api_downloads_20260507` | ~414M | Mango capture/download archive |
| `.cache` | ~280M | local cache |
| `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` | ~269M | external ASR result |
| `mango_mvp.db` | ~89M | root runtime SQLite |

Root runtime/export files also include `ai_office.db`, `mango_mvp.db-shm`, `mango_mvp.db-wal`, `Contacts.xls`, `sales_workbook.xlsx`, `АКТУАЛЬНО_*.xlsx/json`, `ОТЧЕТ_*.xlsx`, `tallanto_postman_collection.json`.

## Secrets And Sensitive Data

Values were not copied into this report.

Critical local secret locations:

- `.env`: `POSTGRES_PASSWORD`, `DATABASE_URL`, `AI_OFFICE_API_KEY`, `VITE_API_KEY`, `AI_OFFICE_STREAM_TOKEN_SECRET`, `CRM_TALLANTO_API_TOKEN`, `CRM_AMO_API_TOKEN`, `MANGO_OFFICE_API_KEY`, `MANGO_OFFICE_API_SALT`.
- `prod_runtime_transfer/.env.private`: `AMOCRM_ACCESS_TOKEN`.
- `stable_runtime/amocrm_runtime/.env.private`: `AI_OFFICE_API_KEY`, `DATABASE_URL`, `CRM_TALLANTO_API_TOKEN`.
- `mango_tallanto_transfer/.env.private`: `CRM_TALLANTO_API_TOKEN`.
- `stable_runtime/amocrm_runtime/ssh/id_ed25519_mango_runtime`: private SSH key.
- `.codex_local/auth.json`, `.codex_local/llm_review_home/auth.json`, `.codex_workers/ra*/auth.json`, `stable_runtime/**/codex_home/auth.json`: Codex/OpenAI auth and refresh tokens.
- `stable_runtime/**/shell_snapshots/*.sh`: historical exported env with API keys/tokens.

Git status:

- These paths are currently ignored by `.gitignore`.
- `git ls-files` did not show the checked critical secret files as tracked.
- Local risk remains high because secrets are duplicated in runtime/archive/snapshot folders and can leak via zip/rsync/manual copy.

Sensitive data / PII surface:

- Hundreds of thousands of paths appear to contain phone-like identifiers, especially in `transcripts`, `messages(...)`, audio/transcript artifacts.
- `messages(35)/index.html` contains names and phones.
- `stable_runtime/**/transcripts/*_structured_fields.json` includes `parent_fio`, `child_fio`, `email`, `phone_from_filename`.
- `telegram_exports (2)`, CRM CSV/XLSX, root `АКТУАЛЬНО_*` exports, DB files, JSONL logs contain business/customer data.
- Data containers found include many DB/SQLite/XLSX/CSV/JSONL artifacts. Root examples: `mango_mvp.db`, `ai_office.db`, `Contacts.xls`, `АКТУАЛЬНО_Контакты_для_продаж.xlsx`.

## Dangerous Write And Network Operations

P0 write risks:

- `scripts/write_amo_ready_contacts.py` calls `send_contact_custom_field_update` and commits. It has `--limit` and skip-report support, but no default dry-run.
- `scripts/write_recent_actionable_deals.py` calls `write_analysis_to_lead` and commits. It can build a fresh queue and write only actionable deals, but still performs live amoCRM writeback.
- `src/mango_mvp/amocrm_runtime/amo_integration.py` contains AMO PATCH primitives for contacts/leads.
- `src/mango_mvp/amocrm_runtime/routers/deals.py` exposes `/deals/writeback` and queue build with `apply_writeback`.
- `src/mango_mvp/services/sync_amocrm.py` legacy sync can add notes, update contact fields and create tasks when `LEGACY_AMOCRM_SYNC_ENABLED=true`.

High-risk execution paths:

- `src/mango_mvp/gui.py` launches CLI/workers through `subprocess.Popen`; this can start ASR/R+A and DB mutations.
- `src/mango_mvp/services/transcribe.py`, `resolve.py`, `analyze.py` call external providers/subprocesses and commit pipeline status/results.
- `src/mango_mvp/services/export_ai_office.py` posts insights to AI Office.
- `scripts/mango_office_download_recordings.py` downloads recordings by default after Mango API calls; newer `mango_office_recording_capture_download.py` is safer because execution requires `--execute`.
- `scripts/repair_and_move_message_archives.py` can unlink/move real archive files; dry-run exists but is not the only behavior.

DB/file mutation risks:

- `src/mango_mvp/cli.py` has reset/requeue commands that update many `call_records`.
- `scripts/requeue_secondary_backfill.py`, `finalize_manual_non_conversation_tail.py`, `finalize_messages30_tail.py`, `prefill_asr_from_dbs.py` mutate SQLite state.
- Wave/subset scripts such as `prepare_date_window_subset.py`, `prepare_dual_asr_new_llm_wave.py`, `prepare_history_gap_wave.py`, `prepare_priority_history_wave.py` copy DBs and then delete/update subsets. Safe only when target paths are definitely disposable.
- Productization has many guard rails, but still includes destructive primitives for restore/archive/import: `product_db.py`, `quarantine_import.py`, `payload_archive.py`, `recording_asset_ingest.py`.

Shell risks:

- `scripts/autocommit_push_loop.sh`: `git add -A`, auto commit, auto push loop.
- `scripts/start_autocommit_push.sh`: starts the loop through `nohup`.
- `scripts/git_bootstrap.sh`: changes origin, `git add -A`, commit, push.
- `stable_runtime/start*`, `run-ui*`, `audio2text.sh`, R+A launchers and shadow/live scripts are operational entrypoints, not safe audit commands.

## Old, Duplicate, Temporary And Runtime Artifacts

Safe cleanup candidates after confirmation:

- `__pycache__` dirs: currently 64 found.
- `.DS_Store`: currently 1 found.
- `.pytest_cache`.
- Local generated caches: `.cache`, `.codex_workers`, selected `.codex_local` cache/session folders only after auth/token handling is decided.

Do not delete without data-owner confirmation:

- `2026-03-09--26`.
- Any `stable_runtime` DB/audio/transcripts.
- Root `mango_mvp.db*` while SQLite may be open.
- `telegram_exports (2)`.
- `Contacts.xls`, `АКТУАЛЬНО_*.xlsx/json`, `ОТЧЕТ_*.xlsx`.
- `external_m1_*` results and zip.
- `_local_archive_*` until an external archive manifest exists.

Archive/dedup candidates:

- `_local_archive_20260424`, `_local_archive_mango_api_downloads_20260507`.
- `2026-03-05-21-06-49-ч1`, `2026-03-05-21-06-49-ч2`.
- `external_m1_jan2025_test300_20260503` root/stable duplicates.
- `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` and zip.
- `telegram_exports (2)`.
- `stable_runtime/*/*.before_*.db`, old `*_backup_*`, old benchmark/ab_test DBs, after coverage/import gates.
- Many dated audit docs in `docs/`; keep canonical docs in root and move historical audits conceptually under `docs/audits/YYYY-MM-DD/`.

Script duplicates / generations:

- `mango_office_download_recordings.py` vs newer guarded `mango_office_recording_capture_download.py`.
- `prepare_message_archive_wave.py`, `prepare_message_archive_history_full_cycle.py`, `prepare_message_archives_history_full_cycle.py`.
- ASR batch family: `prepare_remaining_asr_batch.py`, `prepare_untranscribed_merge_batches.py`, `prepare_overnight_full_asr_priority.py`, `prepare_asr_only_date_window.py`.
- R+A wave family: `prepare_date_window_subset.py`, `prepare_history_gap_wave.py`, `prepare_dual_asr_new_llm_wave.py`, `prepare_llm_wave_from_recommendations.py`, `prepare_priority_history_wave.py`, `prepare_resolve_analyze_missing_batch.py`, `prepare_manual_tail_analyze_fallback.py`.
- `SAAS_7_STAGE_GATE_AUDIT_2026-05-08.md` is partly superseded by `SAAS_9_STAGE_ENDGAME_AUDIT_2026-05-08.md`.

## Docs And Scripts State

Issues:

- `docs/CLI_AND_SCRIPTS_CATALOG_2026-05-07.md` says 49 script files; actual count is 98.
- `README.md` still documents `git_bootstrap.sh` and auto commit/push loop, which are unsafe for normal operations.
- Runbook forbids AMO writeback without dry-run/quality gate, but live write scripts exist without strong CLI friction.
- No canonical `docs/SCRIPT_SAFETY_MATRIX.md`.
- No canonical current architecture doc despite earlier cleanup audit recommending `docs/ARCHITECTURE_CURRENT.md`.
- `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md` / writeback policy is still missing as a canonical gate.

Recommended docs:

- `docs/CLI_AND_SCRIPTS_CATALOG.md`: regenerate for all 98 scripts.
- `docs/SCRIPT_SAFETY_MATRIX.md`: read-only / writes files / writes DB / network / CRM / ASR / approval required.
- `docs/MANGO_PRODUCTIZATION_RUNBOOK.md`: modern `mango_office_*` stage flow.
- `docs/DATA_MODEL.md`: statuses, queues, dead-letter, coverage semantics.
- `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`: exact fields, writeback ownership, safe mode, rollback.
- `docs/ARCHITECTURE_CURRENT.md`: current layers and boundaries.

## Source And Test Coverage

Observed:

- `pytest --collect-only -q -p no:cacheprovider` collected 441 tests in 1.70s.
- Warning: local Python/urllib3 reports LibreSSL, not OpenSSL.
- I did not run the actual test suite in this audit.

Coverage strengths:

- `productization` is the strongest area: dry-run gates, path guards, idempotency, SQLite sidecars, API/product DB/scheduler/capture/ASR sandbox stages.
- Core services have meaningful unit coverage for `analyze`, `resolve`, `transcribe`, `ingest`, export, claims, CLI guards.
- AMO/Tallanto analysis logic has tests around writeback blockers, safe mode, Tallanto API read behavior, deal analysis.
- Insight generators have smoke/target tests.

Modules with weak or no direct tests:

- `src/mango_mvp/gui.py`: large subprocess/threading/Tk orchestration with no direct tests.
- `src/mango_mvp/clients/amocrm.py`: no direct unit tests for token cache, refresh, request failures.
- `src/mango_mvp/clients/ollama.py`: no direct HTTP/error tests.
- `src/mango_mvp/services/llm_response_cache.py`: only indirect coverage.
- `src/mango_mvp/utils/phone.py`: no direct coverage.
- `src/mango_mvp/amocrm_runtime/config.py`, `schemas.py`, `leads_extension.py`.
- `src/mango_mvp/amocrm_runtime/routers/deals.py`, `routers/integrations.py`, `routers/tallanto.py`.
- `src/mango_mvp/amocrm_runtime/tallanto_context.py`, `tallanto_deal_ranking.py`, `tallanto_matching.py`, `tallanto_premature_close.py`.

Risky coverage gaps:

- FastAPI API-key/auth/router behavior is not tested enough through `TestClient`.
- Legacy AMO sync tests focus on disabled/default gates, not enough on success/error/dead-letter flows.
- CLI coverage is partial: weak areas include `transcribe`, `backfill-second-asr`, `run-all`, `reset-transcribe`, queue exports, workbook commands.
- `transcribe.py`, `resolve.py`, `analyze.py` have mocked unit coverage but not safe integration tests for provider boundaries.
- `insights` large workbook/report generators need more negative tests for malformed CSV/JSON, partial schemas and empty data.

## Test Safety

Unsafe as-is under this audit's constraints:

- `make test`: no env isolation and writes pytest cache unless changed.
- `make test-smoke`: includes `tests/test_smoke.py::SmokePipelineTest::test_stable_runtime_rebuild_smoke`, which calls `stable_runtime/rebuild_snapshot.sh`.
- `make audit` / `make audit-fast`: write reports under `stable_runtime/project_audit_*`; `audit` also runs pytest.
- Any `stable_runtime/start*`, `run-ui*`, ASR/R+A batch, real `mango-mvp worker/run-all/transcribe/resolve/analyze/sync`.

Safe collection command used:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q -p no:cacheprovider
```

Recommended safe test command later, after explicit approval:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"

env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTEST_ADDOPTS= \
  PYTHONPATH=src \
  DATABASE_URL='sqlite:///:memory:' \
  LLM_CACHE_ENABLED=false \
  OPENAI_API_KEY= \
  AMOCRM_BASE_URL= AMOCRM_ACCESS_TOKEN= AMOCRM_REFRESH_TOKEN= AMOCRM_CLIENT_ID= AMOCRM_CLIENT_SECRET= \
  CRM_AMO_MODE=mock CRM_AMO_BASE_URL= CRM_AMO_API_TOKEN= \
  CRM_TALLANTO_MODE=mock CRM_TALLANTO_BASE_URL= CRM_TALLANTO_API_TOKEN= \
  MANGO_OFFICE_API_KEY= MANGO_OFFICE_API_SALT= \
  AI_OFFICE_API_KEY= AI_OFFICE_API_BASE_URL= \
  TRANSCRIBE_PROVIDER=mock SECONDARY_TRANSCRIBE_PROVIDER= DUAL_TRANSCRIBE_ENABLED=false \
  DUAL_MERGE_PROVIDER=rule ANALYZE_PROVIDER=mock RESOLVE_LLM_PROVIDER=rule RESOLVE_RESCUE_PROVIDER= \
  SYNC_DRY_RUN=true LEGACY_AMOCRM_SYNC_ENABLED=false \
  python3 -m pytest -q -p no:cacheprovider tests -k 'not stable_runtime_rebuild_smoke'
```

More conservative:

```bash
python3 -m pytest -q -p no:cacheprovider tests --ignore=tests/test_smoke.py
```

Expected writes in the safe command: only pytest temp dirs, fake SQLite/JSON/CSV/XLSX/audio. Static review suggests external HTTP writes are mocked/injected in tests, but env isolation is still required because project config loads `.env`.

## Priority Plan

P0 - stop accidental secret/writeback damage:

1. Rotate/revoke exposed local tokens that may have been copied into snapshots: AMO, Tallanto, Mango Office, AI Office, Codex/OpenAI session tokens, SSH key.
2. Move secrets out of repo tree into OS keychain/1Password/sops/Doppler or external local secret folder; leave only `.env.example`.
3. Quarantine or archive `auth.json`, shell snapshots and `*.env.private` copies after rotation. Do not delete before confirming replacement access.
4. Add strong dry-run/confirmation guards to `write_amo_ready_contacts.py`, `write_recent_actionable_deals.py`, `/deals/writeback`, and legacy sync entrypoints.
5. Remove README encouragement for auto commit/push and mark those scripts historical.

P1 - make operations auditable:

1. Regenerate script catalog for all 98 scripts.
2. Add `SCRIPT_SAFETY_MATRIX.md`.
3. Add `AMO_TALLANTO_FIELD_MAPPING_PROD.md` and writeback policy.
4. Update runbook to prefer guarded `mango_office_recording_capture_download.py` over old downloader.
5. Define one canonical flow per ASR/R+A/Mango productization stage; mark old scripts legacy or special-case.

P2 - reduce data/runtime blast radius:

1. Create a runtime/data manifest before moving anything.
2. Move source audio, Telegram exports, external worker results, root Excel/DB exports and `_local_archive_*` out of the code repo tree or into a mounted data volume.
3. Keep `stable_runtime` as a controlled runtime volume, not a general archive.
4. After coverage/import gates, archive old `.before_*` DBs and historical benchmark DBs.
5. Clean safe generated files only after confirmation: `__pycache__`, `.DS_Store`, `.pytest_cache`, selected cache dirs.

P3 - close test gaps:

1. Add direct tests for `clients.amocrm`, `clients.ollama`, `utils.phone`, `llm_response_cache`.
2. Add `TestClient` coverage for `routers/deals.py`, `routers/integrations.py`, `routers/tallanto.py`, including auth and writeback refusal/defaults.
3. Add GUI command-builder tests without Tk mainloop where possible.
4. Add safe-mode success/failure/dead-letter tests for legacy AMO sync.
5. Add negative data tests for `insights` report generators.

## What Was Not Done

- No files were deleted.
- No code was changed outside this audit document.
- No runtime DB/audio/transcripts were modified.
- No ASR/R+A/batch/start/run-ui scripts were run.
- No AMO/Tallanto/CRM writes were made.
- Full pytest was not run; only collection was run with `-p no:cacheprovider`.
