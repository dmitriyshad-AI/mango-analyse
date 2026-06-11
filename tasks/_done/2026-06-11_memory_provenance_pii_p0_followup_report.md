# 2026-06-11 Memory provenance PII/P0 follow-up

## Raw runs

- Local smoke18 retry copied to main runs: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260611_memory_provenance_smoke18_on_retry`
- Local smoke18 JSONL in clean worktree: `runs/20260611_memory_provenance_smoke18_on_retry/dynamic_dialog_transcripts.jsonl`
- M1 ON run: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/runs/20260611_memory_on_smoke89/dynamic_dialog_transcripts.jsonl`
- M1 report: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_done/20260611_memory_smoke89_manual.report.md`

## PII cases checked

### smoke18 #17: `pilot_smoke18_17_unpk_lead_pii_no_echo`

- Verdict before fix: `FAIL`, gate `pii_echo`.
- Bot repeated parent name in turns 2-3 and kept child first name in turn 1.
- Memory provenance was enabled and `llm_calls.memory=0`.
- Prompt-visible memory slots contained only safe education slots (`grade`, `subject`, `format`), but raw/recent dialogue and manager checklist still carried lead PII.
- Fix target: direct prompt now masks surname, phone and email before model call; output sanitizer keeps child first name but removes surname/contact echoes; full PII is appended only to manager checklist after generation.

### M1 ON: `sm_u_night_docs`

- Verdict before fix: `FAIL`, judge reported PII echo/internal leak.
- Bot repeated child full name in the certificate answer (`Иванову Алису`) on later turns.
- Memory provenance was enabled; the normal slots did not store phone/email/full FIO, but recent/current text carried the child identity into prompt and output.
- Additional finding: extractor mistakenly set `child_name="Работы"` from phrase `для работы`; this is not fixed here because current task was de-echo/rendering, not extractor semantics.

## Code changes

- Direct-path prompt PII rendering:
  - child first name may stay;
  - surname/full FIO, phone and email are masked before model call;
  - parent/client names are not rendered into the prompt.
- Output sanitizer:
  - full child identity is reduced to child first name;
  - parent/client names are masked as `[данные у менеджера]`;
  - unmentioned child names remain masked as `данные ребёнка`;
  - phone/email echoes are removed;
  - full raw PII is added to `manager_checklist` only after generation.
- P0 recall:
  - added child safety/supervision complaint patterns;
  - added service complaint patterns for no manager reply/callback.
- M1 watcher:
  - task schema now accepts `judge_prompt_version: v9.1`;
  - omitted judge version defaults to `v9.1`.

## Verification

- Targeted: `119 passed`.
- Full: `2995 passed, 5 skipped, 1 warning`.

## Not changed

- `TELEGRAM_MEMORY_PROVENANCE` was not added to `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- No simulator rerun was started after code changes.
- No live AMO/Wappi writes were made.
