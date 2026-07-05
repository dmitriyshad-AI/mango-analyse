# ADR003 final approach completion audit

Status: not complete yet.

Current branch: `codex/adr003-semanticframe-migration`

Current HEAD source of truth: `git rev-parse HEAD` at package build time. The M1 package must carry the exact source in `SOURCE_HEAD.txt`.

## Objective Being Audited

Finish the ADR003 final approach:

1. Atomic profile enablement of `intent_actions`.
2. Deletion №1.
3. Prepare/check remaining Ж-steps without live/push and with local measurements.

## Requirement Audit

| Requirement | Evidence inspected | Status |
|---|---|---|
| `intent_actions` is in pilot profile defaults | `src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py`; commit `0a68e285`; `docs/ADR003_ETAP_T_DECISIONS.md` D-032 | Done |
| Deletion №1 completed | `docs/ADR003_ETAP_T_DECISIONS.md` D-032; tests around `intent_actions`; current code no longer applies the removed output live-availability branch in the guarded path | Done |
| Ж2 `live_status_read` apply is built but not profile-enabled | `src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py`; `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`; D-031 | Done as default-OFF |
| Ж2 local measurement exists | local micro-pair referenced by Fable/D1 reports; final local dry-check `runs/adr003_final_livestatus_dry_add0a873_20260705_202109` proves runner attribution only | Partially done: local checks pass, full M1 pair pending |
| Ж3 `reask_read` checked | `docs/ADR003_ETAP_T_DECISIONS.md` D-033; Foton export `2026-07-05_Zh3_reask_read_export_6896673b` | Done: NO-GO, remains trace-only |
| Ж4 `roles_read` checked | `docs/ADR003_ETAP_T_DECISIONS.md` D-034; Foton export `2026-07-05_Zh4_roles_read_export_6896673b` | Done: NO-GO, remains trace-only |
| Final M1 package prepared for the only remaining safe class | Latest package folder named `adr003_final_livestatus_pair_<source-sha>_20260705`; `SHA256SUMS.txt`; bundle verify; `PROMPT_M1.md` | Done |
| Final M1 package uses current HEAD | Proved by `SOURCE_HEAD.txt` matching bundle HEAD inside the latest package | Done |
| Final M1 package attribution is locally rehearsed | dry-check validation: ON `required_trace_turns_by_class.live_status_read=2`; B `forbidden_trace_turns_by_class.live_status_read=0` | Done for dry-check only |
| Full final M1 pair completed | No `runs/`, `REPORT`, `sha_manifest.json`, or M1 output found inside package at audit time | Missing |
| Raw regrade of final M1 pair completed | No M1 full pair result yet | Missing |
| Atom №2 applied after M1 regrade and explicit approval | `docs/ADR003_ATOMIC2_LIVE_STATUS_READ_PREP_20260705.md` is a gated plan only; runtime code not changed | Pending by design |
| No live/push/runtime writes | `git status` clean; no live commands run in this step; package/dry-check only | Satisfied for current work |

## Why This Goal Is Not Complete Yet

The local work has reached a safe handoff point, but completion is not proven because the full final M1 pair has not returned. The next runtime-affecting step, atom №2, is explicitly gated on:

1. Full M1 pair for `live_status_read`.
2. Raw regrade of route/text/safety diffs.
3. Dmitry explicit approval.

Until those three conditions are satisfied, `live_status_read` must not be added to profile defaults and `_asks_live_availability` must not be removed from the normal path.

## Next Action

Wait for M1 output for package:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_final_livestatus_pair_<source-sha>_20260705`

When M1 returns:

1. Verify `SOURCE_HEAD` / bundle / `sha_manifest`.
2. Inspect `VALID_E3_ON`, `VALID_E3_B`.
3. Inspect `REPORT/adr003_semantic_frame_eval_report.json`.
4. Regrade raw diffs for:
   - P0 route lost;
   - live-seats questions;
   - paid/no-place and paid/no-access;
   - brand/fact/number gates;
   - trace attribution.
5. Only after green regrade and explicit approval, implement atom №2 using `docs/ADR003_ATOMIC2_LIVE_STATUS_READ_PREP_20260705.md`.
