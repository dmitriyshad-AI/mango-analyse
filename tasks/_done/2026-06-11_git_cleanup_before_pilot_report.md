# Git cleanup before pilot — 2026-06-11

## Scope

Goal: make the main project folder use canonical branch `main`, reduce merged branch/worktree clutter, and avoid `force` / `git clean`.

No live AMO/Wappi/Telegram actions were run. No runtime artifacts were deleted.

## Initial inventory

Worktrees before cleanup:

| path | branch | head | status |
|---|---|---:|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `codex/tz12-d4-client-history-profile` | `8327d37a` | dirty |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.tz13-main` | `main` | `a6aac3b6` | untracked `runs/` |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.wave1-probe` | `codex/wave1-number-scope-handoff` | `c84a16cf` | untracked `runs/` |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor` | `codex/refactor-subscription-llm` | `23336e3b` | untracked plan file |
| `/private/tmp/mango_bundle_build_8b48f196` | detached | `8b48f196` | clean |

Dirty tracked files in the main folder before switch:

- `CLAUDE.md` — local D4-folder edit was older than `main` for the judge line (`v9` vs canonical `v9.1`).
- `tasks/_done/2026-06-10_TZ7_route_rubric_report.md` — local report extension with the night rubric smoke.

Blocking untracked checkout collision:

- `tasks/_done/2026-06-11_memory_provenance_profile_enable_report.md` — byte-identical to the tracked file already present on `main`.

The full untracked set in the main folder was intentionally left in place. Main groups visible in `git status`:

- `runs/`, `_bundles/`;
- `D1_audit_backlog/*` new audit/intake materials;
- old local KB v6.4 schedule copies under `product_data/knowledge_base/`;
- `product_data/process_playbooks_20260609/`;
- local scripts for questions/process coverage;
- `tasks/_inbox_codex/*` and several untracked `tasks/_done/*` reports;
- local finance / Tallanto / Google Sheets artifacts.

## D4 / TZ-12 track

`codex/tz12-d4-client-history-profile` is already merged into `main`.

Evidence: `git merge-base --is-ancestor codex/tz12-d4-client-history-profile main` returned `0`.

The tracked report `tasks/_done/2026-06-11_TZ13_D4_merge_and_followups_report.md` also states:

- `dc6ee1ec` — `Merge TZ12 client history profile into main`;
- follow-up commits `02dd2ae6`, `64b0fd32`, `07c6947d`, `77732520`;
- full pytest after merge was green at that time.

No extra D4 merge was needed.

## Actions performed

1. Detached `/Users/dmitrijfabarisov/Projects/Mango analyse.tz13-main` from branch `main` to free the branch for the main folder. Its untracked `runs/` directory was preserved.
2. Tried normal `git switch main` in the main folder. Git blocked on `CLAUDE.md` and the untracked memory report, so no overwrite happened.
3. Preserved only those two blocking files in an explicit stash:
   - `stash@{0}: On codex/tz12-d4-client-history-profile: pilot-cleanup-conflicting-local-files-20260611`
4. Switched the main folder to `main` successfully.
5. Removed clean temporary bundle worktree:
   - `/private/tmp/mango_bundle_build_8b48f196`
6. Tried to remove `.tz13-main` and `.wave1-probe` without force. Git refused because both contain untracked `runs/`; they were not removed.
7. Ran `git worktree prune`.
8. Deleted merged local branches with safe `git branch -d`.

Deleted merged branches:

- `codex/audio-working-store-20260523`
- `codex/claude-cli-visible-error-20260605`
- `codex/consolidation-clean-type-20260605`
- `codex/d1-semantic-roles-20260525`
- `codex/g2scope-phase1-20260605`
- `codex/git-order-20260513`
- `codex/gpt-g2p1-reggrade-fixes-20260605`
- `codex/kb-v6_4-20260529`
- `codex/levelA-travel-fix-20260604`
- `codex/m1-intermediate-20260529-final`
- `codex/output-sanitizer-20260605`
- `codex/phase12-p0-20260529`
- `codex/phase2-humanity-20260605`
- `codex/presale-safety-fixes`
- `codex/project-cleanline-20260604`
- `codex/project-cleanline-main-20260604`
- `codex/quality-roadmap-block1-20260604`
- `codex/saas-productization-baseline`
- `codex/semantic-diagnosis-guard-20260605`
- `codex/semantic-output-verifier-20260606`
- `codex/step0-trace-deferral-reasons-20260605`
- `codex/telegram-native-draft-mvp`
- `codex/travel-handoff-trace-20260605`
- `codex/tz12-d4-client-history-profile`
- `kb-v6_4`

## Remaining worktrees

| path | branch/state | head | why remains |
|---|---|---:|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `main` | `a6aac3b6` | canonical main folder |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor` | `codex/refactor-subscription-llm` | `23336e3b` | has untracked `D1_audit_backlog/PLAN_refactor_subscription_llm_2026-06-11.md`; not removed without owner decision |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.tz13-main` | detached | `a6aac3b6` | has untracked `runs/`; `git worktree remove` refused without force |
| `/Users/dmitrijfabarisov/Projects/Mango analyse.wave1-probe` | `codex/wave1-number-scope-handoff` | `c84a16cf` | has untracked `runs/`; `git worktree remove` refused without force |

## Remaining local branches

| branch | head | merged into main | date | last commit |
|---|---:|---|---|---|
| `codex/d1-humanity-layer-20260525` | `fd6f56ba` | no | 2026-06-06 | `Pre-cleanup snapshot` |
| `codex/d7-claude-first-class-brain` | `fa956a78` | no | 2026-06-04 | `Use Claude subscription mode for D7 runner` |
| `codex/overhandoff-G1G2G3-20260604` | `9054f5e6` | no | 2026-06-04 | `Fix over-handoff v2 leaks and partial yield` |
| `codex/refactor-subscription-llm` | `23336e3b` | yes | 2026-06-11 | `Report memory provenance pilot profile smoke` |
| `codex/tone-score-metric-20260605` | `99613fa2` | no | 2026-06-05 | `Add deterministic tone metric to dynamic summary` |
| `codex/wave1-number-scope-handoff` | `c84a16cf` | yes | 2026-06-08 | `Add wave1 scoped number gate and handoff verifier flag` |
| `codex/wave2-input-understanding` | `aecd1c72` | no | 2026-06-08 | `Merge branch 'main' into codex/wave2-input-understanding` |
| `codex/wave3-keep-grounded` | `b800fa87` | no | 2026-06-09 | `Soften direct Wave 3 negative alternative text` |
| `codex/wave4-graded-gate` | `d80f8ed6` | no | 2026-06-09 | `Add direct graded gate for low-risk findings` |
| `codex/wave5-anti-promise` | `b203b637` | no | 2026-06-09 | `Add direct path anti-promise wave5 guards` |
| `codex/wave6-llm-retriever` | `7389d07a` | no | 2026-06-09 | `Add flagged LLM fact retriever for direct path` |

Merged-but-remaining branches:

- `codex/refactor-subscription-llm` — checked out in a worktree with untracked file.
- `codex/wave1-number-scope-handoff` — checked out in a worktree with untracked `runs/`.

Unmerged branches were not deleted.

## Tests

Full pytest on `main` in the main folder:

```text
3007 passed, 2 skipped, 1 warning in 39.01s
```

Warning: `urllib3` reports LibreSSL/OpenSSL compatibility warning from the local Python environment.

## Current status after cleanup

Main folder is on `main` at `a6aac3b6`, ahead of `origin/main` by 409 commits.

Tracked state before this report commit:

- modified `tasks/_done/2026-06-10_TZ7_route_rubric_report.md` with the night local rubric measurement;
- this cleanup report.

Untracked runtime/audit artifacts remain untouched.

## Follow-up decisions needed

1. Decide whether to archive or keep untracked `runs/` in `.tz13-main` and `.wave1-probe`; only after that can those worktrees be removed without force.
2. Decide whether to keep or archive `codex/refactor-subscription-llm` worktree and its untracked plan file.
3. Decide what to do with unmerged historical branches:
   - `codex/d1-humanity-layer-20260525`
   - `codex/d7-claude-first-class-brain`
   - `codex/overhandoff-G1G2G3-20260604`
   - `codex/tone-score-metric-20260605`
   - `codex/wave2-input-understanding`
   - `codex/wave3-keep-grounded`
   - `codex/wave4-graded-gate`
   - `codex/wave5-anti-promise`
   - `codex/wave6-llm-retriever`

