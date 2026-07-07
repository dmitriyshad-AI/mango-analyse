# 2026-07-08 Codex skills top5 report

## Scope

Implemented read-only workflow scripts from `Foton/2026-07-08_TZ_skilly_top5_dlya_Codex2_i_Claude_M4.md`.

S1 is intentionally not implemented here: owner installs `mango-regrade` and `mango-semantic-safety` on M4.
S7 was not implemented; S2-S6 were completed first.

## Added tools

- `scripts/skills/tz_lint.py` - TZ lint before taking work.
- `scripts/skills/inventory_before_build.py` - existing-work inventory before building a feature.
- `scripts/skills/live_truth.py` - read-only runtime PID/worktree/HEAD/env/DB sentinel.
- `scripts/skills/fail_raw_export.py` - raw evidence export for FAIL rows.
- `scripts/skills/wappi_draft_loop_replay.py` - read-only Wappi draft-loop replay gate.

## Reuse discipline

- TZ header parsing reuses `scripts/preflight.py::parse_tz_header`.
- PII masking reuses `scripts/make_audit_pack.py::mask_pii`.
- Inventory wraps `build_project_inventory`, `graphify_structural_query.py`, and `git log -S`.
- Live/Wappi checks reuse `scripts/wappi_draft_loop_ops.py`.

No second parser, scanner, or PII masker was introduced.

## Demonstrations

Outputs are in this folder:

- `s2_tz_lint_demo.txt` - real Foton TZ lint: PASS.
- `s3_inventory_demo.json` - inventory for `memory step guard`: FOUND.
- `s4_live_truth_demo.json` - current runtime process scan: PASS.
- `s5_fail_raw_export/` - exported 1 FAIL from real `tz137` run; phone/email grep returned 0.
- `s6_wappi_replay_demo.json` - real Wappi-derived set: FAIL on missing chat->card mapping, STOP-file guard PASS.

The S6 FAIL is expected and useful: the selected Wappi what-if set is not a draft-loop apply-ready replay because it lacks AMO lead/contact mapping.

## Tests

- Targeted: `22 passed`.
- Full: `4128 passed, 5 skipped, 2 warnings`.

## Live/write safety

AMO/Tallanto/CRM/live/client sends: 0.
All scripts are read-only except writing local reports/snapshots.
