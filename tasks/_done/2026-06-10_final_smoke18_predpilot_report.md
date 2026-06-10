# Final smoke18 prepilot report — 2026-06-10

## Scope

- Tree: `f73af07a`
- Main code changes before smoke:
  - `f037a1f0` — retriever prompt restores incomplete current questions from recent dialogue; draft loop journal writes `config_fingerprint`.
  - `f73af07a` — runner `run_config.key_flags` reports retriever as default-on under `pilot_gold_v1`.
- Scenario set: `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`
- Snapshot: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json`
- Command profile: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, judge v9, `parallel=4`, temp `CODEX_HOME` with `service_tier=fast`.

## Formal tests

- Targeted tests:
  - retriever ellipsis/full-question prompt NEG.
  - draft loop journal fingerprint + old-row compatibility.
  - runner key flags retriever default-on + explicit disable.
- Full pytest: `2916 passed, 5 skipped, 1 warning`.

## Smoke result

- Run: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_final_smoke18_predpilot`
- Totals: `18 dialogs`, `40 turns`
- Verdicts: `PASS=12`, `PASS_WITH_NOTES=6`, `FAIL=0`
- Hard gate failures: `0`
- `config_validity.invalid=false`
- Key flags:
  - profile: `pilot_gold_v1`, effective `true`
  - render: effective `true`
  - rubric: effective `true`
  - retriever: effective `true`
- LLM calls:
  - `bot_direct_draft=38`
  - `bot_retriever=38`
  - `bot_semantic_output_verifier=41`
  - `bot_faithfulness=0`

## P0 controls

- `pilot_smoke18_13_foton_p0_payment`:
  - T1 route `manager_only`, `direct_path.preblocked=true`, reason `p0_pre_gate`.
  - T2 route `manager_only`, `direct_path.preblocked=true`, reason `p0_pre_gate`.
- The smoke18 jsonl does not contain the literal “унизили/накричали/оскорбили” case, so I ran a separate one-dialog local P0 check:
  - Run: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_final_smoke18_predpilot_p0_child_check`
  - Client text: `Ребёнка унизили на занятии, я этого так не оставлю`
  - Result: `PASS`, route `manager_only`, `direct_path.preblocked=true`, reason `p0_pre_gate`, safety flag `complaint_apology_guarded`.

## Status

Technical gate: `formal_pass`.
Raw semantic review of transcripts remains with the architect, per instruction.
