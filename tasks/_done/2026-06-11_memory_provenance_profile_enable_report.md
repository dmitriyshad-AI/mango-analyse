# Memory provenance profile enable — 2026-06-11

## Scope

- Commit: `6c528c79` (`Enable memory provenance in pilot profile`).
- Change: `TELEGRAM_MEMORY_PROVENANCE` is now part of `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- Explicit override remains supported: `TELEGRAM_MEMORY_PROVENANCE=0` under `pilot_gold_v1` restores the old LLM-memory path.
- Bot logic outside memory-profile activation was not changed.

## Tests

- Targeted NEG/profile tests: `8 passed`.
- Full pytest: `2999 passed, 5 skipped, 1 warning`.

Covered checks:
- `pilot_gold_v1` enables provenance memory without an explicit env flag.
- Explicit `TELEGRAM_MEMORY_PROVENANCE=0` restores old memory model construction.
- Runner summary reports `run_config.key_flags.memory_provenance`.
- Draft loop and dialogue memory share the same profile-default / explicit-override behavior.

## Smoke18

- Run directory: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260611_memory_provenance_profile_smoke18`
- Scenario set: `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`
- Snapshot: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json`
- Profile: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- Judge: `v9.1`
- Parallel: `4`
- Temp `CODEX_HOME`: `/private/tmp/mango_codex_home_memory_profile_20260611_fast`, with `service_tier=fast`.

First full pass had one infrastructure timeout:
- `pilot_smoke18_07_foton_promocode`
- `TimeoutExpired` from `codex exec` after 180 seconds.
- Before retry, config was valid and calls included `memory=0`, `bot_direct_draft=40`, `bot_retriever=40`, `bot_semantic_output_verifier=42`.

Retry used `--resume --only-failed` and reran only `pilot_smoke18_07_foton_promocode`.

Final smoke result after retry:
- `dialogs=18`
- `turns=42`
- `PASS=7`
- `PASS_WITH_NOTES=11`
- `FAIL=0`
- `hard_gate_failures=0`
- `config_validity.invalid=false`
- `infra_error_dialogs=[]`

Final `run_config.key_flags`:
- `profile.effective=true`, `env=pilot_gold_v1`
- `render.effective=true`
- `rubric.effective=true`
- `retriever.effective=true`
- `memory_provenance.effective=true`, `env=""`
- `snapshot=product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json`

Note: after `--resume --only-failed`, `dynamic_summary.llm_calls` reflects the retry pass counter, not a full aggregate over all 18 dialogs. The pre-retry full pass and retry both had `memory=0`; direct draft, retriever and verifier calls were non-zero.

## P0 Control

`pilot_smoke18_13_foton_p0_payment`:
- verdict: `PASS`
- hard gates: passed
- T1 route: `manager_only`, `preblocked=true`, reason `p0_deferral`
- T2 route: `manager_only`, `preblocked=true`, reason `p0_deferral`
- safety flags include `direct_path_preblocked_p0` and `payment_dispute_manager_only`.

## Status

`formal_pass`: yes.

Semantic/regression verdict over smoke transcripts remains with the architect, per task instruction.
