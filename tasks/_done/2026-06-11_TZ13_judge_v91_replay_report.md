# TZ-13 judge v9.1 + replay mode report

Date: 2026-06-11
Branch: `codex/preserve-wave6-profile-final-dirty`
Commit: `525f3128` (`Add judge v9.1 and dynamic replay mode`)

## Scope

Implemented `tasks/_inbox_codex/2026-06-10_TZ13_judge_v91_replay.md`.

- Added judge prompt version `v9.1` with calibrated hard-gate rules.
- Added `--replay-from` mode to `scripts/run_telegram_dynamic_client_sim.py`.
- Updated offline rejudge sidecar script to write `judge_results_v91.jsonl`.
- Updated docs in `CLAUDE.md` and `README.md`.

Bot runtime logic was not changed.

## Formal tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py -k 'judge_v9 or judge_v91 or replay_from or rejudge_v9'
13 passed, 81 deselected
```

Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
2920 passed, 5 skipped, 1 warning
```

## Smoke: canary10 rejudge

Input transcripts:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_wave6_profile_canary10_5d2aa969_fast/dynamic_dialog_transcripts.jsonl
```

Output:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260611_tz13_canary10_rejudge_v91/judge_results_v91.jsonl
```

Result:

```json
{"dialogs": 10, "llm_calls": {"judge": 10}, "ok": true}
```

## Smoke: smoke18 replay

Source transcripts:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260610_final_smoke18_predpilot/dynamic_dialog_transcripts.jsonl
```

Output directory:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260611_tz13_replay_smoke18_v91
```

Runner result:

```json
{
  "dialogs": 18,
  "turns": 40,
  "pass": 13,
  "pass_with_notes": 5,
  "fail": 0,
  "hard_gate_failures": 0,
  "ok": true
}
```

Technical validity from `dynamic_summary.json`:

```json
{
  "replay": true,
  "run_config.replay": true,
  "judge_prompt_version": "v9.1",
  "judge_prompt_version_id": "judge_v9_1_pilot_calibrated",
  "config_validity.invalid": false,
  "llm_calls.client": 0,
  "llm_calls.judge": 18,
  "llm_calls.bot_direct_draft": 38,
  "llm_calls.bot_retriever": 38,
  "llm_calls.bot_semantic_output_verifier": 43
}
```

## Notes

This is a formal/infrastructure pass. Semantic review of PASS/PASS_WITH_NOTES content remains with the architect.
