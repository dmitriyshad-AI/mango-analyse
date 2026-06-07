# Judge v9 calibration

Base before change: `a34a1345`.

Implemented according to `D1_audit_backlog/TZ_judge_v9_calibration_2026-06-08.md`:

- Added `--judge-prompt-version {v2,v9}` to `scripts/run_telegram_dynamic_client_sim.py`; default remains `v2`.
- Added separate prompt ids: `JUDGE_PROMPT_VERSION_V2=judge_v2_current` and `JUDGE_PROMPT_VERSION=judge_v9_verifier_aware`; `JUDGE_FACT_AUDIT_VERSION` was not changed.
- Added judge v9 matrix: semantic verifier/gate context is visible to the judge, final route controls hard vs soft classification, and `derived_claim_draft` is a review-priority soft flag.
- Added judge gate re-ask for v9 only. It fills missing `violated_gates` for completed FAIL dialogs without reconsidering the verdict.
- Fixed hard-gate text inference patterns, including literal `p0_mishandled` family.
- Added judge-only broad fact-claim extraction via `include_judge_generic_claims`; runtime callers keep the default narrow extractor.
- Added `scripts/rejudge_dynamic_transcripts_v9.py`, which reads saved turn fields and writes a sidecar `judge_results_v9.jsonl` without re-attaching current context facts.
- Added summary fields: `run_config.judge_prompt_version`, `run_config.judge_prompt_version_id`, `derived_claim_draft.count`, and `judge_parse_issues`.
- Added measurement discipline note to `CLAUDE.md`: runs with different judge prompt versions must not be compared without re-judging both sides with the same judge.

Entry-test confirmations:

- `build_judge_prompt` is the judge prompt construction point.
- `_infer_failed_hard_gates` is the existing hard-gate inference fallback.
- `judge_spec` is still included unchanged; v9 prompt rules explicitly take priority over it.
- `fact_claim_audit.audit_fact_claims` is shared with runtime gates, so the broader extraction is opt-in for judge v9 only.
- `--transcripts-in` still re-attaches context facts, so a separate offline sidecar re-judge script is required.

Activation:

- v9 is implemented but not activated by default.
- No saved production/regression runs were re-judged in this block.
