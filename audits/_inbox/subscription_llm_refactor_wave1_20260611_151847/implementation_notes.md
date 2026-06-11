# Subscription LLM refactor wave 1

Wave: 1, characterization only.

No runtime source code was moved or edited in `src/mango_mvp/channels/subscription_llm.py`.

Added harness artifacts:

- `scripts/check_subscription_llm_facade_exports.py`
- `scripts/check_subscription_llm_move_only.py`
- `scripts/run_subscription_llm_equivalence_replay.py`
- frozen export snapshot: `D1_audit_backlog/subscription_llm_refactor_exports_snapshot_2026-06-11.json`
- frozen move-only body snapshot: `D1_audit_backlog/subscription_llm_refactor_body_snapshot_2026-06-11.json`
- AST refresh/diff artifacts
- deterministic replay cases and baseline JSONL

Baseline replay covers 19 deterministic cases: legacy, fake provider, direct path, direct LLM retrieve, route rubric, dialogue-contract seam, P0 preblock, brand separation, payment confirmation, known-context no-reask, timeout, retryable/non-retryable rc, cache put-hit, stable_runtime cache guard, default gold pack path, valid_until, memory provenance, night note.

Baseline freeze status: frozen after deterministic repeat passed. Do not re-freeze after this wave starts.

No test imports were rewritten. Harness subclasses only use approved seam points: `_direct_path_draft_runner`, `_direct_path_llm_retrieve_runner`, `_build_dialogue_contract_pipeline_draft`.
