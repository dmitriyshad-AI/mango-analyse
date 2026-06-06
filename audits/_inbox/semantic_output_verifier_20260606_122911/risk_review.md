# Risk Review

Hard-safety risks:
- P0, brand, number and meta gates are not replaced by the verifier. They still run in `apply_authoritative_output_gate` after the verifier and after one optional regen.
- `downgrade_keep_text` does not set provider `error`, to avoid misclassifying semantic downgrades as runtime failures.
- The verifier is fail-soft in draft mode: one retry, then annotate-only. This matches Дмитрий's decision for current pilot.

Quality risks:
- False positives can increase `draft_for_manager` via `downgrade_keep_text`; summary exposes `downgrade_budget_turns` with deterministic dedupe.
- False negatives remain possible because this is a model classifier; live regrade is required before enabling the flag.

Operational risks:
- New runner model defaults to Codex/medium through `--semantic-verifier-*` and env `TELEGRAM_SEMANTIC_VERIFIER_MODEL`.
- When flag is OFF, the model object can exist in runner but produces zero calls because the verifier path is not executed.

Known non-goals:
- No change to faithfulness inner-loop.
- No change to P0 floor.
- No change to output number grounding.
