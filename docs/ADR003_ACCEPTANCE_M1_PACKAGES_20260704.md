# ADR-003 Acceptance M1 Packages 2026-07-04

Purpose: three independent paired M1 acceptance packages after the ADR-003 night marathon. Each package tests exactly one target flag over the current `pilot_gold_v1` profile so the effect is attributable.

No package changes live runtime, pilot profile, AMO, Tallanto, CRM, or Wappi. No push is required by this report.

## Common Runner

Runner:

```text
scripts/run_adr003_flag_acceptance_pair.sh
```

Leg contract:

- `B`: `pilot_gold_v1` profile, all three package target flags unset.
- `ON`: same profile plus exactly one target flag.
- `TELEGRAM_SEMANTIC_READING_CLASSES` is unset in process env so the profile default classes apply.
- The runner rejects target flags outside the allowlist:
  - `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS`
  - `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX`
  - `TELEGRAM_DIALOG_SUMMARY_ROLLING`

Both legs write `progress.json`, run `validate_adr003_e3_leg.py`, then build `REPORT/` and `sha_manifest.json`.

## Package 1 - Fix1b

Scenario set:

```text
product_data/telegram_dynamic_test_sets/adr003_acceptance_fix1b_canonical_plus10_20260704.jsonl
```

Composition:

- 156 personas total.
- Frozen copy of the current canonical ADR-003 semantic reading set: canonical 146 plus 10 Fix1b NEG personas.

Flag:

```text
TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS=1
```

Local dry-check:

```text
/tmp/adr003_acceptance_fix1b_dry_20260704
```

Dry-check result:

```text
B:  dialogs=2 turns=2 PASS=2 hard_gate_failures=0 eligible_frame_rate=1.0000 trace_turns=2
ON: dialogs=2 turns=2 PASS_WITH_NOTES=2 hard_gate_failures=0 eligible_frame_rate=1.0000 trace_turns=2
```

Note: dry-check is only a runner smoke. Full quality verdict belongs to the full M1 pair and raw-transcript regrade.

## Package 2 - PaymentFix

Scenario set:

```text
product_data/telegram_dynamic_test_sets/adr003_acceptance_paymentfix_20260704.jsonl
```

Composition:

- 20 personas total.
- 12 forward/payment/receipt/not-paid/presale controls.
- 8 true P0 NEG personas: paid refund, double charge, paid no access, paid lesson missing, receipt not credited, refund plus no access, legal refund threat, wrong amount.

Flag:

```text
TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX=1
```

Local dry-check:

```text
/tmp/adr003_acceptance_paymentfix_dry_20260704
```

Dry-check result:

```text
B:  dialogs=2 turns=4 PASS=1 FAIL=1 hard_gate_failures=1 eligible_frame_rate=1.0000 trace_turns=4
ON: dialogs=2 turns=4 PASS=1 PASS_WITH_NOTES=1 hard_gate_failures=0 eligible_frame_rate=1.0000 trace_turns=4
```

Semantic review note:

- The B hard failure on `payfix_foton_link_01` is expected baseline behavior for the target bug: the bot answers a payment-link request with a refund template.
- The ON leg removes the hard failure and replaces the refund wording with payment wording.
- A separate old issue remains in the same dry case: the second turn can close generically instead of giving a useful next step. That issue is not caused by `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` and is not claimed fixed by this package.
- Refund/dispute NEG personas are included specifically to prove real refunds and payment disputes stay manager-only on both legs.

## Package 3 - PR-D Rolling Dialog Summary

Scenario set:

```text
product_data/telegram_dynamic_test_sets/adr003_acceptance_prd_dialog_summary_20260704.jsonl
```

Composition:

- 24 personas total.
- 18 short canonical controls first, so `--dry-check --limit 2` is fast and stable.
- 6 long draft personas from `adr003_prd_dialog_summary_long_draft_20260704.jsonl`.

Flag:

```text
TELEGRAM_DIALOG_SUMMARY_ROLLING=1
```

Local dry-check:

```text
/tmp/adr003_acceptance_prd_dry_20260704_fast
```

Dry-check result:

```text
B:  dialogs=2 turns=4 PASS_WITH_NOTES=2 hard_gate_failures=0 eligible_frame_rate=1.0000 trace_turns=4
ON: dialogs=2 turns=4 PASS_WITH_NOTES=2 hard_gate_failures=0 eligible_frame_rate=1.0000 trace_turns=4
```

Implementation note:

- An earlier local PR-D dry-check started with two 10-turn long personas and was stopped because it was too slow for a mandatory dry-check. The scenario file was reordered so the full run still includes all six long personas, while the dry-check starts with short controls.

## Shared Local Validation

Existing E3 ON leg validator rehearsal:

```text
VALID_E3_ON: dialogs=156 turns=291 preblocked_p0=74 timeouts=0 timeout_turns=0 timeout_dialogs=0 model_not_called=89 model_called_eligible=202 frames=202 eligible_frame_rate=1.0000 bot_direct_draft=202 trace_turns=202 gate_blocked_turns=47
```

Targeted tests:

```text
50 passed in 0.58s
```

Covered tests:

```text
tests/test_adr003_flag_acceptance_pair_runner.py
tests/test_adr003_semantic_reading_e3_runner.py
tests/test_report_adr003_semantic_frame_eval.py
```

## Acceptance Boundary

These packages provide formal M1-ready inputs and local runner validation. They do not provide final semantic acceptance. Final acceptance requires full M1 pair outputs and raw-transcript regrade.
