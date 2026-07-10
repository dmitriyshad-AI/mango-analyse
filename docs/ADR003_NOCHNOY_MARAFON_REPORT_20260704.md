# ADR-003 Night Marathon Report 2026-07-04

Source TZ: `tasks/_done/2026-07-04_TZ_nochnoy_marafon_adr003_shest_blokov.md`

Scope: local implementation and local measurements only. No M1 handoff, no live bot restart, no profile deployment, no push, no AMO/Tallanto/CRM writes.

## Result

All six requested blocks were completed on `codex/adr003-semanticframe-migration`.

Block commits:

1. `a584542c` - `feat(adr003): enable semantic reading defaults in pilot profile`
2. `5ef4cfdf` - `feat(adr003): add verified-fact autonomy corridor`
3. `663fcb78` - `test(adr003): inventory semantic slots reask`
4. `0533ec2e` - `feat(adr003): add rolling dialog summary`
5. `f570b7bf` - `feat(adr003): gate payment hygiene wording fix`
6. `a2794361` - `docs(adr003): inventory package 2 regex slices`

Final local formal check after Block 6:

```text
4056 passed, 5 skipped, 1 warning in 83.78s
```

## Block Notes

### Block 1 - Profile Defaults

Semantic reading classes are wired into the pilot profile default path behind the existing explicit-env precedence. No live profile was deployed.

### Block 2 - PR-A Fix1b

Added the verified-fact autonomy corridor behind the default-OFF flag. The corridor is limited to fully verified, client-safe fact use and keeps stop tests for partial support, extra numbers, wrong brand, and live availability.

Audit pack:

```text
audits/_inbox/adr003_pr_a_fix1b_verified_facts_20260704043106
```

### Block 3 - PR-B Slots Reask Inventory

Added the inventory/test surface for semantic slot re-ask behavior. No production routing was changed by this block.

Audit pack:

```text
audits/_inbox/adr003_pr_b_slots_reask_inventory_20260704044234
```

### Block 4 - PR-D Rolling Dialog Summary

Added the rolling dialog summary candidate path behind its default-OFF flag and connected it through the public pilot script surfaces.

Local measurements:

```text
/tmp/adr003_prd_dialog_summary_fake_smoke_20260704_045937
/tmp/adr003_prd_dialog_summary_direct_on_20260704_050319
/tmp/adr003_prd_dialog_summary_direct_off_20260704_050405
```

One real-model pair at `/tmp/adr003_prd_dialog_summary_real_smoke_20260704_050049` was rejected as infrastructure/auth noise (`codex_retryable_error` / `llm_fallback`) and was not used for semantic decisions.

Audit pack:

```text
audits/_inbox/adr003_pr_d_dialog_summary_20260704050641
```

### Block 5 - Payment Hygiene Fix

Added `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` as a default-OFF output-hygiene flag. It narrows payment/refund wording so payment-link or payment-operation context does not fall into refund wording, while true refund wording remains protected.

Targeted check:

```text
34 passed in 3.37s
```

Full check after Block 5:

```text
4056 passed, 5 skipped, 1 warning in 82.83s
```

Audit pack:

```text
audits/_inbox/adr003_block5_payment_hygiene_fix_20260704052330
```

### Block 6 - Package 2 Inventory

Created `docs/ADR003_PACKAGE2_INVENTORY.md` as a data-only inventory for the next regex-slice package. Graphify was used only for navigation; every count in the document is grounded in the frozen direct-path regex snapshot at the relevant commit.

Count summary:

```text
answer_quality_rewriter.py: 133
conversation_intent_plan.py: 55
policy_routing.py: 276
post_layers.py: 208
semantic_roles.py: 35
new_lead_funnel.py: 39
dialogue_memory.py: 83
total: 829
```

Audit pack:

```text
audits/_inbox/adr003_block6_package2_inventory_20260704053600
```

## Boundaries Kept

- Live Telegram bot was not restarted or touched.
- Wappi was not touched.
- M1 was not used.
- No push was performed.
- No AMO/Tallanto/CRM writes were performed.
- No runtime data cleanup or destructive git operation was performed.

## Remaining Acceptance

This report records local implementation and formal local checks. Final semantic acceptance remains with the external raw-transcript/regression review requested in the TZ. The changes are not declared production-ready by this report.
