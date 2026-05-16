# Deal-Aware Stage20 Status And Next Steps — 2026-05-13

## Current Result

Deal-aware pipeline reached the first controlled production writeback stage.

Completed:

- Stage 1-5 chain audit: Claude PASS.
- Stage 6 preflight: Claude PASS.
- Stage20 live writeback: 20/20 AMO deals written.
- Stage20 readback gate: passed=true, evaluated_rows=20, blocking_rows=0, risk_counts={}
- Stage20 post-live Claude audit: PASS.

Artifacts:

- Stage6 runtime: `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_v1/`
- Live report: `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_v1/stage20_live_report/summary.json`
- Readback report: `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_v1/readback_after_stage20/summary.json`
- Claude post-live audit: `audits/_results/2026-05-13_deal_aware_stage20_post_live/CLAUDE_REAUDIT_RESULT.md`

Important implementation fixes made during Stage20:

- Load AMO env before importing AMO integration modules in deal-aware writer/readback scripts.
- Convert AMO `date` / `date_time` field values to Unix timestamps before writing.
- Treat ISO date-time and AMO timestamp as equivalent in readback gate.

## Current Constraints

- Do not run broad live writeback yet.
- Full 698 dry-run rows are not automatically approved for live.
- 18 Stage6 rows remain blocked by live-level text quality conflicts:
  - `completed_payment_next_step_conflict`: 17
  - `cross_field_duplicate_information`: 1
- Next live batches require the same staged pattern: preflight -> Claude audit -> operator approval -> live write -> readback -> post-live audit.

## Next Big Stages

### 1. Human/ROP Spot Check Of Stage20

Goal: confirm that deal-level AI fields are useful in real AMO cards, not only technically written.

Check:

- 5-10 of the 20 written deals.
- Whether deal summary is understandable.
- Whether next step fits the actual deal status.
- Whether text belongs to the deal, not only to the contact.
- Whether fields do not duplicate each other too much.

Exit criterion:

- No severe product-quality issue.
- ROP says the information is useful enough to expand to 100.

### 2. Stage100 Deal-Aware Live Batch

Goal: write a larger controlled batch after Stage20 passes human review.

Plan:

- Build next 100 candidates from the 698 dry-run-allowed rows, excluding already written Stage20 and blocked rows.
- Run dry-run/preflight.
- Build nano audit pack for Claude.
- Claude PASS required.
- Live write 100.
- Immediate readback gate.
- Post-live Claude audit.

Exit criterion:

- 100/100 written.
- readback passed=true.
- no P0/P1/P2 product-quality issue from ROP/Claude.

### 3. Fix The 18 Blocked Rows As Classes

Goal: reduce false blocks without allowing real bad rows.

Classes:

- Payment terms vs actual completed payment: price quote/payment condition must not look like paid deal.
- Duplicate text: expected repeated next step in history should not always be a hard block.

Approach:

- Analyze all 18 blocked rows.
- Split true blockers vs false positives.
- Update class-level detector logic, not literal phone/deal IDs.
- Add regression tests.

Exit criterion:

- False positives downgraded to warning/manual-review.
- True payment conflicts remain blocked.

### 4. Deal-Aware Batch Progression To Remaining Safe Rows

Goal: safely cover all currently eligible deal-aware rows.

Plan:

- After Stage100: proceed by 200/300/remaining batches depending on quality.
- Every batch requires readback green before the next batch.
- If any readback mismatch appears, stop batch progression.

Exit criterion:

- All safe deal-aware rows written or consciously blocked/manual-review.

### 5. Contact-Layer Refresh After Deal-Layer Stabilizes

Goal: remove duplication between contact and deal fields.

Plan:

- Contacts should contain global client history/status.
- Deals should contain concrete deal state and action.
- Refresh contact AI fields only after deal writeback proves stable.

Exit criterion:

- Manager can read contact for global context and deal for action without repeated text.

### 6. Multi-Source Customer Timeline

Goal: move beyond calls-only context.

Sources:

- AMO deal events/tasks/notes.
- Tallanto payments, groups, visits, subscriptions.
- Telegram and email later.

Exit criterion:

- Deal-aware next step considers commercial events, not only call analysis.
