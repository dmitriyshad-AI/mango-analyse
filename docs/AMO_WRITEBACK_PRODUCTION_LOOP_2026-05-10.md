# AMO Writeback Production Loop 2026-05-10

## Status

This document defines the controlled production loop for AMO contact writeback after the post-backfill CRM text quality work.

It is a runbook and policy document only. It does not authorize a live writeback by itself, does not replace the quality gates, and does not permit code or `stable_runtime` mutation.

## Scope

Covered:

- AMO contact writeback of the five AI manager-assist fields.
- Safe bucket lifecycle from strict CRM export to dry-run, live batch, readback, and next batch.
- Dry-run, live-write, and post-writeback readback gates.
- Bucket definitions and forbidden classes before stage50/full rollout.
- Operator checklist for stage50 and full writeback.

Not covered:

- Deal writeback.
- Bot/KB publication.
- CRM field schema changes.
- New AMO contact creation.
- Manual correction of source CRM/Tallanto records.

## Source Evidence

This loop is based on the current project evidence, especially:

- `audits/_results/2026-05-10_amo_post_backfill_writeback_v5_product_gate/CLAUDE_REAUDIT_RESULT.md`
- `audits/_results/2026-05-10_amo_writeback_f008_closure/CLAUDE_REAUDIT_RESULT.md`
- `audits/_results/2026-05-10_crm_text_quality_stage20/CLAUDE_REAUDIT_RESULT.md`
- `audits/_results/2026-05-10_crm_text_quality_stage69_preflight/CLAUDE_REAUDIT_RESULT.md`
- `audits/_results/2026-05-10_crm_writeback_defect_classes_v1/CLAUDE_DEFECT_CLASS_MAP.md`
- `audits/_inbox/amo_stage40_live20_readback_remaining20_20260510_v1/README.md`

Current facts from those reports:

- v5 reduced AMO-ready to 103 strict candidate rows, all `sales_call`, exactly one source `AMO contact IDs`, `CRM writeback policy=live_update_ready`, `AMO entity policy=update_existing_single_amo_contact`.
- Real-tunnel dry-run measured 103 rows as 86 `dry_run`, 17 safety skips, 0 failed. The 17 skips are not writeable until resolved upstream.
- Stage20 live writeback wrote 20/20, but Stage20 CRM text audit found Q1/Q4 blockers that must be fixed before stage50/full.
- Stage69 strict preflight fixed Q1-Q4/Q4b/Q4c at fail-live gate level and produced 69 live-ready strict rows, but required real-tunnel dry-run and readback before further live writeback.
- Stage40 package evidence says: quality gate passed for 40, real-tunnel dry-run passed for 40, first live20 passed, readback passed, and remaining20 dry-run passed. This still authorizes only the next staged batch, not a full unrestricted writeback.

## Production Principles

1. Fail closed.
   Any missing gate, stale summary, input mismatch, unsupported status, ambiguous AMO contact, or readback blocker stops live writeback.

2. Buckets are monotonic.
   A row can move forward only by passing the next gate. A row that becomes ambiguous or unsafe moves to manual review/quarantine and cannot re-enter without a fresh source artifact and fresh gates.

3. Live writeback is staged.
   No audit result authorizes collapsing stage20, stage50, stage69, stage86, or full population into one live run.

4. Dry-run must use real AMO lookup before live.
   Offline preview is useful for payload shape, but it does not prove contact matching.

5. Readback is mandatory after every live batch.
   AMO-stored values, not only local preview payloads, must be fetched and checked before unlocking the next batch.

6. Only five fields are write targets.
   Allowed target fields are:
   - `Статус матчинга`
   - `AI-приоритет`
   - `AI-рекомендованный следующий шаг`
   - `Последняя AI-сводка`
   - `Авто история общения`

7. Protected fields are never write targets.
   `Id Tallanto`, `ID Tallanto`, `Филиал Tallanto`, source AMO IDs, phones, names, and Tallanto metadata can be read for checks but must not be sent as AMO update fields.

## Safe Bucket Lifecycle

### Bucket 0: Source Population

Definition: full post-backfill master/contact population before AMO live eligibility.

Required evidence:

- Source is post-backfill canonical DB and post-backfill phone-chain CSV.
- No dependency on old April export pointer.
- Stage15 summary exists with `passed=true` and `readiness.crm_quality_writeback_ready=true`.

Allowed action: build strict exports and review queues only.

Forbidden action: no AMO writeback.

### Bucket 1: Manual Review / Quarantine

Rows enter this bucket when any of these are true:

- No contentful sales context.
- Latest meaningful call is `service_call`, `existing_client_progress`, or `technical_call` and service policy does not explicitly allow live sales writeback.
- Missing AMO contact ID.
- Multiple AMO contact IDs.
- Runtime AMO lookup returns multiple contacts.
- Runtime AMO lookup contact ID differs from source `AMO contact IDs`.
- CRM text quality fail-live risk exists.
- Passive/closing/vague next step requires manager confirmation.
- Review-precision population marker requires manager confirmation.

Allowed action: manual triage, upstream correction, future rebuild.

Forbidden action: live AMO writeback.

### Bucket 2: Strict AMO-Ready Export

A row can enter only if all are true:

- `Готово к записи в AMO=Да`.
- `CRM writeback policy=live_update_ready`.
- `AMO entity policy=update_existing_single_amo_contact`.
- Exactly one source `AMO contact IDs`.
- Latest writeback context is sales-call eligible.
- CRM quality gate has `passed=true`, `blocking_rows=0`.
- `population_recall.passed_for_live=true`.
- `crm_text_quality.passed_for_live=true`, `blocking_rows=0`.

Allowed action: offline preview and real-tunnel dry-run.

Forbidden action: live writeback without real-tunnel dry-run and staged approval.

### Bucket 3: Real-Tunnel Dry-Run Eligible

A row enters only after a dry-run with live AMO lookup returns:

- `mode=dry_run`.
- `live_write=false`.
- `offline_preview=false`.
- `status=dry_run`.
- `reason=live_write_not_confirmed`.
- Exactly one resolved runtime AMO contact.
- Runtime contact ID equals source `AMO contact IDs`.
- Updated fields are exactly the five allowed AI fields.
- `failed=0` for the dry-run batch.

Rows with `multiple_exact_contacts_in_amo`, `contact_id_mismatch_with_source_amo_contact_ids`, missing contact, failed lookup, or target-field violation move back to Bucket 1.

Allowed action: select a staged live bucket.

Forbidden action: full writeback or writing skipped rows.

### Bucket 4: Staged Live Pending

A live batch is a bounded subset of Bucket 3. The batch size must be explicit in the audit package name and input file.

Examples from current loop:

- `stage20`: first 20 rows.
- `stage40`: 40-row strict pool after Stage69 preflight, usually executed as live20 + readback + remaining20.
- `stage50/full`: not allowed until all stage-specific gates and readback criteria pass.

Required before live:

- Same input CSV used by quality summary and dry-run.
- Stage15 summary still `passed=true`, `crm_quality_writeback_ready=true`.
- CRM writeback quality summary `passed=true`.
- `population_recall.passed_for_live=true`.
- `crm_text_quality.passed_for_live=true`, `blocking_rows=0`.
- Live guard can verify `summary.input` equals actual `--input`.
- Dry-run report for this exact input has 0 failed and no skipped rows in the selected live subset.
- Explicit live confirmation string is present: `WRITE_AMO_LIVE`.

Allowed action: one live batch only.

Forbidden action: expanding the input after approval, reusing a green summary from another CSV, or running a larger batch than approved.

### Bucket 5: Written Pending Readback

Rows enter after live report records:

- `mode=live_write`.
- `status=written`.
- 0 failed.
- 0 skipped, unless the audit explicitly permits skipped rows to remain excluded.
- Updated fields are exactly the five allowed AI fields.

Allowed action: readback only.

Forbidden action: proceeding to the next live batch before readback passes.

### Bucket 6: Readback Green

Rows enter only after AMO readback fetches the actual stored five AI fields and re-runs CRM text checks.

Readback must show:

- `passed=true`.
- `evaluated_rows` equals written rows.
- `blocking_rows=0`.
- `failed_rows=0`.
- `empty_auto_history=0` or equivalent no-empty risk count.
- `lossy_ellipsis_truncation=0`.
- `duplicate_label_and_count=0`.
- `protected_field_hits=0`.
- `low_value_marker_in_written_history=0`.
- `closure_next_step_requires_downgrade=0`.
- `priority_next_step_conflict=0`.
- `vague_next_step=0`.
- `stale_uniform_followup_date=0`.
- `strong_negative_objection_* = 0`.

Allowed action: submit the next staged batch for audit.

Forbidden action: treating readback-green for one batch as authorization for all remaining rows.

## Gate Sequence

### Gate A: Source and Stage15

Pass conditions:

- Post-backfill source paths are named in export summary.
- `Stage15 passed=true`.
- `readiness.crm_quality_writeback_ready=true`.
- No fallback to `stable_runtime/CANONICAL_EXPORT.txt` unless the pointer itself is verified as current strict post-backfill export.

Blocks:

- Stale source pointer.
- Bot-only readiness mistaken for CRM readiness.
- Missing Stage15 summary.

### Gate B: CRM Writeback Quality

Pass conditions:

- `passed=true`.
- `blocking_rows=0`.
- `decision_counts.allow` equals rows in the proposed strict input.
- Frozen corpus `failures=0`.
- `population_recall.passed_for_live=true`.
- High-precision uncovered marker rows = 0.
- `crm_text_quality.passed_for_live=true`.
- `crm_text_quality.blocking_rows=0`.

Blocks:

- Any P0/P1/P2 fail-live risk.
- Any detector/frozen corpus failure.
- Any high-precision population marker not covered by detector.
- Any mismatch between summary input and actual input.

### Gate C: Offline Preview

Pass conditions:

- All rows produce payloads.
- Target fields are exactly the five AI fields.
- No protected fields in update payload.
- No empty payload rows.
- No CRM text quality fail-live class in payload.

Use: payload shape check only.

Limit: does not prove AMO contact matching and cannot authorize live writeback.

### Gate D: Real-Tunnel Dry-Run

Pass conditions:

- Runtime DB/tunnel preflight succeeds.
- `offline_preview=false`.
- `live_write=false`.
- `failed=0`.
- Selected live subset has only `status=dry_run`, `reason=live_write_not_confirmed`.
- Runtime AMO contact ID matches the source single AMO ID.
- Target fields remain exactly the five AI fields.

Blocks:

- Multiple exact AMO contacts.
- Contact ID mismatch.
- Missing contact where policy is update-only.
- Runtime DB/tunnel failure.
- Any failed row.

### Gate E: Live Write

Pass conditions:

- Gates A-D are green for the exact input.
- Batch size is staged and explicit.
- `--execute-live-write` and `--live-confirmation WRITE_AMO_LIVE` are intentionally supplied by an authorized operator.
- Dry-run report for the same input is attached to the audit package.

Blocks:

- No live confirmation.
- Missing or failed quality summary.
- Summary/input mismatch.
- Attempt to combine `--offline-preview` with live write.
- Any per-row live guard reason.

### Gate F: Post-Writeback Readback

Pass conditions:

- AMO fields are fetched after live write.
- Readback report evaluates every written row.
- CRM text detector runs against the fetched fields.
- Blocking rows = 0, failed rows = 0.
- Target fields are still exactly the five AI fields.

Blocks:

- Any readback failure.
- Any hard CRM text class stored in AMO.
- Any protected-field write or unexpected field update.
- Any row count mismatch.

## Forbidden Classes for Live Writeback

The following classes block live writeback until fixed, excluded, or routed to manual review:

- Q1: lossy ellipsis/truncation in any CRM text field, including internal `...` or `…`, not only end-of-field.
- Q2: duplicate raw label plus count artifact, e.g. `летний лагерь | летний лагерь: 14`.
- Q3 strong-negative conflict: `неактуально` or equivalent strong negative emitted as current without date/current-vs-historical framing, especially with active sales priority/next step.
- Q4: priority / probability / next-step contradiction.
- Q4b: stale uniform recommended follow-up date derived from run date instead of next-step semantics.
- Q4c: vague/passive/closing next step such as `не беспокоить`, `отменить`, `ждать обращения`, `связаться позже`, `связаться в мае`, unless downgraded/routed to manual review.
- Q6: missing or failed post-writeback readback gate.
- A1/A3: non-conversation, wrong number, no meaningful EdTech request, no contentful dialogue.
- A2: out-of-domain B2B/vendor/government/HR/carrier/service request.
- A4: service/existing-client/technical context promoted as sales lead writeback.
- B6/C9/F9: orphan, no exact AMO match, ambiguous AMO match, or undefined create/update policy.
- C1/F5: detector recall drift where population markers show high-precision uncovered risks.
- C4/F008: wrong target fields or protected-field write attempt.
- C5: third-party phone in CRM text.
- C7: stale source pointer drift.
- C8/F8: claiming class closure from self-seeded corpus only.
- H4: secrets/token leakage in audit or runtime artifacts.

Soft counters that do not block by themselves, but must be reported:

- C12/history chronology overlap when AMO payload suppresses redundant chronology and summary overlap is 0.
- Q5 manager UX too verbose when no dialogue dump, no truncation, and hard CRM text risks are 0.
- Third-party FIO in internal CRM for current Foton tenant, if tenant policy is `warn_internal_crm`; this becomes hard for SaaS/bot-facing surfaces.
- Rolling closure `monitoring/can_claim_closed=false`; this blocks closure claims, not necessarily the next staged dry-run.

## Stage50 / Full Writeback Checklist

Before stage50 or any larger/full writeback, verify all items below for the exact input file:

1. Source and export identity

- Input is from the strict post-backfill layer.
- Export summary references the expected canonical DB and client chains CSV.
- No old April export pointer is used.
- Input SHA/path is recorded and matches the quality summary.

2. Stage15 and CRM quality

- Stage15 summary `passed=true`.
- `readiness.crm_quality_writeback_ready=true`.
- CRM quality summary `passed=true`.
- `blocking_rows=0`.
- Frozen corpus failures = 0.
- Population recall `passed_for_live=true`.
- `crm_text_quality.passed_for_live=true`.
- `crm_text_quality.blocking_rows=0`.

3. Row eligibility

- Every row has `Готово к записи в AMO=Да`.
- Every row has `CRM writeback policy=live_update_ready`.
- Every row has `AMO entity policy=update_existing_single_amo_contact`.
- Every row has exactly one source `AMO contact IDs`.
- Latest call type is `sales_call`.
- Manual-review rows are excluded.
- Rows flagged by review-precision population markers have explicit manager confirmation or remain excluded.

4. Payload shape

- Only five AI fields will be updated.
- Protected fields are not update targets.
- No empty auto history.
- No lossy ellipsis/truncation.
- No duplicate label+count artifacts.
- No closure/vague next-step conflicts.
- Recommended follow-up dates are semantically derived and not uniform run-date defaults.

5. Real-tunnel dry-run

- Dry-run is real-tunnel, not offline preview.
- `failed=0`.
- Selected live subset has `status=dry_run` only.
- Contact IDs from runtime AMO lookup match source AMO IDs.
- Multi-contact and mismatch rows are excluded.

6. Live batch approval

- Batch size is named and bounded.
- Audit package asks for exactly one next stage.
- Live command includes explicit confirmation and both quality summaries.
- No operator broadens the CSV after audit approval.

7. Readback from previous batch

- Previous live batch has readback `passed=true`.
- `evaluated_rows` equals written rows.
- `blocking_rows=0`, `failed_rows=0`.
- Stored AMO fields still pass CRM text detector.

8. Observability

- Writeback report, readback report, quality summaries, and skipped-row reasons are archived in the audit pack.
- Counts are recorded: input rows, dry-run rows, skipped rows, failed rows, written rows, readback evaluated rows, readback blockers.
- Mismatch skip rate is monitored as source/AMO drift indicator.

## Full Writeback Rule

A full writeback is allowed only after repeated staged batches show:

- At least two consecutive staged live batches with 0 failed, 0 unexpected skipped, and green readback.
- No new P0/P1/P2 class in the latest audit.
- Population recall remains `passed_for_live=true`.
- Rolling closure is no longer dependent on a single self-seeded audit cycle for any class being claimed closed.
- Manual review and ambiguous-contact buckets are not silently reintroduced.
- The operator explicitly approves the larger scope after reading the latest readback evidence.

Until those conditions hold, use the staged loop only.

## Operator Stop Conditions

Stop immediately and do not run live writeback if any of these appear:

- Gate summary missing, failed, or references another input.
- Runtime DB/tunnel preflight fails.
- Any dry-run row in the proposed live subset is `skipped` or `failed`.
- Runtime AMO contact differs from source AMO ID.
- Any row would write outside the five target AI fields.
- Any protected field appears in update payload.
- Any Q1/Q2/Q3-strong/Q4/Q4b/Q4c/Q6 fail-live risk appears.
- Readback from prior batch is missing or not green.
- The requested live batch is larger than the audited input.
- The task requester asks to bypass gates because rows were already reviewed informally.

## Current Recommended Posture

As of the 2026-05-10 evidence, the safe posture is:

- Stage20 already written: keep as-is; no automatic overwrite solely for UX defects.
- Further writes: require strict post-backfill export, real-tunnel dry-run, staged live batch, and readback gate.
- Stage50/full: not authorized by documentation alone; must pass the checklist above using the exact stage input and latest reports.
- Skipped/ambiguous rows: remain manual review until upstream source/AMO identity is corrected and re-gated.
