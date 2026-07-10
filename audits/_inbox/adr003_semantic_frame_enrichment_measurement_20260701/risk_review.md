# Risk Review

## Live/write risk

Live bot, AMO, Tallanto, CRM and Telegram sends were not touched.

Outputs were written only under `audits/_inbox/`.

## Measurement risks

- Paired enrichment proves a narrow invariant by construction: route/text/safety/checklist are copied from OFF transcript and must remain unchanged. This is correct for metadata no-op, but not enough for decision policy.
- Independent full ON draft runs are nondeterministic and can show route/text diffs even without a behavior bug. They must not be used as strict no-op proof.
- `SemanticFrame` quality remains unproven without manual expected-frame gold.
- Full131 paired enrichment passed no-op, but frame-vs-current-route/P0 mismatches remain high enough that active behavior must stay blocked until gold labelling/regreade.
- Gold queue JSONL/CSV contains client/bot transcript text and must remain local/ignored. Do not force-add those queue files into git unless a separate PII-reviewed archival decision is made.

## Guardrails added

- Report now rejects partial enrichment.
- Report now rejects enrichment with non-frame ON calls.
- Report now rejects mismatched input turns.
- Report and gold queue no longer treat the pilot-wide `manager_approval_required` safety flag as route-handoff; route-handoff is based on `bot_route in {manager_only, draft_for_manager}`.
- Report and gold queue require strict boolean `frame.must_handoff`; string values no longer pass through `bool(...)`.
- Gold queue writes `input_status`, so OFF/no-frame input cannot be mistaken for "zero mismatches".
- Gold queue writes `pii_risk`/`pii_risk_rows` summary fields.
- Manifest no longer presents same-payload `TELEGRAM_SEMANTIC_FRAME_SHADOW=1` as the canonical Stage 1 command.
- Parallel enrichment preserves input order and has regression coverage for out-of-order futures, frozen fields, and thread-safe LLM-call counting.
- Gold queue builder turns the mismatch set into a reviewable manual queue instead of treating current route/P0 detectors or the frame as automatic truth.
