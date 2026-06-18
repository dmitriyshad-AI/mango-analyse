# TZ139 Work B2 Tallanto Payments - Implementation Notes

Status: formal_pass, semantic_self_review_done, STOP for Claude regrade.

## Scope

- Added `scripts/import_tallanto_payments_to_timeline.py`.
- Extended raw-payload scrub keys in `src/mango_mvp/customer_timeline/store.py`.
- Added B2 tests in `tests/test_import_tallanto_payments_to_timeline.py`.
- Extended store NEG in `tests/test_customer_timeline_store.py`.

## Implementation

- Input is a local JSON snapshot or stdin from read-only `crm_call.sh tallanto_select`.
- The importer itself does not call `crm_call.sh`, network, subprocess, AMO, Tallanto, ASR, RA, or LLM.
- Default mode is dry-run. `--apply` writes only the configured local `customer_timeline.sqlite`.
- Supported modules:
  - `most_finances` -> `tallanto_payment`
  - `most_abonements` -> `tallanto_abonement`
  - `most_class` -> class lookup only
- Stored event/opportunity fields are a whitelist projection: ids, dates, direction/status/type, amount, currency, visits, filial, class labels.
- Free text `description`, `contact_notice`, `internal_notice`, raw MCP response wrappers, and raw payload keys are not stored.
- Payment/abonement records create no `bot_context_chunks`; exact amounts and balances are stored only in events/opportunities.
- Existing customers are resolved read-only by `identity_links.link_type='tallanto_student_id'`.
- Ambiguous Tallanto ids create `tallanto_identity_ambiguous` conflicts and do not first-match merge.

## Real-Data Read-Only Probe

Source access:

- `crm_call.sh list` confirmed only read-only tools.
- `tallanto_fields` confirmed fields for `most_finances`, `most_abonements`, `most_class`.
- Real sample used `tallanto_select` with `limit=50` for each module.
- Snapshot was passed through stdin/in-memory; raw JSON was not saved to repo or audit pack.

Dry-run results:

- Records loaded: 100
- Payment events: 50
- Abonement events: 50
- Class lookup rows: 50
- Rejected: 0
- Bot context chunks: 0
- Bot-safe amount leaks: 0
- Source path in report: stdin
- `write_product_timeline_db`: false

Temp apply idempotency probe:

- Target: `/tmp/mango_tz139_b2_apply_rerun/customer_timeline.sqlite`
- First apply: validation_ok=true, created=491, duplicate=6
- Second apply: validation_ok=true, duplicate=12, updated=291
- Row counts after second apply:
  - customers: 97
  - events: 100
  - tallanto_payment events: 50
  - tallanto_abonement events: 50
  - opportunities: 100
  - bot_context_chunks: 0
  - bot_safe_amount_leaks: 0
  - ingestion_runs: 1

Note: second apply reports `updated` from the existing generic store upsert/mapping behavior, but row counts and event dedupe stayed stable. No duplicate payment/abonement events were created.

## STOP

B2 is stopped after this audit pack and commit. B3 is not started.
