# Backward Compatibility

- Existing contracts are unchanged; `TimelineEventType.TALLANTO_PAYMENT`, `TimelineEventType.TALLANTO_ABONEMENT`, and `OpportunityType.TALLANTO_COURSE` already existed.
- Existing importers and read APIs are not modified.
- SQLite schema is unchanged.
- Store scrubber only removes additional raw Tallanto payload key names; safe projected event/opportunity fields remain unaffected.
- Default B2 mode is dry-run and does not create a target timeline DB.
- `--apply` is opt-in and scoped to the configured local `customer_timeline.sqlite`.

NEG coverage:

- Dry-run stdin test asserts target DB is not created.
- Safety test asserts no subprocess/network/runtime/live-write path in the B2 importer.
- Store raw-payload test asserts Tallanto raw keys are stripped from persisted JSON.
- Apply/idempotency test asserts repeated import does not duplicate payment/abonement events and creates no bot context chunks.
