# Risk Review

## Primary risks checked

- Raw payload leakage: expanded scrub list and test coverage for WhatsApp/Wappi nested payload keys.
- Wrong identity merge: ambiguous phone matches do not create ordinary `phone` links, avoiding resolver auto-union.
- Live side effects: dry-run default, safety contract, no network/Tallanto/AMO/CRM writes.
- Synthetic data: MAX is explicitly blocked/no-source; no fake events are generated.

## Residual risks

- Real timeline DB used for lookup is an ignored historical artifact and lacks some newer schema tables; B3 used direct read-only SQL lookup rather than treating that DB as fully migrated.
- WhatsApp source brand is passed as a run-level brand hint; source archive itself does not prove brand per message.
- Full import apply on the real 40k message archive was not executed in this stage; apply/idempotency is covered on fixtures.
