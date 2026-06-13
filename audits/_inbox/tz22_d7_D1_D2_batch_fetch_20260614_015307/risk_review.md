Risk review

Primary risks addressed:
- Tallanto live-card path making unnecessary requests for blocks not used in the card.
- Phone search continuing across all Tallanto phone fields after a usable contact is already found.
- AMO resolve_target_lead making N fetch_lead requests for embedded lead ids.

Guardrails:
- TALLANTO_BATCH_FETCH defaults OFF.
- AMO_LEADS_BATCH_FETCH defaults OFF.
- OFF NEG tests assert old request counts.
- ON tests assert lower request counts and identical selected/card output.
- No live external requests were executed.

Known gaps:
- Tallanto class id fetching still uses per-class get_entry_by_id when class ids are unique; no unconfirmed batch endpoint was guessed.
- AMO batch preserves input order for fetched leads, but missing ids are naturally omitted like a failed/not returned batch item.
