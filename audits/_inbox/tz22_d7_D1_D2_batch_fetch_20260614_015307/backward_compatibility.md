Backward compatibility

Default behavior:
- TALLANTO_BATCH_FETCH unset/OFF keeps the previous full phone scan and full related-block loading.
- AMO_LEADS_BATCH_FETCH unset/OFF keeps the previous per-lead fetch loop.

NEG:
- Tallanto OFF mock count remains 15 search calls for the fixture phone.
- AMO OFF mock count remains 2 fetch_lead calls for two embedded leads.

Changed behavior when enabled:
- TALLANTO_BATCH_FETCH=1 returns after the first Tallanto phone hit and skips live-card-unused related blocks in live_card_only mode.
- AMO_LEADS_BATCH_FETCH=1 fetches embedded lead ids through one batch collection request for up to 50 ids per chunk.

Unaffected:
- build_tallanto_live_card output remains unchanged for the tested context.
- resolve_target_lead selected lead remains unchanged for the tested embedded-leads fixture.
