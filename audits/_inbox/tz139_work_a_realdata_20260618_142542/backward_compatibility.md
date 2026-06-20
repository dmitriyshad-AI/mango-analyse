# Backward Compatibility

- `exact_phone_single` remains `strong_unique` and produces `identity_status=strong`.
- `no_exact_phone_match` intentionally changes from old `strong` behavior to `partial` customer status plus `unmatched` Tallanto match.
- Shared family-phone with multiple Tallanto IDs in one source row intentionally changes from one merged customer to one customer per Tallanto student plus open conflict.
- Shared family-phone phone links intentionally change from `strong_unique` to `ambiguous`.
- No schema migration was added in this patch; Work A existing `customer_id_mappings` schema remains unchanged.
