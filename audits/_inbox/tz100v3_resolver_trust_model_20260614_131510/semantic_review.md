# Semantic review

This block changes measurement and diagnostics, not customer-facing text.

Semantic conclusion:

- Do not weaken name-veto rules based on this run.
- `merge_confidence` is not a reliable trust signal yet: known-bad cases include `high`.
- The correct next step is manual classification of local name-veto diagnostics by the architect.

Manual classification source:

`/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz100_microprobe_v3_20260614_124341/name_veto_diagnostics.local.jsonl`

This file intentionally remains local/ignored and is not included in git/audit pack because it contains raw child-name spellings.
