# Semantic review

This block changes profile child grouping logic, not customer-facing text.

Semantic conclusion:

- Automatic string/alias name identity checks were removed as requested by TZ30.
- Quality gate for names is manual regread over full names by the architect.
- Current microprobe gives a manual-review queue:
  - raw local: `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz30_microprobe_v4_20260614_142934/name_review_diagnostics.local.jsonl`
  - anonymized low-confidence multi-named focus: `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz30_microprobe_v4_20260614_142934/low_confidence_multi_named.anonymized.jsonl`

Do not treat this as approved for full 7512/prod until the manual regread is complete.
