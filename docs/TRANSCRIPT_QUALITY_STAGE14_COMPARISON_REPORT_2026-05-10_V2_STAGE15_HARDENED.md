# Stage 14 Quality Comparison Report

Generated at: `2026-05-09T23:12:53.029274+00:00`

## Decision

- Acceptance passed: `True`
- Audit sample rows: `200`
- Over-sanitization candidates: `250`
- Residual risk samples: `0`

## Key Metric Deltas

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `kb_no_live_revenue_risk` | 0 | 0 | 0 |
| `kb_bot_ready_money_or_terms` | 552 | 0 | -552 |
| `kb_ideal_answer_brand_risk` | 13 | 0 | -13 |
| `kb_bot_safe_answer_brand_risk` |  | 0 |  |
| `kb_bot_safe_answer_personal_data_risk` |  | 0 |  |
| `rop_p0_no_live_or_artifact` | 0 | 0 | 0 |
| `rop_revenue_risk_no_live_or_artifact` | 0 | 0 | 0 |
| `rop_bot_candidate_money_or_terms` | 85 | 0 | -85 |
| `rop_bot_safe_answer_brand_risk` |  | 0 |  |
| `rop_bot_safe_answer_personal_data_risk` |  | 0 |  |
| `kb_raw_ideal_answer_brand_risk` |  | 76 |  |
| `kb_raw_ideal_answer_money_or_terms` |  | 1927 |  |

## Acceptance Checks

- `required_kb_columns_present`: `True`
- `required_rop_columns_present`: `True`
- `bot_seed_safe_columns_present`: `True`
- `no_residual_bot_safe_risks`: `True`
- `kb_no_live_revenue_risk_zero`: `True`
- `rop_p0_no_live_or_artifact_zero`: `True`
- `rop_revenue_no_live_or_artifact_zero`: `True`
- `kb_bot_ready_money_or_terms_zero`: `True`
- `rop_bot_candidate_money_or_terms_zero`: `True`
- `bot_ready_rows_have_safe_answer`: `True`
- `audit_sample_built`: `True`

## Sanitizer Metrics

- Bot-ready rows missing safe answer: `0`
- Bot seed safe-answer risks: `{'brand': 0, 'money_or_terms': 0, 'personal_data': 0}`
- ROP bot safe-answer risks: `{'brand': 0, 'money_or_terms': 0, 'personal_data': 0}`
- Top sanitizer flags: `{'deadline_redacted': 1667, 'person_name_redacted': 1159, 'refund_policy_redacted': 533, 'discount_terms_redacted': 399, 'price_redacted': 370, 'installment_terms_redacted': 230, 'percent_redacted': 157, 'brand_normalized': 135, 'email_redacted': 27, 'phone_redacted': 8}`

## Audit Sample Buckets

- `coverage_filler`: 48
- `money_terms_sanitized`: 34
- `brand_sanitized`: 25
- `bot_ready_clean_no_changes`: 20
- `legal_deadline_sanitized`: 19
- `rop_revenue_risk`: 15
- `rop_top_answer`: 15
- `installment_sanitized`: 13
- `rop_bot_draft`: 7
- `personal_data_sanitized`: 4

## Interpretation

Stage 13 removed unsafe bot/ROP leakage without changing raw source fields. Stage 14 does not prove semantic perfection; it proves that safety gates are measurable, residual bot-safe risks are zero, and a stratified review package exists to check usefulness and over-sanitization.

## Next Step

Run GPT/Claude or ROP audit on `audit_sample.csv` and `over_sanitization_candidates.csv`. If accepted, proceed to stage 15: wire these gates into the permanent pipeline before KB/ROP/bot/CRM exports.
