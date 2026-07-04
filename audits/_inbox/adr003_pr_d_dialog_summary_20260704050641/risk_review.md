# Risk review

## Main risks

1. Summary treated as fact.
   - Mitigation: PR-D writes only `conversation_summary_short`; it does not write confirmed slots/facts or customer-safe fact stores.

2. PII in memory summary.
   - Mitigation: `_dialog_summary_candidate` rejects phone/email through existing direct-path support regexes; prompt/memory tests cover phone/email.

3. Foreign-brand bleed.
   - Mitigation: `_dialog_summary_candidate` rejects obvious foreign brand tokens for Foton/UNPK; Wappi history also filters accumulated summary before prompt insertion.

4. Unsupported model-made numbers in summary.
   - Mitigation: summaries with long digit groups, numeric dates, percent words, currency markers, or month dates are rejected fail-closed.

5. New hidden behavior at OFF.
   - Mitigation: prompt field and metadata parsing are gated; OFF one-turn direct-path smoke shows no candidate and empty summary.

6. Extra LLM call.
   - Mitigation: summary is an additive field in the same direct-path JSON payload; no new caller/model client was introduced.

## Residual risk

The summary is model-authored. It can omit nuance even when safe. This is acceptable only for shadow/default-OFF measurement until long-dialogue semantic review passes.
