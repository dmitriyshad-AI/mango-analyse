# Risk review

Risk level: medium-low.

Checked risks:
- `unsupported_promise` is not weakened: the flag remains on the original unsafe draft.
- Approved numeric template is re-verified without falling to generic fallback.
- Statistics and price questions are not swallowed by the refusal template.

Known uncovered risk:
- A true P0 complaint combined with a result-guarantee phrase still depends on the surrounding high-risk guards and semantic regreyde. This commit does not change refund/payment/legal latch behavior.

