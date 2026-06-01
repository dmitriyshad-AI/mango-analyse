# Semantic Review

Status: local semantic self-review passed for Block 1.1 scope; independent Claude regрейд still required before any pilot/live claim.

Checks:
- Real refund/payment/legal/complaint history is sticky as safety context even after autonomous latch release. This prevents the unsafe semantic drift where a later "жду менеджера" can receive a presale refund answer.
- Clean presale refund remains useful: hypothetical pre-payment questions can still answer with "остаток неистраченных средств" from the client-safe fact.
- Tax informational yield keeps useful client-safe tax facts, but rejects generated extrapolations such as unbacked two-child amounts and document/contract scope substitution.
- Complaint/refund P0 remains manager-handled and does not collect child data, phone, email, contract, amount or reason.
- Brand behavior was not changed; cross-brand guards remain covered by existing tests.

Known semantic risk:
- The full customer-facing behavior still needs Claude raw-transcript regрейд on the target scenario set. This pack is a formal/local semantic pass, not a launch approval.
