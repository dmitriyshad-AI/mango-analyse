# Risk review

Safety invariants checked:
- P0 hard latch remains manager-only.
- Cross-brand output keeps brand separation and does not compare conditions.
- Provider timeout fails closed to manager review.
- Internal jargon is not exposed to customers.

What tests do not prove:
- They do not replay the full four raw regread runs. Claude #1 should still regrade raw outputs after this patch.
- They do not prove every possible internal phrase is sanitized; only the confirmed `client-safe` class is covered.
- They do not exercise live Codex timeout under real CLI load, only the subprocess timeout behavior.

Classification:
- hp_topic_change hang: infrastructure/provider_runtime bug.
- hp_when_start client-safe leak: object bug in output source plus missing sanitizer backstop.
- ov_refund_hypo sticky latch: object bug in P0 memory interpretation.
- ov_xbrand_unpk delayed canonical phrase: object bug in early brand guard output text.

