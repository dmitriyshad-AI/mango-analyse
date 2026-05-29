# Risk Review

Safety risk:
- Main risk is accidental early release of real P0. Mitigation: hard latch codes `legal`, `legal_threat`, and `payment_dispute` are excluded from autonomous release and covered by negative tests.

Business risk:
- Soft refund/complaint latches can now release after 5 neutral turns. This is intended to avoid losing long autonomous dialogues after a false or resolved P0 signal. It should be monitored in the next long-dialog harness.

Regression risk:
- Historical P0 risk flags are no longer reintroduced after an explicit latch release. This is necessary for release to have behavioral effect; otherwise `handoff_state` remains required even with inactive latch.
- Existing P0 and draft-provider targeted suites passed.

Operational risk:
- No live-send, CRM/AMO/Tallanto, KB rebuild, ASR, or 212-dialog run was performed.
