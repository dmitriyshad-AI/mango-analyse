# Semantic review

Verdict: PASS

Artifact and audience:
- Customer-facing Telegram draft text and manager handoff text.

What passed:
- Timeout path does not produce empty or stale customer text; it fails into provider-runtime manager review.
- `client-safe` internal jargon is blocked at source for known rules-engine no-fact payloads and at output sanitizer as backstop.
- Benign presale refund explanation no longer poisons the next harmless course-selection turn.
- Real refund/P0 latch remains manager-only via existing hard-latch regression.
- Cross-brand mixed draft now gets a safe non-comparative phrase: "Это отдельные организации..." and does not name Foton/UNPK conditions.

Blocking issues:
- None found in changed surfaces.

Non-blocking risks:
- Sanitizer removes the `client-safe` sentence broadly. If the surrounding text depended on that sentence for meaning, the final fallback may be generic. This is safer than leaking internal wording.
- Benign refund latch calibration depends on `had_hard_p0_claim`; if memory fails to set that flag for a real dispute, current-turn hard P0 regex/classifier still protects explicit claims, but a very indirect follow-up may need future review.

Regression coverage added:
- Understand timeout -> `provider_runtime`, draft model not called.
- Normal understanding JSON still parses.
- `client-safe` jargon stripped; clean text preserved.
- Benign refund latch next harmless turn answers from fact.
- Hard refund latch remains manager-only.
- Cross-brand mixed draft returns canonical separation phrase.

